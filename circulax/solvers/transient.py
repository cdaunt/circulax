"""Transient solvers to be used with Diffrax."""

from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
from diffrax import AbstractAdaptiveStepSizeController, AbstractSolver, ConstantStepSize
from jax.typing import ArrayLike

from circulax.solvers.circuit_diffeq import circuit_diffeqsolve

try:
    from klujax import free_numeric as _klujax_free_numeric
except ImportError:
    _klujax_free_numeric = None


def free_numeric(handle, dependency=None):  # noqa: ANN001, ANN201, ARG001
    """Dispatch free to the correct backend based on handle type.

    ``klujax.KLUHandleManager`` exposes ``.close()``, which calls the backend's FFI free
    function.  Duck-typing is used so handles are freed correctly without ``isinstance``
    checks against concrete backend types.
    """
    if hasattr(handle, "close"):
        handle.close()
        return
    if _klujax_free_numeric is not None:
        _klujax_free_numeric(handle)


from circulax.solvers.assembly import (
    assemble_residual_only_complex,
    assemble_residual_only_real,
    assemble_system_complex,
    assemble_system_real,
    min_active_tau,
)
from circulax.solvers.linear import (
    DAMPING_EPS,
    DAMPING_FACTOR,
    GROUND_STIFFNESS,
    CircuitLinearSolver,
)


def _unpack_args(args):
    """Unpack solver ``args`` into ``(component_groups, num_vars, hist_t, hist_y)``.

    ``hist_t``/``hist_y`` are threaded in as a trailing pair by
    ``circuit_diffeq.py``'s ``_circuit_loop`` only when the circuit has
    delayed components (``record_history=True``); otherwise ``args`` is the
    plain ``(component_groups, num_vars)`` 2-tuple and both are ``None``, so
    ``assemble_system_real``/``assemble_system_complex`` skip the delayed-read
    branch at zero cost.
    """
    if len(args) == 4:
        return args
    component_groups, num_vars = args
    return component_groups, num_vars, None, None


def _compute_history(component_groups, y_c, t, num_vars) -> ArrayLike:
    """Computes total charge Q at time t (Initial Condition)."""
    from circulax.solvers.assembly import _assemble_osdi_group, _is_osdi

    is_complex = jnp.iscomplexobj(y_c)
    total_q = jnp.zeros(
        2 * num_vars if is_complex else num_vars,
        dtype=jnp.float64 if is_complex else y_c.dtype,
    )

    for group in component_groups.values():
        if _is_osdi(group):
            _, q_l, _ = _assemble_osdi_group(
                y_c, group, alpha=1.0, dt=1.0, residual_only=True,
            )
        else:
            v_locs = y_c[group.var_indices]
            _, q_l = jax.vmap(lambda v, p: group.physics_func(y=v, args=p, t=t))(v_locs, group.params)

        if is_complex:
            total_q = total_q.at[group.eq_indices].add(q_l.real)
            total_q = total_q.at[group.eq_indices + num_vars].add(q_l.imag)
        else:
            total_q = total_q.at[group.eq_indices].add(q_l)
    return total_q


# --- Transient Solvers ---


class VectorizedTransientSolver(AbstractSolver):
    """Transient solver that works strictly on FLAT (Real) vectors.

    Delegates complexity handling to the 'linear_solver' strategy.
    """

    linear_solver: CircuitLinearSolver
    newton_rtol: float = 1e-5
    newton_atol: float = 1e-5
    newton_max_steps: int = 20

    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms) -> int:  # noqa: D102
        return 1

    def init(self, terms, t0, t1, y0, args):
        return (y0, 1.0)

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: ANN201, D102
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        dt = t1 - t0
        y_prev_step, dt_prev = solver_state

        is_complex = getattr(self.linear_solver, "is_complex", False)

        rate = (y0 - y_prev_step) / (dt_prev + 1e-30)
        y_pred = y0 + rate * dt

        y_c = y0[:num_vars] + 1j * y0[num_vars:] if is_complex else y0
        q_prev = _compute_history(component_groups, y_c, t0, num_vars)

        def newton_update_step(y, _) -> jax.Array:
            if is_complex:
                total_f, total_q, all_vals = assemble_system_complex(y, component_groups, t1, dt, hist_t=hist_t, hist_y=hist_y)
                ground_indices = [0, num_vars]
            else:
                total_f, total_q, all_vals = assemble_system_real(y, component_groups, t1, dt, hist_t=hist_t, hist_y=hist_y)
                ground_indices = [0]

            residual = total_f + (total_q - q_prev) / dt
            for idx in ground_indices:
                residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])

            sol = self.linear_solver._solve_impl(all_vals, -residual)  # noqa: SLF001
            delta = sol.value
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
            return y + delta * damping

        solver = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
        sol = optx.fixed_point(newton_update_step, solver, y_pred, max_steps=self.newton_max_steps, throw=False)

        y_next = sol.value
        y_error = y_next - y_pred

        # Map result to Diffrax
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )

        return y_next, y_error, {"y0": y0, "y1": y_next}, (y0, dt), result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class FactorizedTransientSolver(VectorizedTransientSolver):
    """Transient solver using a Modified Newton (frozen-Jacobian) scheme.

    At each timestep the system Jacobian is assembled and factored once at a
    predicted state, then reused across all Newton iterations. Compared to a
    full Newton-Raphson solver this trades quadratic convergence for a much
    cheaper per-iteration cost — one triangular solve instead of a full
    factorisation — making it efficient for circuits where the Jacobian varies
    slowly between steps.

    Convergence is linear rather than quadratic, so ``newton_max_steps`` is set
    higher than a standard Newton solver would require. Adaptive damping
    ``min(1, 0.5 / max|δy|)`` is applied at each iteration to stabilise
    convergence in stiff or strongly nonlinear regions.

    Both real and complex assembly paths are supported; the complex path
    concatenates real and imaginary parts into a single real-valued vector,
    allowing purely real linear algebra kernels to be reused for
    frequency-domain-style analyses.

    Requires a :class:`~circulax.solvers.linear.KLUSplitFactorSolver` as the
    ``linear_solver`` — use ``analyze_circuit(..., backend="klu_split_factor")``.
    """

    newton_max_steps: int = 100

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: ANN201, D102
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        dt = t1 - t0
        y_prev_step, dt_prev = solver_state

        is_complex = getattr(self.linear_solver, "is_complex", False)

        rate = (y0 - y_prev_step) / (dt_prev + 1e-30)
        y_pred = y0 + rate * dt

        y_c = y0[:num_vars] + 1j * y0[num_vars:] if is_complex else y0
        q_prev = _compute_history(component_groups, y_c, t0, num_vars)

        if is_complex:
            _, _, frozen_jac_vals = assemble_system_complex(y_pred, component_groups, t1, dt, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0, num_vars]
        else:
            _, _, frozen_jac_vals = assemble_system_real(y_pred, component_groups, t1, dt, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0]

        numeric_handle = self.linear_solver.factor_jacobian(frozen_jac_vals)

        def newton_update_step(y: jax.Array, _: Any) -> jax.Array:
            if is_complex:
                total_f, total_q = assemble_residual_only_complex(y, component_groups, t1, dt, hist_t=hist_t, hist_y=hist_y)
            else:
                total_f, total_q = assemble_residual_only_real(y, component_groups, t1, dt, hist_t=hist_t, hist_y=hist_y)

            residual = total_f + (total_q - q_prev) / dt
            for idx in ground_indices:
                residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])

            sol = self.linear_solver.solve_with_frozen_jacobian(-residual, numeric_handle)
            delta = sol.value
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
            return y + delta * damping

        solver = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
        sol = optx.fixed_point(
            newton_update_step,
            solver,
            y_pred,
            max_steps=self.newton_max_steps,
            throw=False,
        )

        free_numeric(numeric_handle)

        y_next = sol.value
        y_error = y_next - y_pred

        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )

        return y_next, y_error, {"y0": y0, "y1": y_next}, (y0, dt), result


class RefactoringTransientSolver(FactorizedTransientSolver):
    """Transient solver with full Newton (quadratic) convergence using ``klujax.refactor``.

    At each timestep the Jacobian is factored once at the predicted state to allocate the
    numeric handle.  Each Newton iteration then calls ``klujax.refactor`` — which reuses
    the existing memory and fill-reducing permutation but recomputes L/U values for the
    current iterate J(y_k) — followed by a triangular solve.  This gives full quadratic
    Newton convergence at a fraction of the cost of re-factoring from scratch each iteration.

    Convergence is quadratic so ``newton_max_steps`` is set to 20, matching
    :class:`VectorizedTransientSolver`.  Adaptive damping ``min(1, 0.5 / max|δy|)``
    is applied at each iteration to stabilise convergence in stiff or strongly nonlinear
    regions.

    Requires :class:`~circulax.solvers.linear.KLUSplitQuadratic` as the ``linear_solver``
    — use ``analyze_circuit(..., backend="klu_split")``.
    """

    newton_max_steps: int = 20  # quadratic convergence; matches VectorizedTransientSolver default

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: ANN201, D102
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        dt = t1 - t0
        y_prev_step, dt_prev = solver_state

        is_complex = getattr(self.linear_solver, "is_complex", False)

        rate = (y0 - y_prev_step) / (dt_prev + 1e-30)
        y_pred = y0 + rate * dt

        y_c = y0[:num_vars] + 1j * y0[num_vars:] if is_complex else y0
        q_prev = _compute_history(component_groups, y_c, t0, num_vars)

        if is_complex:
            _, _, init_vals = assemble_system_complex(y_pred, component_groups, t1, dt, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0, num_vars]
        else:
            _, _, init_vals = assemble_system_real(y_pred, component_groups, t1, dt, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0]

        numeric_handle = self.linear_solver.factor_jacobian(init_vals)

        def newton_update_step(y: jax.Array, _: Any) -> jax.Array:
            if is_complex:
                total_f, total_q, all_vals = assemble_system_complex(y, component_groups, t1, dt, hist_t=hist_t, hist_y=hist_y)
            else:
                total_f, total_q, all_vals = assemble_system_real(y, component_groups, t1, dt, hist_t=hist_t, hist_y=hist_y)

            residual = total_f + (total_q - q_prev) / dt
            for idx in ground_indices:
                residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])

            if hasattr(self.linear_solver, "refactor_and_solve_jacobian"):
                sol, _ = self.linear_solver.refactor_and_solve_jacobian(all_vals, -residual, numeric_handle)
            else:
                refreshed_handle = self.linear_solver.refactor_jacobian(all_vals, numeric_handle)
                sol = self.linear_solver.solve_with_frozen_jacobian(-residual, refreshed_handle)
            delta = sol.value
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
            return y + delta * damping

        fpi = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
        sol = optx.fixed_point(
            newton_update_step,
            fpi,
            y_pred,
            max_steps=self.newton_max_steps,
            throw=False,
        )

        free_numeric(numeric_handle)

        y_next = sol.value
        y_error = y_next - y_pred

        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )

        return y_next, y_error, {"y0": y0, "y1": y_next}, (y0, dt), result


# ==============================================================================
# BDF2 — 2nd-ORDER COMPANION SOLVER
# ==============================================================================

# Variable-step BDF2 (Gear's Method) companion formulation.
# At each step the dQ/dt term is replaced by:
#   (α₀ Q_{n+1} + α₁ Q_n + α₂ Q_{n-1}) / h_n = 0
# where ω = h_n / h_{n-1} and
#   α₀ = (1 + 2ω)/(1 + ω),  α₁ = -(1 + ω),  α₂ = ω²/(1 + ω).
# For uniform steps: α₀ = 3/2, α₁ = -2, α₂ = 1/2.
# The effective Jacobian is J_eff = dF/dy + (α₀/h_n) dQ/dy.
# Step 0 falls back to Backward Euler automatically: h_nm1 is initialised to
# +inf so ω = h_n/inf = 0 (IEEE 754), giving α₀=1, α₁=-1, α₂=0, which
# reduces the BDF2 formula to the BE formula with no branching.


def _bdf2_preamble(y0, t0, h_n, solver_state, component_groups, num_vars, is_complex):
    """Compute BDF2 coefficients, history charges, predictor, and next solver_state.

    Args:
        y0: Current state at t0.
        t0: Current time.
        h_n: Current step size (t1 - t0).
        solver_state: 3-tuple ``(y_nm1, h_nm1, q_nm1)`` where ``y_nm1`` is the
            solution at ``t0 - h_nm1`` and ``q_nm1`` is Q(y_{n-1}) cached from the
            previous step.  On the first step ``h_nm1 = +inf`` so ``ω = 0``, which
            makes the coefficients ``(α₀, α₁, α₂) = (1, -1, 0)`` — identical to
            Backward Euler — with no branching required.
        component_groups: Compiled component groups.
        num_vars: Number of real state variables (half of flat vector for complex).
        is_complex: Whether the circuit uses complex arithmetic.

    Returns:
        Tuple ``(y_pred, alpha, make_residual, new_state)`` where:

        - ``y_pred``: 1st-order extrapolation predictor.
        - ``alpha``: effective Jacobian scaling α₀ (equals 1.0 on the first step).
        - ``make_residual(total_f, total_q)``: returns the discretised residual.
        - ``new_state``: updated solver_state for the next step.

    """
    y_nm1, h_nm1, q_nm1 = solver_state

    # Variable-step BDF2 coefficients; ω→0 when h_nm1=inf (first step) → BE.
    omega = h_n / (h_nm1 + 1e-30)
    alpha0 = (1.0 + 2.0 * omega) / (1.0 + omega)
    alpha1 = -(1.0 + omega)
    alpha2 = omega**2 / (1.0 + omega)
    alpha = alpha0  # equals 1.0 on the first step

    # Compute Q(y_n) at t0 — Q(y_{n-1}) is carried in solver_state to avoid recomputation.
    y_c0 = y0[:num_vars] + 1j * y0[num_vars:] if is_complex else y0
    q_n = _compute_history(component_groups, y_c0, t0, num_vars)

    # 1st-order predictor (same as BE; rate=0 on first step when h_nm1=inf)
    rate = (y0 - y_nm1) / (h_nm1 + 1e-30)
    y_pred = y0 + rate * h_n

    def make_residual(total_f, total_q):
        return total_f + (alpha0 * total_q + alpha1 * q_n + alpha2 * q_nm1) / h_n

    # Shift history; cache q_n as q_nm1 for the next step.
    new_state = (y0, h_n, q_n)

    return y_pred, alpha, make_residual, new_state


class BDF2VectorizedTransientSolver(VectorizedTransientSolver):
    """BDF2 upgrade of :class:`VectorizedTransientSolver`.

    Implements variable-step BDF2 via the companion method.  On the first
    step Backward Euler is used automatically; from step 2 onward BDF2 is
    activated.  The Jacobian scaling changes from ``1/h`` (BE) to ``α₀/h``
    (BDF2) where ``α₀ = (1 + 2ω)/(1 + ω)`` and ``ω = h_n/h_{n-1}``.

    ``solver_state`` is a 3-tuple ``(y_nm1, h_nm1, q_nm1)``.  ``h_nm1`` is
    initialised to ``+inf`` so that ``ω = 0`` on the first step, making the
    BDF2 formula reduce to Backward Euler via IEEE 754 arithmetic (no branching).
    ``q_nm1`` caches Q(y_{n-1}) to avoid recomputing it each step.
    """

    def order(self, terms) -> int:  # noqa: D102
        return 2

    def init(self, terms, t0, t1, y0, args):
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        is_complex = getattr(self.linear_solver, "is_complex", False)
        y_c = y0[:num_vars] + 1j * y0[num_vars:] if is_complex else y0
        q0 = _compute_history(component_groups, y_c, t0, num_vars)
        return (y0, jnp.float64(jnp.inf), q0)

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: ANN201, D102
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        h_n = t1 - t0
        is_complex = getattr(self.linear_solver, "is_complex", False)

        y_pred, alpha, make_residual, new_state = _bdf2_preamble(y0, t0, h_n, solver_state, component_groups, num_vars, is_complex)

        def newton_update_step(y, _) -> jax.Array:
            if is_complex:
                total_f, total_q, all_vals = assemble_system_complex(y, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
                ground_indices = [0, num_vars]
            else:
                total_f, total_q, all_vals = assemble_system_real(y, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
                ground_indices = [0]

            residual = make_residual(total_f, total_q)
            for idx in ground_indices:
                residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])

            sol = self.linear_solver._solve_impl(all_vals, -residual)  # noqa: SLF001
            delta = sol.value
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
            return y + delta * damping

        fpi = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
        sol = optx.fixed_point(newton_update_step, fpi, y_pred, max_steps=self.newton_max_steps, throw=False)

        y_next = sol.value
        y_error = y_next - y_pred
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )
        return y_next, y_error, {"y0": y0, "y1": y_next}, new_state, result


class BDF2FactorizedTransientSolver(FactorizedTransientSolver):
    """BDF2 upgrade of :class:`FactorizedTransientSolver` (frozen-Jacobian Newton).

    Factors J_eff once at the predictor state and reuses it across all Newton
    iterations, trading quadratic convergence for cheaper per-iteration cost.
    The BDF2 Jacobian scaling ``α₀/h`` is used when factoring.
    """

    def order(self, terms) -> int:  # noqa: D102
        return 2

    def init(self, terms, t0, t1, y0, args):
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        is_complex = getattr(self.linear_solver, "is_complex", False)
        y_c = y0[:num_vars] + 1j * y0[num_vars:] if is_complex else y0
        q0 = _compute_history(component_groups, y_c, t0, num_vars)
        return (y0, jnp.float64(jnp.inf), q0)

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: ANN201, D102
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        h_n = t1 - t0
        is_complex = getattr(self.linear_solver, "is_complex", False)

        y_pred, alpha, make_residual, new_state = _bdf2_preamble(y0, t0, h_n, solver_state, component_groups, num_vars, is_complex)

        if is_complex:
            _, _, frozen_jac_vals = assemble_system_complex(y_pred, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0, num_vars]
        else:
            _, _, frozen_jac_vals = assemble_system_real(y_pred, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0]

        numeric_handle = self.linear_solver.factor_jacobian(frozen_jac_vals)

        def newton_update_step(y: jax.Array, _: Any) -> jax.Array:
            if is_complex:
                total_f, total_q = assemble_residual_only_complex(y, component_groups, t1, h_n, hist_t=hist_t, hist_y=hist_y)
            else:
                total_f, total_q = assemble_residual_only_real(y, component_groups, t1, h_n, hist_t=hist_t, hist_y=hist_y)

            residual = make_residual(total_f, total_q)
            for idx in ground_indices:
                residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])

            sol = self.linear_solver.solve_with_frozen_jacobian(-residual, numeric_handle)
            delta = sol.value
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
            return y + delta * damping

        fpi = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
        sol = optx.fixed_point(newton_update_step, fpi, y_pred, max_steps=self.newton_max_steps, throw=False)
        free_numeric(numeric_handle)

        y_next = sol.value
        y_error = y_next - y_pred
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )
        return y_next, y_error, {"y0": y0, "y1": y_next}, new_state, result


class BDF2RefactoringTransientSolver(RefactoringTransientSolver):
    """BDF2 upgrade of :class:`RefactoringTransientSolver` (KLU refactor per iteration).

    Full quadratic Newton convergence via ``klujax.refactor`` at each iteration,
    combined with BDF2 time discretisation for 2nd-order accuracy.
    """

    def order(self, terms) -> int:  # noqa: D102
        return 2

    def init(self, terms, t0, t1, y0, args):
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        is_complex = getattr(self.linear_solver, "is_complex", False)
        y_c = y0[:num_vars] + 1j * y0[num_vars:] if is_complex else y0
        q0 = _compute_history(component_groups, y_c, t0, num_vars)
        return (y0, jnp.float64(jnp.inf), q0)

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: ANN201, D102
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        h_n = t1 - t0
        is_complex = getattr(self.linear_solver, "is_complex", False)

        y_pred, alpha, make_residual, new_state = _bdf2_preamble(y0, t0, h_n, solver_state, component_groups, num_vars, is_complex)

        if is_complex:
            _, _, init_vals = assemble_system_complex(y_pred, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0, num_vars]
        else:
            _, _, init_vals = assemble_system_real(y_pred, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0]

        numeric_handle = self.linear_solver.factor_jacobian(init_vals)

        def newton_update_step(y: jax.Array, _: Any) -> jax.Array:
            if is_complex:
                total_f, total_q, all_vals = assemble_system_complex(y, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            else:
                total_f, total_q, all_vals = assemble_system_real(y, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)

            residual = make_residual(total_f, total_q)
            for idx in ground_indices:
                residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])

            if hasattr(self.linear_solver, "refactor_and_solve_jacobian"):
                sol, _ = self.linear_solver.refactor_and_solve_jacobian(all_vals, -residual, numeric_handle)
            else:
                refreshed_handle = self.linear_solver.refactor_jacobian(all_vals, numeric_handle)
                sol = self.linear_solver.solve_with_frozen_jacobian(-residual, refreshed_handle)
            delta = sol.value
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
            return y + delta * damping

        fpi = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
        sol = optx.fixed_point(newton_update_step, fpi, y_pred, max_steps=self.newton_max_steps, throw=False)
        free_numeric(numeric_handle)

        y_next = sol.value
        y_error = y_next - y_pred
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )
        return y_next, y_error, {"y0": y0, "y1": y_next}, new_state, result


# ==============================================================================
# SDIRK3 — 3rd-ORDER A-STABLE SOLVER
# ==============================================================================

# Alexander (1977) 3-stage, 3rd-order, L-stable SDIRK method.
# γ is the A-stable root of  6γ³ - 18γ² + 9γ - 1 = 0  (≈ 0.4359).
# The same diagonal value γ means J_eff = dF/dy + (1/(γh)) dQ/dy is
# identical for all three stages — factor once, reuse across all stages.
#
# Butcher tableau  (a31 = b1, a32 = b2, so y_{n+1} = Y3):
#
#   c1 = γ    |  γ      0      0
#   c2        |  a21    γ      0
#   c3 = 1    |  b1     b2     γ
#             ──────────────────────
#               b1     b2     γ
#
# Companion residual at stage i:
#   R_i(Y_i) = F(Y_i) + (Q(Y_i) - Q_hist_i) / (γh) = 0
#
# History terms:
#   Q_hist_1 = Q(y_n)
#   Q_hist_2 = Q(y_n) + (a21/γ) · (Q(Y1) - Q(y_n))
#   Q_hist_3 = Q(y_n) + (b1/γ)  · (Q(Y1) - Q(y_n)) + (b2/γ) · (Q(Y2) - Q(y_n))

_SDIRK3_G = 0.4358665215084589  # A-stable root of 6γ³ - 18γ² + 9γ - 1 = 0
_SDIRK3_INV_G = 1.0 / _SDIRK3_G  # alpha = 1/γ for assembly (step size = γh)
_SDIRK3_C2 = (1.0 + _SDIRK3_G) / 2.0
_SDIRK3_A21 = _SDIRK3_C2 - _SDIRK3_G  # = (1 - γ) / 2
_SDIRK3_B1 = _SDIRK3_G * (_SDIRK3_G - 2.0) / (_SDIRK3_G - 1.0)  # a31 = b1
_SDIRK3_B2 = 1.0 - _SDIRK3_G - _SDIRK3_B1  # a32 = b2


class SDIRK3VectorizedTransientSolver(VectorizedTransientSolver):
    """3rd-order A-stable SDIRK3 solver using full Newton-Raphson at each stage.

    Uses Alexander's L-stable 3-stage SDIRK tableau with the companion method.
    Each timestep performs 3 sequential Newton solves (one per stage) with the
    Jacobian reassembled at every iteration.  The same ``solver_state`` 2-tuple
    ``(y_prev, dt_prev)`` as Backward Euler is used — SDIRK3 is a one-step method.
    """

    def order(self, terms) -> int:  # noqa: D102
        return 3

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: ANN201, D102
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        h_n = t1 - t0
        is_complex = getattr(self.linear_solver, "is_complex", False)
        y_prev_step, dt_prev = solver_state

        # Predictor (1st-order extrapolation, same as BE)
        rate = (y0 - y_prev_step) / (dt_prev + 1e-30)
        y_pred = y0 + rate * h_n

        alpha = _SDIRK3_INV_G  # J_eff = dF/dy + (1/(γh)) dQ/dy

        # History at y_n (= y0)
        y_c0 = y0[:num_vars] + 1j * y0[num_vars:] if is_complex else y0
        q_n = _compute_history(component_groups, y_c0, t0, num_vars)

        ground_indices = [0, num_vars] if is_complex else [0]

        def _run_stage(y_init, q_hist, t_stage):
            def newton_update_step(y, _) -> jax.Array:
                if is_complex:
                    total_f, total_q, all_vals = assemble_system_complex(y, component_groups, t_stage, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
                else:
                    total_f, total_q, all_vals = assemble_system_real(y, component_groups, t_stage, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
                residual = total_f + (total_q - q_hist) / (_SDIRK3_G * h_n)
                for idx in ground_indices:
                    residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])
                sol = self.linear_solver._solve_impl(all_vals, -residual)  # noqa: SLF001
                delta = sol.value
                max_change = jnp.max(jnp.abs(delta))
                damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
                return y + delta * damping

            fpi = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
            stage_sol = optx.fixed_point(newton_update_step, fpi, y_init, max_steps=self.newton_max_steps, throw=False)
            return stage_sol.value, stage_sol.result

        # Stage 1 — t_s1 = t0 + γ·h
        t_s1 = t0 + _SDIRK3_G * h_n
        Y1, r1 = _run_stage(y_pred, q_n, t_s1)
        Y1_c = Y1[:num_vars] + 1j * Y1[num_vars:] if is_complex else Y1
        q_Y1 = _compute_history(component_groups, Y1_c, t_s1, num_vars)
        q_hist_2 = q_n + (_SDIRK3_A21 / _SDIRK3_G) * (q_Y1 - q_n)

        # Stage 2 — t_s2 = t0 + c2·h
        t_s2 = t0 + _SDIRK3_C2 * h_n
        Y2, r2 = _run_stage(Y1, q_hist_2, t_s2)
        Y2_c = Y2[:num_vars] + 1j * Y2[num_vars:] if is_complex else Y2
        q_Y2 = _compute_history(component_groups, Y2_c, t_s2, num_vars)
        q_hist_3 = q_n + (_SDIRK3_B1 / _SDIRK3_G) * (q_Y1 - q_n) + (_SDIRK3_B2 / _SDIRK3_G) * (q_Y2 - q_n)

        # Stage 3 — t_s3 = t1 (c3 = 1), y_{n+1} = Y3
        Y3, r3 = _run_stage(Y2, q_hist_3, t1)

        y_next = Y3
        y_error = Y3 - y_pred
        all_ok = (r1 == optx.RESULTS.successful) & (r2 == optx.RESULTS.successful) & (r3 == optx.RESULTS.successful)
        result = jax.lax.cond(
            all_ok,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )
        return y_next, y_error, {"y0": y0, "y1": y_next}, (y0, h_n), result


class SDIRK3FactorizedTransientSolver(FactorizedTransientSolver):
    """3rd-order A-stable SDIRK3 solver with frozen-Jacobian Newton across all stages.

    Factors J_eff = dF/dy + (1/(γh))·dQ/dy once at the predictor state,
    then reuses it for all Newton iterations in all three SDIRK stages.
    This is the recommended backend for large sparse circuits — the single
    factorisation is shared across all stages because SDIRK's constant diagonal
    γ gives the same effective Jacobian at every stage.
    """

    def order(self, terms) -> int:  # noqa: D102
        return 3

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: ANN201, D102
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        h_n = t1 - t0
        is_complex = getattr(self.linear_solver, "is_complex", False)
        y_prev_step, dt_prev = solver_state

        rate = (y0 - y_prev_step) / (dt_prev + 1e-30)
        y_pred = y0 + rate * h_n

        alpha = _SDIRK3_INV_G

        # Factor J_eff ONCE at predictor — reused for all stages and iterations
        if is_complex:
            _, _, frozen_jac_vals = assemble_system_complex(y_pred, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0, num_vars]
        else:
            _, _, frozen_jac_vals = assemble_system_real(y_pred, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0]

        numeric_handle = self.linear_solver.factor_jacobian(frozen_jac_vals)

        y_c0 = y0[:num_vars] + 1j * y0[num_vars:] if is_complex else y0
        q_n = _compute_history(component_groups, y_c0, t0, num_vars)

        def _run_stage(y_init, q_hist):
            def newton_update_step(y: jax.Array, _: Any) -> jax.Array:
                if is_complex:
                    total_f, total_q = assemble_residual_only_complex(y, component_groups, t1, h_n, hist_t=hist_t, hist_y=hist_y)
                else:
                    total_f, total_q = assemble_residual_only_real(y, component_groups, t1, h_n, hist_t=hist_t, hist_y=hist_y)
                residual = total_f + (total_q - q_hist) / (_SDIRK3_G * h_n)
                for idx in ground_indices:
                    residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])
                sol = self.linear_solver.solve_with_frozen_jacobian(-residual, numeric_handle)
                delta = sol.value
                max_change = jnp.max(jnp.abs(delta))
                damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
                return y + delta * damping

            fpi = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
            stage_sol = optx.fixed_point(newton_update_step, fpi, y_init, max_steps=self.newton_max_steps, throw=False)
            return stage_sol.value, stage_sol.result

        t_s1 = t0 + _SDIRK3_G * h_n
        Y1, r1 = _run_stage(y_pred, q_n)
        Y1_c = Y1[:num_vars] + 1j * Y1[num_vars:] if is_complex else Y1
        q_Y1 = _compute_history(component_groups, Y1_c, t_s1, num_vars)
        q_hist_2 = q_n + (_SDIRK3_A21 / _SDIRK3_G) * (q_Y1 - q_n)

        t_s2 = t0 + _SDIRK3_C2 * h_n
        Y2, r2 = _run_stage(Y1, q_hist_2)
        Y2_c = Y2[:num_vars] + 1j * Y2[num_vars:] if is_complex else Y2
        q_Y2 = _compute_history(component_groups, Y2_c, t_s2, num_vars)
        q_hist_3 = q_n + (_SDIRK3_B1 / _SDIRK3_G) * (q_Y1 - q_n) + (_SDIRK3_B2 / _SDIRK3_G) * (q_Y2 - q_n)

        Y3, r3 = _run_stage(Y2, q_hist_3)
        free_numeric(numeric_handle)

        y_next = Y3
        y_error = Y3 - y_pred
        all_ok = (r1 == optx.RESULTS.successful) & (r2 == optx.RESULTS.successful) & (r3 == optx.RESULTS.successful)
        result = jax.lax.cond(
            all_ok,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )
        return y_next, y_error, {"y0": y0, "y1": y_next}, (y0, h_n), result


class SDIRK3RefactoringTransientSolver(RefactoringTransientSolver):
    """3rd-order A-stable SDIRK3 solver with KLU refactor at each Newton iteration.

    Provides full quadratic Newton convergence via ``klujax.refactor`` within
    each stage, combined with SDIRK3 time discretisation for 3rd-order accuracy.
    """

    def order(self, terms) -> int:  # noqa: D102
        return 3

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: ANN201, D102
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        h_n = t1 - t0
        is_complex = getattr(self.linear_solver, "is_complex", False)
        y_prev_step, dt_prev = solver_state

        rate = (y0 - y_prev_step) / (dt_prev + 1e-30)
        y_pred = y0 + rate * h_n

        alpha = _SDIRK3_INV_G

        if is_complex:
            _, _, init_vals = assemble_system_complex(y_pred, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0, num_vars]
        else:
            _, _, init_vals = assemble_system_real(y_pred, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0]

        numeric_handle = self.linear_solver.factor_jacobian(init_vals)

        y_c0 = y0[:num_vars] + 1j * y0[num_vars:] if is_complex else y0
        q_n = _compute_history(component_groups, y_c0, t0, num_vars)

        def _run_stage(y_init, q_hist, t_stage):
            def newton_update_step(y: jax.Array, _: Any) -> jax.Array:
                if is_complex:
                    total_f, total_q, all_vals = assemble_system_complex(y, component_groups, t_stage, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
                else:
                    total_f, total_q, all_vals = assemble_system_real(y, component_groups, t_stage, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
                residual = total_f + (total_q - q_hist) / (_SDIRK3_G * h_n)
                for idx in ground_indices:
                    residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])
                if hasattr(self.linear_solver, "refactor_and_solve_jacobian"):
                    sol, _ = self.linear_solver.refactor_and_solve_jacobian(all_vals, -residual, numeric_handle)
                else:
                    refreshed_handle = self.linear_solver.refactor_jacobian(all_vals, numeric_handle)
                    sol = self.linear_solver.solve_with_frozen_jacobian(-residual, refreshed_handle)
                delta = sol.value
                max_change = jnp.max(jnp.abs(delta))
                damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
                return y + delta * damping

            fpi = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
            stage_sol = optx.fixed_point(newton_update_step, fpi, y_init, max_steps=self.newton_max_steps, throw=False)
            return stage_sol.value, stage_sol.result

        t_s1 = t0 + _SDIRK3_G * h_n
        Y1, r1 = _run_stage(y_pred, q_n, t_s1)
        Y1_c = Y1[:num_vars] + 1j * Y1[num_vars:] if is_complex else Y1
        q_Y1 = _compute_history(component_groups, Y1_c, t_s1, num_vars)
        q_hist_2 = q_n + (_SDIRK3_A21 / _SDIRK3_G) * (q_Y1 - q_n)

        t_s2 = t0 + _SDIRK3_C2 * h_n
        Y2, r2 = _run_stage(Y1, q_hist_2, t_s2)
        Y2_c = Y2[:num_vars] + 1j * Y2[num_vars:] if is_complex else Y2
        q_Y2 = _compute_history(component_groups, Y2_c, t_s2, num_vars)
        q_hist_3 = q_n + (_SDIRK3_B1 / _SDIRK3_G) * (q_Y1 - q_n) + (_SDIRK3_B2 / _SDIRK3_G) * (q_Y2 - q_n)

        Y3, r3 = _run_stage(Y2, q_hist_3, t1)
        free_numeric(numeric_handle)

        y_next = Y3
        y_error = Y3 - y_pred
        all_ok = (r1 == optx.RESULTS.successful) & (r2 == optx.RESULTS.successful) & (r3 == optx.RESULTS.successful)
        result = jax.lax.cond(
            all_ok,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )
        return y_next, y_error, {"y0": y0, "y1": y_next}, (y0, h_n), result


# ==============================================================================
# TRAPEZOIDAL — 2nd-ORDER A-STABLE SOLVER (zero numerical damping)
# ==============================================================================
#
# Trapezoidal-rule companion-method discretisation of  F(V) + dQ/dt = 0:
#
#   Q(V_new) - Q(V_old) = -h/2 · (F(V_new) + F(V_old))
#
# Rearranged into the residual form circulax's Newton loop expects:
#
#   R(V_new) = F(V_new) + (2/h) · (Q(V_new) - Q_old) + F_old = 0
#
# So  α = 2  for the Jacobian stamp  J_eff = G + (α/h)·C  (vs α = 1 for BE,
# α = α₀ ∈ [1, 5/3] for variable-step BDF2, α = 1/γ ≈ 2.29 for SDIRK3).


def _compute_history_fq(component_groups, y_flat, t, num_vars, is_complex, dt=0.0, hist_t=None, hist_y=None):
    """Compute the (F, Q) residual pair at a given flat state vector — used by trap.

    ``dt`` here is only used by delayed components to guard ``tau >= dt``
    (see ``circulax.solvers.assembly``); it plays no role in the F/Q
    computation itself, unlike the ``dt`` accepted by ``assemble_system_*``.
    Callers with no real step size available (e.g. ``_trap_init``, evaluated
    at ``t0`` before any step) pass ``0.0``, which makes the guard trivial.
    """
    if is_complex:
        total_f, total_q = assemble_residual_only_complex(y_flat, component_groups, t, dt, hist_t=hist_t, hist_y=hist_y)
    else:
        total_f, total_q = assemble_residual_only_real(y_flat, component_groups, t, dt, hist_t=hist_t, hist_y=hist_y)
    return total_f, total_q


def _trap_preamble(y0, t0, h_n, solver_state, component_groups, num_vars, is_complex):
    """Compute trapezoidal coefficients, history terms (F_old, Q_old), and predictor.

    ``solver_state`` is a 4-tuple ``(y_nm1, h_nm1, f_nm1, q_nm1)`` where
    ``f_nm1`` and ``q_nm1`` are the cached F/Q values at ``y_nm1`` from the
    previous step's converged Newton solution.  The cache avoids a redundant
    physics evaluation at the start of each timestep.
    """
    y_nm1, h_nm1, f_old, q_old = solver_state

    alpha = 2.0

    rate = (y0 - y_nm1) / (h_nm1 + 1e-30)
    y_pred = y0 + rate * h_n

    def make_residual(total_f, total_q):
        return total_f + (alpha * (total_q - q_old)) / h_n + f_old

    return y_pred, alpha, make_residual


def _trap_init(component_groups, y0, t0, num_vars, is_complex, hist_t=None, hist_y=None):
    """Build the initial 4-tuple solver state for the trap solvers.

    Computes f0, q0 at y0 so the first step can use the cached values
    rather than recomputing them in ``_trap_preamble``.
    """
    f0, q0 = _compute_history_fq(component_groups, y0, t0, num_vars, is_complex, hist_t=hist_t, hist_y=hist_y)
    return (y0, jnp.float64(jnp.inf), f0, q0)


class TrapVectorizedTransientSolver(VectorizedTransientSolver):
    """Trapezoidal-rule transient solver — circulax's default integrator.

    2nd-order A-stable with zero numerical damping at all frequencies.
    Preserves limit-cycle frequencies exactly, unlike BDF2/BE which
    introduce L²-stable damping that pulls oscillator frequencies.

    Use BDF2 instead if you see trapezoidal ringing (sawtooth at 2·dt)
    on circuits with very sharp digital edges.
    """

    def order(self, terms) -> int:  # noqa: D102
        return 2

    def init(self, terms, t0, t1, y0, args):
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        is_complex = getattr(self.linear_solver, "is_complex", False)
        return _trap_init(component_groups, y0, t0, num_vars, is_complex, hist_t=hist_t, hist_y=hist_y)

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: ANN201, D102
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        h_n = t1 - t0
        is_complex = getattr(self.linear_solver, "is_complex", False)

        y_pred, alpha, make_residual = _trap_preamble(
            y0, t0, h_n, solver_state, component_groups, num_vars, is_complex,
        )

        def newton_update_step(y, _) -> jax.Array:
            if is_complex:
                total_f, total_q, all_vals = assemble_system_complex(y, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
                ground_indices = [0, num_vars]
            else:
                total_f, total_q, all_vals = assemble_system_real(y, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
                ground_indices = [0]

            residual = make_residual(total_f, total_q)
            for idx in ground_indices:
                residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])

            sol = self.linear_solver._solve_impl(all_vals, -residual)  # noqa: SLF001
            delta = sol.value
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
            return y + delta * damping

        fpi = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
        sol = optx.fixed_point(newton_update_step, fpi, y_pred, max_steps=self.newton_max_steps, throw=False)

        y_next = sol.value
        y_error = y_next - y_pred

        # Cache f/q at the converged solution for the next step's preamble.
        f_new, q_new = _compute_history_fq(component_groups, y_next, t1, num_vars, is_complex, dt=h_n, hist_t=hist_t, hist_y=hist_y)
        new_state = (y0, h_n, f_new, q_new)

        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )
        return y_next, y_error, {"y0": y0, "y1": y_next}, new_state, result


class TrapFactorizedTransientSolver(FactorizedTransientSolver):
    """Trapezoidal upgrade of :class:`FactorizedTransientSolver`.

    Factors J_eff once at the predictor and reuses the factorisation
    across the Newton iteration.
    """

    def order(self, terms) -> int:  # noqa: D102
        return 2

    def init(self, terms, t0, t1, y0, args):
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        is_complex = getattr(self.linear_solver, "is_complex", False)
        return _trap_init(component_groups, y0, t0, num_vars, is_complex, hist_t=hist_t, hist_y=hist_y)

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: ANN201, D102
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        h_n = t1 - t0
        is_complex = getattr(self.linear_solver, "is_complex", False)

        y_pred, alpha, make_residual = _trap_preamble(
            y0, t0, h_n, solver_state, component_groups, num_vars, is_complex,
        )

        if is_complex:
            _, _, frozen_jac_vals = assemble_system_complex(y_pred, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0, num_vars]
        else:
            _, _, frozen_jac_vals = assemble_system_real(y_pred, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0]

        numeric_handle = self.linear_solver.factor_jacobian(frozen_jac_vals)

        def newton_update_step(y: jax.Array, _: Any) -> jax.Array:
            if is_complex:
                total_f, total_q = assemble_residual_only_complex(y, component_groups, t1, h_n, hist_t=hist_t, hist_y=hist_y)
            else:
                total_f, total_q = assemble_residual_only_real(y, component_groups, t1, h_n, hist_t=hist_t, hist_y=hist_y)

            residual = make_residual(total_f, total_q)
            for idx in ground_indices:
                residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])

            sol = self.linear_solver.solve_with_frozen_jacobian(-residual, numeric_handle)
            delta = sol.value
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
            return y + delta * damping

        fpi = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
        sol = optx.fixed_point(newton_update_step, fpi, y_pred, max_steps=self.newton_max_steps, throw=False)
        free_numeric(numeric_handle)

        y_next = sol.value
        y_error = y_next - y_pred

        # Cache f/q at the converged solution for the next step's preamble.
        f_new, q_new = _compute_history_fq(component_groups, y_next, t1, num_vars, is_complex, dt=h_n, hist_t=hist_t, hist_y=hist_y)
        new_state = (y0, h_n, f_new, q_new)

        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )
        return y_next, y_error, {"y0": y0, "y1": y_next}, new_state, result


class TrapRefactoringTransientSolver(RefactoringTransientSolver):
    """Trapezoidal upgrade of :class:`RefactoringTransientSolver`.

    Refactors the KLU factorisation at every Newton iteration (full quadratic
    convergence).
    """

    def order(self, terms) -> int:  # noqa: D102
        return 2

    def init(self, terms, t0, t1, y0, args):
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        is_complex = getattr(self.linear_solver, "is_complex", False)
        return _trap_init(component_groups, y0, t0, num_vars, is_complex, hist_t=hist_t, hist_y=hist_y)

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: ANN201, D102
        component_groups, num_vars, hist_t, hist_y = _unpack_args(args)
        h_n = t1 - t0
        is_complex = getattr(self.linear_solver, "is_complex", False)

        y_pred, alpha, make_residual = _trap_preamble(
            y0, t0, h_n, solver_state, component_groups, num_vars, is_complex,
        )

        if is_complex:
            _, _, init_vals = assemble_system_complex(y_pred, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0, num_vars]
        else:
            _, _, init_vals = assemble_system_real(y_pred, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            ground_indices = [0]

        numeric_handle = self.linear_solver.factor_jacobian(init_vals)

        def newton_update_step(y: jax.Array, _: Any) -> jax.Array:
            if is_complex:
                total_f, total_q, all_vals = assemble_system_complex(y, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)
            else:
                total_f, total_q, all_vals = assemble_system_real(y, component_groups, t1, h_n, alpha=alpha, hist_t=hist_t, hist_y=hist_y)

            residual = make_residual(total_f, total_q)
            for idx in ground_indices:
                residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])

            if hasattr(self.linear_solver, "refactor_and_solve_jacobian"):
                sol, _ = self.linear_solver.refactor_and_solve_jacobian(all_vals, -residual, numeric_handle)
            else:
                refreshed_handle = self.linear_solver.refactor_jacobian(all_vals, numeric_handle)
                sol = self.linear_solver.solve_with_frozen_jacobian(-residual, refreshed_handle)
            delta = sol.value
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
            return y + delta * damping

        fpi = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
        sol = optx.fixed_point(newton_update_step, fpi, y_pred, max_steps=self.newton_max_steps, throw=False)
        free_numeric(numeric_handle)

        y_next = sol.value
        y_error = y_next - y_pred

        # Cache f/q at the converged solution for the next step's preamble.
        f_new, q_new = _compute_history_fq(component_groups, y_next, t1, num_vars, is_complex, dt=h_n, hist_t=hist_t, hist_y=hist_y)
        new_state = (y0, h_n, f_new, q_new)

        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )
        return y_next, y_error, {"y0": y0, "y1": y_next}, new_state, result


def setup_transient(
    groups: list, linear_strategy: CircuitLinearSolver, transient_solver: AbstractSolver = None
) -> Callable[..., diffrax.Solution]:
    """Configures and returns a function for executing transient analysis.

    This function acts as a factory, preparing a transient solver that is
    pre-configured with the circuit's linear strategy. It returns a callable
    that executes the time-domain simulation using `diffrax.diffeqsolve`.

    Args:
        groups (list): A list of component groups that define the circuit.
        linear_strategy (CircuitLinearSolver): The configured linear solver
            strategy, typically obtained from `analyze_circuit`.
        transient_solver (optional): The transient solver class to use.
            If None, `BDF2VectorizedTransientSolver` will be used.

    Returns:
        Callable[..., Any]: A function that executes the transient analysis.
        This returned function accepts the following arguments:

            t0 (float): The start time of the simulation.
            t1 (float): The end time of the simulation.
            dt0 (float): The initial time step for the solver.
            y0 (ArrayLike): The initial state vector of the system.
            saveat (diffrax.SaveAt, optional): Specifies time points at which
                to save the solution. Defaults to None.
            max_steps (int, optional): The maximum number of steps the solver
                can take. Defaults to 100000.
            throw (bool, optional): If True, the solver will raise an error on
                failure. Defaults to False.
            term (diffrax.AbstractTerm, optional): The term defining the ODE.
                Defaults to a zero-value ODETerm.
            stepsize_controller (diffrax.AbstractStepSizeController, optional):
                The step size controller. Defaults to `ConstantStepSize()`.
                For circuits with delayed components, an adaptive controller
                (e.g. `PIDController`) is automatically clamped so the
                proposed step never exceeds the smallest active delay `tau`;
                `ConstantStepSize` is left unclamped (an explicit
                `dt0 > tau` still raises). Adaptive+delay circuits may need
                a larger `max_steps` budget than constant-step ones, sized
                against `(t1 - t0) / tau_min` as a floor, since the clamp
                can force more steps than accuracy alone would require.
            **kwargs: Additional keyword arguments to pass directly to
                `diffrax.diffeqsolve`.

    """
    fdomain_names = [g.name for g in groups.values() if getattr(g, "is_fdomain", False)]
    if fdomain_names:
        msg = (
            "Frequency-domain components cannot be used in transient simulation "
            "(time-domain convolution is not supported). "
            f"Offending groups: {fdomain_names}. "
            "Use setup_harmonic_balance() instead."
        )
        raise RuntimeError(msg)

    # Fixed time-delay support: gate the delay-history buffer behind whether
    # any group actually has a delayed component, so undelayed circuits pay
    # no cost. Adaptive step-size controllers are supported (circulax GitHub
    # issue #2, follow-up): the proposed step is proactively clamped to the
    # smallest active tau (see `min_active_tau`/`min_tau` below) so
    # assembly.py's `tau >= dt` guard stays a pure safety net rather than
    # something adaptive stepping can trip mid-run. `ConstantStepSize`
    # behavior is unchanged -- an explicit `dt0 > tau` still raises.
    has_delay = any(getattr(g, "has_delay", False) for g in groups.values())

    if transient_solver is None:
        # Pick the best Trap variant the linear solver supports.
        # Trap (2nd order, A-stable, zero numerical damping) is the right
        # pick for oscillators, LC tanks, and harmonic balance.  For circuits
        # with sharp digital edges that cause trapezoidal ringing, pass
        # BDF2*TransientSolver explicitly.
        if hasattr(linear_strategy, "refactor_jacobian"):
            transient_solver = TrapRefactoringTransientSolver
        elif hasattr(linear_strategy, "factor_jacobian"):
            transient_solver = TrapFactorizedTransientSolver
        else:
            transient_solver = TrapVectorizedTransientSolver

    import inspect
    tsolver = transient_solver(linear_solver=linear_strategy) if inspect.isclass(transient_solver) else transient_solver

    sys_size = linear_strategy.sys_size // 2 if linear_strategy.is_complex else linear_strategy.sys_size

    def _execute_transient(
        *,
        t0: float,
        t1: float,
        dt0: float,
        y0: ArrayLike,
        saveat: diffrax.SaveAt = None,
        max_steps: int = 100000,
        throw: bool = False,
        **kwargs: Any,
    ) -> diffrax.Solution:
        """Executes the transient simulation for the pre-configured circuit."""
        term = kwargs.pop("term", diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y)))
        solver = kwargs.pop("solver", tsolver)
        args = kwargs.pop("args", (groups, sys_size))
        stepsize_controller = kwargs.pop("stepsize_controller", ConstantStepSize())
        checkpoints = kwargs.pop("checkpoints", None)

        # Derive min_tau from the per-call component groups (args[0]), not the
        # `groups` closed over above -- callers (e.g. gradient checks) may
        # override `args` with perturbed params, and min_tau must track
        # whatever is actually being simulated.
        is_adaptive = isinstance(stepsize_controller, AbstractAdaptiveStepSizeController)
        min_tau = min_active_tau(args[0]) if (has_delay and is_adaptive) else None

        sol = circuit_diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=args,
            saveat=saveat,
            max_steps=max_steps,
            throw=throw,
            stepsize_controller=stepsize_controller,
            checkpoints=checkpoints,
            record_history=has_delay,
            min_tau=min_tau,
        )

        return sol

    return _execute_transient
