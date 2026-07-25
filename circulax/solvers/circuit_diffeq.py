"""Stripped-down ODE integrator for circuit simulation.

Compared to ``diffrax.diffeqsolve`` this removes all features not used by
Circulax transient solvers:

Removed:
- Events (``event=``, event state, root-finding after the loop)
- Dense output (``saveat.dense``)
- Progress meter
- SDE / CDE support and related warnings
- Backward time direction – always assumes ``t0 < t1``
- ``made_jump`` – treated as static ``False``, not carried in state
- Deprecated ``discrete_terminating_event``
- Term-compatibility checking (assumed ``ODETerm`` always)
- ``solver_state`` / ``controller_state`` passthrough arguments
- ``saveat.steps`` saving mode

Retained:
- ``SaveAt(ts=..., t1=True, t0=False, fn=...)`` – used by benchmarks
- ``RecursiveCheckpointAdjoint`` – binomial checkpointing for autodiff
- Full step-size control (``PIDController``, ``ConstantStepSize``)
- ``diffrax.Solution`` output – fully compatible with existing code
"""

from __future__ import annotations

import functools as ft
from typing import Any

import diffrax
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax.internal as lxi
from diffrax._custom_types import (
    FloatScalarLike,
    IntScalarLike,
    RealScalarLike,
)
from diffrax._integrate import SaveState, _clip_to_end, _save  # reuse helpers
from diffrax._misc import linear_rescale, static_select  # noqa: F401 (static_select used below)
from diffrax._saveat import SaveAt, SubSaveAt, save_y
from diffrax._solution import RESULTS, Solution, is_okay, is_successful
from diffrax._step_size_controller import (
    AbstractAdaptiveStepSizeController,
    AbstractStepSizeController,
    ConstantStepSize,
)
from diffrax._term import AbstractTerm, WrapTerm
from jaxtyping import Array, ArrayLike, PyTree

# ---------------------------------------------------------------------------
# Simplified state – no event fields, no dense fields, no progress meter
# ---------------------------------------------------------------------------


class CircuitState(eqx.Module):
    """Carry state for the circuit integration while-loop."""

    # Evolving state
    y: PyTree[Array]
    tprev: FloatScalarLike
    tnext: FloatScalarLike
    solver_state: PyTree[ArrayLike]
    controller_state: PyTree[ArrayLike]

    # Result tracking
    result: RESULTS
    num_steps: IntScalarLike
    num_accepted_steps: IntScalarLike
    num_rejected_steps: IntScalarLike

    # Output buffers (updated via .at[].set() during the loop)
    save_state: PyTree[SaveState]

    # Delay-history buffer (updated via .at[].set() during the loop).
    # ``hist_t``/``hist_y`` are ``None`` when the circuit has no delayed
    # components (``record_history=False``), so no buffer cost is paid.
    hist_t: Array | None
    hist_y: Array | None


# ---------------------------------------------------------------------------
# Buffer helpers (used by eqxi.while_loop for checkpointed AD)
# ---------------------------------------------------------------------------


def _is_subsaveat(x: Any) -> bool:
    return isinstance(x, SubSaveAt)


def _is_save_state(x: Any) -> bool:
    return isinstance(x, SaveState)


def _is_none(x: Any) -> bool:
    return x is None


def _outer_buffers(state: CircuitState):
    """Return the mutable output buffers carried through the while-loop.

    ``hist_t``/``hist_y`` are always included structurally (even when
    ``None``, i.e. no delayed components) rather than conditionally omitted
    by value — the ``buffers=`` selector must be a pure function of pytree
    *structure*, not of leaf values, for equinox's checkpointed backward
    pass to resolve buffer perturbations correctly.
    """
    save_states = jtu.tree_leaves(state.save_state, is_leaf=_is_save_state)
    save_states = [s for s in save_states if _is_save_state(s)]
    return [s.ts for s in save_states] + [s.ys for s in save_states] + [state.hist_t, state.hist_y]


def _inner_buffers(save_state: SaveState):
    return save_state.ts, save_state.ys


# ---------------------------------------------------------------------------
# Inner loop helpers for SaveAt(ts=...)
# ---------------------------------------------------------------------------

_inner_loop = jax.named_call(eqxi.while_loop, name="circuit-inner-loop")
_outer_loop = jax.named_call(eqxi.while_loop, name="circuit-outer-loop")


def _circuit_loop(
    *,
    solver,
    stepsize_controller,
    saveat: SaveAt,
    t0: FloatScalarLike,
    t1: FloatScalarLike,
    max_steps: int | None,
    terms,
    args,
    init_state: CircuitState,
    inner_while_loop,
    outer_while_loop,
) -> CircuitState:
    """Core integration loop (no events, no dense output, no progress meter)."""
    # Pre-compute t1 - 100 ULPs for step clipping
    t1_clip_floor = t1
    for _ in range(100):
        t1_clip_floor = eqxi.prevbefore(t1_clip_floor)

    # ------------------------------------------------------------------
    # Optionally save at t0
    # ------------------------------------------------------------------

    def _save_t0(subsaveat: SubSaveAt, save_state: SaveState) -> SaveState:
        if subsaveat.t0:
            save_state = _save(t0, init_state.y, args, subsaveat.fn, save_state, repeat=1)
        return save_state

    save_state = jtu.tree_map(_save_t0, saveat.subs, init_state.save_state, is_leaf=_is_subsaveat)
    init_state = eqx.tree_at(lambda s: s.save_state, init_state, save_state, is_leaf=_is_none)

    # ------------------------------------------------------------------
    # Seed the delay-history buffer with (t0, y0) — delayed reads for
    # query times before t0 clamp to the initial condition (jnp.interp's
    # natural boundary behaviour against an ascending sample array).
    # ------------------------------------------------------------------

    if init_state.hist_t is not None:
        hist_t0 = init_state.hist_t.at[0].set(t0)
        hist_y0 = init_state.hist_y.at[0].set(init_state.y)
        init_state = eqx.tree_at(
            lambda s: (s.hist_t, s.hist_y), init_state, (hist_t0, hist_y0), is_leaf=_is_none,
        )

    # ------------------------------------------------------------------
    # While-loop condition
    # ------------------------------------------------------------------

    def cond_fun(state: CircuitState) -> bool:
        return (state.tprev < t1) & is_successful(state.result)

    # ------------------------------------------------------------------
    # Per-step body
    # ------------------------------------------------------------------

    def body_fun(state: CircuitState) -> CircuitState:
        # Delayed components read history accepted up to (but not including)
        # this step, i.e. the buffer as carried into this iteration — it is
        # appended to further down, once the step below is accepted.
        #
        # `state.hist_t`/`state.hist_y` are `eqxi`'s internal `_Buffer` proxy
        # (registered in `_outer_buffers` for efficient donation), not plain
        # arrays -- ordinary jnp ops like `jnp.interp` don't accept them
        # directly. Materialise via full-slice indexing (`_Buffer.__getitem__`
        # unwraps one level) before handing them to the solver/assembly path;
        # the write further below still operates on `state.hist_t`/`hist_y`
        # themselves, so this doesn't disturb the buffer-donation machinery.
        step_args = args if state.hist_t is None else (*args, state.hist_t[:], state.hist_y[:])

        # made_jump is always False for circuits (no discontinuities in input)
        (y, y_error, dense_info, solver_state, solver_result) = solver.step(
            terms,
            state.tprev,
            state.tnext,
            state.y,
            step_args,
            state.solver_state,
            False,  # made_jump – static, never traced
        )

        # NaN → inf so the step-size controller can reject gracefully without cascading NaNs.
        y_error = jtu.tree_map(lambda x: jnp.where(jnp.isnan(x), jnp.inf, x), y_error)

        error_order = solver.error_order(terms)
        (
            keep_step,
            tprev,
            tnext,
            _made_jump,
            controller_state,
            stepsize_controller_result,
        ) = stepsize_controller.adapt_step_size(
            state.tprev,
            state.tnext,
            state.y,
            y,
            args,
            y_error,
            error_order,
            state.controller_state,
        )

        tprev = jnp.minimum(tprev, t1)
        tnext = _clip_to_end(tprev, tnext, t1, t1_clip_floor, keep_step)

        keep = lambda a, b: jnp.where(keep_step, a, b)
        y = jtu.tree_map(keep, y, state.y)
        solver_state = jtu.tree_map(keep, solver_state, state.solver_state)
        solver_result = RESULTS.where(keep_step, solver_result, RESULTS.successful)

        result = RESULTS.where(is_okay(state.result), solver_result, state.result)
        result = RESULTS.where(is_okay(result), stepsize_controller_result, result)

        num_steps = state.num_steps + 1
        num_accepted_steps = state.num_accepted_steps + jnp.where(keep_step, 1, 0)
        num_rejected_steps = state.num_rejected_steps + jnp.where(keep_step, 0, 1)

        # ------------------------------------------------------------------
        # Record this accepted step in the delay-history buffer. Slot index
        # is derived from the *pre-step* accepted-step count, so a rejected
        # step retries the same slot; the where(...) makes the write a no-op
        # (self-assignment) on rejection rather than clobbering with a stale
        # duplicate point.
        # ------------------------------------------------------------------
        hist_t = state.hist_t
        hist_y = state.hist_y
        if hist_t is not None:
            hist_idx = jnp.minimum(state.num_accepted_steps + 1, hist_t.shape[0] - 1)
            hist_t = hist_t.at[hist_idx].set(jnp.where(keep_step, tprev, hist_t[hist_idx]))
            hist_y = hist_y.at[hist_idx].set(jnp.where(keep_step, y, hist_y[hist_idx]))

        interpolator = solver.interpolation_cls(t0=state.tprev, t1=state.tnext, **dense_info)

        save_state = state.save_state

        def _save_ts(subsaveat: SubSaveAt, ss: SaveState) -> SaveState:
            if subsaveat.ts is None:
                return ss

            def _cond(_ss):
                return keep_step & (subsaveat.ts[_ss.saveat_ts_index] <= state.tnext) & (_ss.saveat_ts_index < len(subsaveat.ts))

            def _body(_ss):
                _t = subsaveat.ts[_ss.saveat_ts_index]
                _y = interpolator.evaluate(_t)
                _ts = _ss.ts.at[_ss.save_index].set(_t)
                _ys = jtu.tree_map(
                    lambda __y, __ys: __ys.at[_ss.save_index].set(__y),
                    subsaveat.fn(_t, _y, args),
                    _ss.ys,
                )
                return SaveState(
                    saveat_ts_index=_ss.saveat_ts_index + 1,
                    ts=_ts,
                    ys=_ys,
                    save_index=_ss.save_index + 1,
                )

            return inner_while_loop(
                _cond,
                _body,
                ss,
                max_steps=len(subsaveat.ts),
                buffers=_inner_buffers,
                checkpoints=len(subsaveat.ts),
            )

        save_state = jtu.tree_map(_save_ts, saveat.subs, save_state, is_leaf=_is_subsaveat)

        return CircuitState(
            y=y,
            tprev=tprev,
            tnext=tnext,
            solver_state=solver_state,
            controller_state=controller_state,
            result=result,
            num_steps=num_steps,
            num_accepted_steps=num_accepted_steps,
            num_rejected_steps=num_rejected_steps,
            save_state=save_state,
            hist_t=hist_t,
            hist_y=hist_y,
        )

    # ------------------------------------------------------------------
    # Run the while-loop
    # ------------------------------------------------------------------

    final_state: CircuitState = outer_while_loop(cond_fun, body_fun, init_state, max_steps=max_steps, buffers=_outer_buffers)

    # ------------------------------------------------------------------
    # Save at t1 (the very end)
    # ------------------------------------------------------------------

    tfinal = final_state.tprev
    yfinal = final_state.y

    # Edge-case: t0 == t1, loop body never ran, save ts anyway
    def _save_if_t0_equals_t1(subsaveat: SubSaveAt, ss: SaveState) -> SaveState:
        if subsaveat.ts is not None:
            ss = _save(t0, yfinal, args, subsaveat.fn, ss, repeat=len(subsaveat.ts))
        return ss

    save_state = jax.lax.cond(
        eqxi.unvmap_any(t0 == t1),
        lambda _ss: jax.lax.cond(
            t0 == t1,
            lambda _s: jtu.tree_map(_save_if_t0_equals_t1, saveat.subs, _s, is_leaf=_is_subsaveat),
            lambda _s: _s,
            _ss,
        ),
        lambda _ss: _ss,
        final_state.save_state,
    )
    final_state = eqx.tree_at(lambda s: s.save_state, final_state, save_state, is_leaf=_is_none)

    def _save_t1(subsaveat: SubSaveAt, ss: SaveState) -> SaveState:
        # `steps` saving mode not supported – always treat as steps==0
        if subsaveat.t1:
            ss = _save(tfinal, yfinal, args, subsaveat.fn, ss, repeat=1)
        return ss

    save_state = jtu.tree_map(_save_t1, saveat.subs, final_state.save_state, is_leaf=_is_subsaveat)
    final_state = eqx.tree_at(lambda s: s.save_state, final_state, save_state, is_leaf=_is_none)

    # Mark max_steps_reached if the loop ended due to exhausting budget
    result = final_state.result
    result = RESULTS.where(cond_fun(final_state), RESULTS.max_steps_reached, result)
    return eqx.tree_at(lambda s: s.result, final_state, result)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@eqx.filter_jit
def circuit_diffeqsolve(
    terms: PyTree[AbstractTerm],
    solver,
    t0: RealScalarLike,
    t1: RealScalarLike,
    dt0: RealScalarLike,
    y0: PyTree[ArrayLike],
    args: PyTree[Any] = None,
    *,
    saveat: SaveAt = SaveAt(t1=True),
    stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
    max_steps: int | None = 4096,
    throw: bool = True,
    checkpoints: int | None = None,
    record_history: bool = False,
) -> Solution:
    """Stripped-down :func:`diffrax.diffeqsolve` for circuit simulation.

    Identical calling convention to ``diffrax.diffeqsolve`` except:

    - ``adjoint``, ``event``, ``progress_meter``, ``solver_state``,
      ``controller_state``, and ``made_jump`` arguments are absent.
    - ``t0 < t1`` is assumed; no backward-time integration.
    - No SDE/CDE terms.
    - ``saveat.dense`` and ``saveat.steps`` are not supported.

    ``checkpoints`` controls the number of binomial checkpoints used by
    ``RecursiveCheckpointAdjoint`` (``None`` = auto from ``max_steps``).

    ``record_history``, when ``True``, allocates a ``(t, y)`` buffer of every
    accepted step (sized ``max_steps + 1``, which must therefore be finite)
    and threads it into ``args`` as ``(*args, hist_t, hist_y)`` for solvers
    to use for fixed-delay lookups (see ``circulax.solvers.assembly``'s
    ``hist_t``/``hist_y`` parameters). No-op cost when ``False`` — the
    circuit has no delayed components.
    """
    # ------------------------------------------------------------------
    # dtype promotion for times (same logic as diffrax)
    # ------------------------------------------------------------------
    timelikes = [t0, t1, dt0] + [s.ts for s in jtu.tree_leaves(saveat.subs, is_leaf=_is_subsaveat) if s.ts is not None]
    with jax.numpy_dtype_promotion("standard"):
        time_dtype = jnp.result_type(*timelikes)
    if jnp.issubdtype(time_dtype, jnp.integer):
        time_dtype = lxi.default_floating_dtype()

    t0 = jnp.asarray(t0, dtype=time_dtype)
    t1 = jnp.asarray(t1, dtype=time_dtype)
    dt0 = jnp.asarray(dt0, dtype=time_dtype)

    # Cast save ts to the same dtype
    def _cast_ts(saveat):
        out = [s.ts for s in jtu.tree_leaves(saveat.subs, is_leaf=_is_subsaveat)]
        return [x for x in out if x is not None]

    saveat = eqx.tree_at(_cast_ts, saveat, replace_fn=lambda ts: ts.astype(time_dtype))

    # Promote y0 dtype to be consistent with time (avoids weak-dtype issues)
    def _promote(yi):
        with jax.numpy_dtype_promotion("standard"):
            _dtype = jnp.result_type(yi, time_dtype)
        return jnp.asarray(yi, dtype=_dtype)

    y0 = jtu.tree_map(_promote, y0)

    # ------------------------------------------------------------------
    # Wrap terms with direction=1 (forward-only; WrapTerm still needed
    # so that solver.step receives a properly wrapped term object)
    # ------------------------------------------------------------------
    direction = jnp.asarray(1, dtype=time_dtype)

    def _wrap(term):
        assert isinstance(term, AbstractTerm)
        return WrapTerm(term, direction)

    terms = jtu.tree_map(
        _wrap,
        terms,
        is_leaf=lambda x: isinstance(x, AbstractTerm) and not isinstance(x, diffrax._term.MultiTerm),
    )

    # ------------------------------------------------------------------
    # Propagate PIDController tolerances into implicit solver root finder
    # ------------------------------------------------------------------
    if isinstance(solver, diffrax.AbstractImplicitSolver):
        from diffrax._root_finder import use_stepsize_tol

        def _get_tols(x):
            outs = []
            for attr in ("rtol", "atol", "norm"):
                if getattr(solver.root_finder, attr) is use_stepsize_tol:
                    outs.append(getattr(x, attr))
            return tuple(outs)

        if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
            solver = eqx.tree_at(
                lambda s: _get_tols(s.root_finder),
                solver,
                _get_tols(stepsize_controller),
            )

    # ------------------------------------------------------------------
    # Validate save timestamps
    # ------------------------------------------------------------------
    def _check_ts(ts):
        ts = eqxi.error_if(ts, ts[1:] < ts[:-1], "saveat.ts must be increasing.")
        ts = eqxi.error_if(ts, (ts > t1) | (ts < t0), "saveat.ts must lie between t0 and t1.")
        return ts

    saveat = eqx.tree_at(_cast_ts, saveat, replace_fn=_check_ts)

    # Wrap custom fn with direction (identity when direction=1, but kept for correctness)
    def _wrap_fn(x):
        if _is_subsaveat(x) and x.fn is not save_y:
            direction_fn = lambda t, y, a: x.fn(direction * t, y, a)
            return eqx.tree_at(lambda x: x.fn, x, direction_fn)
        return x

    saveat = jtu.tree_map(_wrap_fn, saveat, is_leaf=_is_subsaveat)

    # ------------------------------------------------------------------
    # Initialise solver & controller states
    # ------------------------------------------------------------------
    tprev = t0
    error_order = solver.error_order(terms)
    tnext, controller_state = stepsize_controller.init(terms, t0, t1, y0, dt0, args, solver.func, error_order)
    tnext = jnp.minimum(tnext, t1)
    solver_state = solver.init(terms, t0, tnext, y0, args)

    # ------------------------------------------------------------------
    # Allocate output buffers
    # ------------------------------------------------------------------
    def _allocate(subsaveat: SubSaveAt) -> SaveState:
        out_size = 0
        if subsaveat.t0:
            out_size += 1
        if subsaveat.ts is not None:
            out_size += len(subsaveat.ts)
        if subsaveat.t1:
            out_size += 1
        struct = eqx.filter_eval_shape(subsaveat.fn, t0, y0, args)
        ts = jnp.full(out_size, jnp.inf, dtype=time_dtype)
        ys = jtu.tree_map(lambda y: jnp.full((out_size,) + y.shape, jnp.inf, dtype=y.dtype), struct)
        return SaveState(ts=ts, ys=ys, save_index=0, saveat_ts_index=0)

    save_state = jtu.tree_map(_allocate, saveat.subs, is_leaf=_is_subsaveat)

    # ------------------------------------------------------------------
    # Allocate the delay-history buffer
    # ------------------------------------------------------------------
    if record_history:
        if max_steps is None:
            msg = "record_history=True requires a finite max_steps to size the delay-history buffer."
            raise ValueError(msg)
        hist_t = jnp.full(max_steps + 1, jnp.inf, dtype=time_dtype)
        hist_y = jnp.full((max_steps + 1, *y0.shape), jnp.inf, dtype=y0.dtype)
    else:
        hist_t = None
        hist_y = None

    # ------------------------------------------------------------------
    # Build initial CircuitState
    # ------------------------------------------------------------------
    init_state = CircuitState(
        y=y0,
        tprev=tprev,
        tnext=tnext,
        solver_state=solver_state,
        controller_state=controller_state,
        result=RESULTS.successful,
        num_steps=0,
        num_accepted_steps=0,
        num_rejected_steps=0,
        save_state=save_state,
        hist_t=hist_t,
        hist_y=hist_y,
    )

    # ------------------------------------------------------------------
    # Choose while-loop variant (checkpointed for AD, lax otherwise)
    # ------------------------------------------------------------------
    if max_steps is None:
        inner_while_loop = ft.partial(_inner_loop, kind="lax")
        outer_while_loop = ft.partial(_outer_loop, kind="lax")
    else:
        inner_while_loop = ft.partial(_inner_loop, kind="checkpointed")
        outer_while_loop = ft.partial(_outer_loop, kind="checkpointed", checkpoints=checkpoints)

    # ------------------------------------------------------------------
    # Run the integration
    # ------------------------------------------------------------------
    final_state = _circuit_loop(
        solver=solver,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        t0=t0,
        t1=t1,
        max_steps=max_steps,
        terms=terms,
        args=args,
        init_state=init_state,
        inner_while_loop=inner_while_loop,
        outer_while_loop=outer_while_loop,
    )

    # ------------------------------------------------------------------
    # Build the Solution (compatible with diffrax.Solution)
    # ------------------------------------------------------------------
    ts = jtu.tree_map(lambda s: s.ts, final_state.save_state, is_leaf=_is_save_state)
    ys = jtu.tree_map(lambda s: s.ys, final_state.save_state, is_leaf=_is_save_state)

    stats = {
        "num_steps": final_state.num_steps,
        "num_accepted_steps": final_state.num_accepted_steps,
        "num_rejected_steps": final_state.num_rejected_steps,
        "max_steps": max_steps,
    }

    sol = Solution(
        t0=t0,
        t1=t1,
        ts=ts,
        ys=ys,
        interpolation=None,
        stats=stats,
        result=final_state.result,
        solver_state=None,
        controller_state=None,
        made_jump=None,
        event_mask=None,
    )

    if throw:
        sol = final_state.result.error_if(sol, jnp.invert(is_okay(final_state.result)))

    return sol
