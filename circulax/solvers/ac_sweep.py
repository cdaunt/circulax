"""AC small-signal frequency sweep returning S-parameters.

AC analysis linearises the circuit DAE at a DC operating point and sweeps a
range of frequencies, returning the N-port scattering parameters S(f).

The DAE F(y) + dQ/dt = 0 is linearised at y_dc:

- G = ∂F/∂y|_{y_dc}  (conductance matrix)
- C = ∂Q/∂y|_{y_dc}  (capacitance matrix)

The complex nodal admittance matrix at angular frequency ω is::

    Y(jω) = G + jωC

For N circuit ports with reference impedance Z0::

    Y_total = Y(jω) + diag(1/Z0 at port nodes)
    RHS[:, p] = 2/Z0 at port_nodes[p], zero elsewhere
    V = solve(Y_total, RHS)        # shape (num_vars, N_ports)
    S = V[port_nodes, :] - I      # I = N_ports × N_ports identity

Frequency-domain components contribute Y_fdomain(f) to Y_total at each
frequency point and are evaluated inside the frequency sweep loop.

Example::

    groups, num_vars, pmap = compile_netlist(net_dict, models)
    linear_strat = analyze_circuit(groups, num_vars)
    y_dc = linear_strat.solve_dc(groups, jnp.zeros(num_vars))

    port_nodes = [pmap["R1,p1"]]
    run_ac = setup_ac_sweep(groups, num_vars, port_nodes, z0=50.0)

    freqs = jnp.logspace(6, 10, 100)  # 1 MHz to 10 GHz
    S = run_ac(y_dc, freqs)  # shape (100, 1, 1) complex

    # JIT for repeated sweeps:
    S = jax.jit(run_ac)(y_dc, freqs)
"""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from circulax.solvers.assembly import assemble_gc_complex, assemble_gc_complex_2n, assemble_gc_real
from circulax.solvers.linear import GROUND_STIFFNESS, _build_index_arrays


def _normalize_z0(z0: float | Array, n_ports: int) -> Array:
    """Broadcast z0 to shape ``(N_ports,)``."""
    z0 = jnp.atleast_1d(jnp.asarray(z0, dtype=jnp.complex128))
    if z0.ndim == 0 or (z0.ndim == 1 and z0.shape[0] == 1):
        return jnp.broadcast_to(z0.reshape(1), (n_ports,))
    if z0.ndim == 1 and z0.shape[0] == n_ports:
        return z0
    msg = f"z0 must be a scalar or shape ({n_ports},); got shape {z0.shape}"
    raise ValueError(msg)


def renormalize(
    S: Array,
    z0_from: float | Array,
    z0_to: float | Array,
) -> Array:
    """Renormalize S-parameters from one reference impedance to another.

    Transforms S-parameters computed at impedance ``z0_from`` to the
    equivalent S-parameters at impedance ``z0_to``, using the standard
    power-wave renormalization formula.

    Args:
        S: S-parameter array of shape ``(N_freqs, N_ports, N_ports)``.
        z0_from: Original reference impedance.  Scalar or ``(N_ports,)``
            for frequency-independent, or ``(N_freqs, N_ports)`` for
            frequency-dependent impedance.
        z0_to: Target reference impedance, same shape options as ``z0_from``.

    Returns:
        Renormalized S-parameter array, same shape as ``S``.

    """
    n_freqs, n_ports, _ = S.shape
    z0_from = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(z0_from, dtype=jnp.complex128)), (n_freqs, n_ports))
    z0_to = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(z0_to, dtype=jnp.complex128)), (n_freqs, n_ports))

    def _renorm_one(S_f, zf, zt):
        I = jnp.eye(n_ports, dtype=jnp.complex128)
        gamma = (zt - zf) / (zt + zf)
        Gamma = jnp.diag(gamma)
        A = jnp.diag(jnp.sqrt(zt.real / zf.real) * zf / zt)
        return A @ (S_f - Gamma) @ jnp.linalg.solve(I - Gamma @ S_f, I) @ jnp.linalg.inv(A)

    return jax.vmap(_renorm_one)(S, z0_from, z0_to)


def setup_ac_sweep(
    groups: dict[str, Any],
    num_vars: int,
    port_nodes: list[int],
    *,
    z0: float | Array = 50.0,
    is_complex: bool = False,
    holomorphic: bool = True,
) -> Callable[[Array, Array], Array]:
    """Configure and return a callable for AC small-signal S-parameter sweep.

    Linearises the circuit DAE at the DC operating point and sweeps over an
    array of frequencies, returning the complex S-parameter matrix at each
    frequency.  The returned callable is compatible with :func:`jax.jit` and
    :func:`jax.vmap`.

    The analysis solves ``Y(jω) · V = RHS`` at each frequency, where::

        Y(jω) = G + jωC + Y_fdomain(f) + port_terminations + ground_penalty

    ``G = ∂F/∂y`` and ``C = ∂Q/∂y`` are extracted once at the DC operating
    point.  ``Y_fdomain(f)`` is the admittance contribution from
    frequency-domain components, re-evaluated at each frequency.

    **S-parameter convention** — matched-load verification:

    - Matched load (Z_circuit = Z0) → S11 = 0
    - Open circuit (Z_circuit → ∞) → S11 = +1
    - Short circuit (Z_circuit = 0) → S11 = −1

    Args:
        groups: Compiled component groups from :func:`~circulax.compile_netlist`.
        num_vars: Total number of scalar unknowns (second return value of
            :func:`~circulax.compile_netlist`).
        port_nodes: Global node indices for each circuit port, in the desired
            port ordering.  Obtain from the port-to-node map returned by
            :func:`~circulax.compile_netlist`::

                _, _, pmap = compile_netlist(net_dict, models)
                port_nodes = [pmap["R1,p1"], pmap["C1,p1"]]

        z0: Reference impedance in ohms.  Accepts:

            - **scalar** — same impedance for all ports (default: 50.0).
            - **array of shape** ``(N_ports,)`` — per-port impedance.

            For frequency-dependent impedance, solve at a fixed z0 and use
            :func:`renormalize` afterwards.

        is_complex: If ``True``, use complex-valued assembly for photonic
            circuits.  The DC operating point ``y_dc`` is expected in unrolled
            block format (shape ``(2 * num_vars,)``).
        holomorphic: If ``True`` (default), use the N×N Wirtinger system for
            complex circuits.  Set to ``False`` when any component uses
            non-holomorphic operations (e.g. ``jnp.real()``, ``jnp.abs()``),
            which couple the field and its conjugate — the full 2N×2N
            real-block system is then needed.  Implies ``is_complex=True``.

    Returns:
        A callable ``run_ac(y_dc, freqs) -> S`` where:

        - **y_dc** — DC operating point, shape ``(num_vars,)`` for real
          circuits or ``(2 * num_vars,)`` for complex circuits.
        - **freqs** — frequencies in Hz, shape ``(N_freqs,)``.
        - **S** — S-parameter matrix, shape ``(N_freqs, N_ports, N_ports)``
            complex128.

        Compatible with :func:`jax.jit` and :func:`jax.vmap` over ``y_dc``.

    """
    if not holomorphic:
        is_complex = True

    if 0 in port_nodes:
        msg = "Port node cannot be the ground node (index 0)."
        raise ValueError(msg)

    if not holomorphic:
        return _setup_ac_sweep_2n(groups, num_vars, port_nodes, z0=z0)

    # --- Pre-compute static COO index arrays (captured in closure) -----------
    static_rows, static_cols, ground_idxs, _ = _build_index_arrays(groups, num_vars, is_complex=False)
    static_rows_jax = jnp.array(static_rows)
    static_cols_jax = jnp.array(static_cols)
    ground_indices = jnp.array(ground_idxs)

    N_ports = len(port_nodes)
    port_nodes_arr = jnp.array(port_nodes, dtype=jnp.int32)

    fdomain_scatter: dict[str, tuple[Array, Array]] = {
        gk: (
            jnp.array(groups[gk].jac_rows).reshape(-1),
            jnp.array(groups[gk].jac_cols).reshape(-1),
        )
        for gk in sorted(groups)
        if groups[gk].is_fdomain
    }

    gc_assemble = assemble_gc_complex if is_complex else assemble_gc_real

    z0_arr = _normalize_z0(z0, N_ports)

    # -------------------------------------------------------------------------
    def run_ac(y_dc: Array, freqs: Array) -> Array:
        G_vals, C_vals = gc_assemble(y_dc, groups)

        G_mat = jnp.zeros((num_vars, num_vars), dtype=jnp.complex128)
        G_mat = G_mat.at[static_rows_jax, static_cols_jax].add(G_vals)

        C_mat = jnp.zeros((num_vars, num_vars), dtype=jnp.complex128)
        C_mat = C_mat.at[static_rows_jax, static_cols_jax].add(C_vals)

        RHS = jnp.zeros((num_vars, N_ports), dtype=jnp.complex128)
        RHS = RHS.at[port_nodes_arr, jnp.arange(N_ports)].set(2.0 / z0_arr)

        def _solve_one_freq(f: Array) -> Array:
            omega = 2.0 * jnp.pi * f
            Y = G_mat + 1j * omega * C_mat

            for gk, (rows_fd, cols_fd) in fdomain_scatter.items():
                group_fd = groups[gk]
                Y_mats = jax.vmap(functools.partial(group_fd.physics_func, f))(group_fd.params)
                Y = Y.at[rows_fd, cols_fd].add(Y_mats.reshape(-1))

            Y = Y.at[port_nodes_arr, port_nodes_arr].add(1.0 / z0_arr)
            Y = Y.at[ground_indices, ground_indices].add(GROUND_STIFFNESS)

            V = jnp.linalg.solve(Y, RHS)
            V_ports = V[port_nodes_arr, :]
            return V_ports - jnp.eye(N_ports, dtype=jnp.complex128)

        return jax.vmap(_solve_one_freq)(freqs)

    return run_ac


def _setup_ac_sweep_2n(
    groups: dict[str, Any],
    num_vars: int,
    port_nodes: list[int],
    *,
    z0: float | Array = 50.0,
) -> Callable[[Array, Array], Array]:
    """Build an AC sweep using the full 2N×2N real-block system."""
    N = num_vars
    static_rows, static_cols, ground_idxs, _ = _build_index_arrays(groups, N, is_complex=False)
    rows_j = jnp.array(static_rows)
    cols_j = jnp.array(static_cols)
    ground_2n = jnp.concatenate([jnp.array(ground_idxs), jnp.array(ground_idxs) + N])
    offsets = jnp.array([[0, 0], [0, N], [N, 0], [N, N]])

    N_ports = len(port_nodes)
    port_nodes_arr = jnp.array(port_nodes, dtype=jnp.int32)

    fdomain_scatter: dict[str, tuple[Array, Array]] = {
        gk: (
            jnp.array(groups[gk].jac_rows).reshape(-1),
            jnp.array(groups[gk].jac_cols).reshape(-1),
        )
        for gk in sorted(groups)
        if groups[gk].is_fdomain
    }

    z0_arr = _normalize_z0(z0, N_ports)

    def run_ac_2n(y_dc: Array, freqs: Array) -> Array:
        G_blocks, C_blocks = assemble_gc_complex_2n(y_dc, groups)

        RHS = jnp.zeros((2 * N, N_ports), dtype=jnp.complex128)
        RHS = RHS.at[port_nodes_arr, jnp.arange(N_ports)].set(2.0 / z0_arr)

        def _solve_one_freq(f: Array) -> Array:
            omega = 2.0 * jnp.pi * f
            Y = jnp.zeros((2 * N, 2 * N), dtype=jnp.complex128)
            for k in range(4):
                ro, co = offsets[k]
                vals = (G_blocks[k] + 1j * omega * C_blocks[k]).astype(jnp.complex128)
                Y = Y.at[rows_j + ro, cols_j + co].add(vals)

            for gk, (rows_fd, cols_fd) in fdomain_scatter.items():
                group_fd = groups[gk]
                Y_mats = jax.vmap(functools.partial(group_fd.physics_func, f))(group_fd.params)
                Y_flat = Y_mats.reshape(-1)
                Y = Y.at[rows_fd, cols_fd].add(Y_flat.real)
                Y = Y.at[rows_fd + N, cols_fd + N].add(Y_flat.real)
                Y = Y.at[rows_fd, cols_fd + N].add(-Y_flat.imag)
                Y = Y.at[rows_fd + N, cols_fd].add(Y_flat.imag)

            Y = Y.at[port_nodes_arr, port_nodes_arr].add(1.0 / z0_arr)
            Y = Y.at[port_nodes_arr + N, port_nodes_arr + N].add(1.0 / z0_arr)
            Y = Y.at[ground_2n, ground_2n].add(GROUND_STIFFNESS)

            V = jnp.linalg.solve(Y, RHS)
            V_ports = V[port_nodes_arr, :] + 1j * V[port_nodes_arr + N, :]
            return V_ports - jnp.eye(N_ports, dtype=jnp.complex128)

        return jax.vmap(_solve_one_freq)(freqs)

    return run_ac_2n
