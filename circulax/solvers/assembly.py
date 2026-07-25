"""Assembly functions for the transient circuit solver.

Provides functions for evaluating the residual vectors and effective
Jacobian of the discretised circuit equations at each Newton iteration.
Functions are provided in two variants:

- **Full assembly** (:func:`assemble_system_real`, :func:`assemble_system_complex`)
  — evaluates both the residual and the forward-mode Jacobian via
  ``jax.jacfwd``. Used once per timestep to assemble and factor the frozen
  Jacobian in :class:`~circulax.solver.FactorizedTransientSolver`.

- **Residual only** (:func:`assemble_residual_only_real`,
  :func:`assemble_residual_only_complex`) — evaluates only the primal
  residual, with no Jacobian computation. Used inside the Newton loop where
  the Jacobian has already been factored and only needs to be applied.

Each pair has a real and a complex variant. The complex variant operates on
state vectors in unrolled block format — real parts concatenated with imaginary
parts — allowing complex circuit analyses to reuse real-valued sparse linear
algebra kernels.
"""

import functools

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jax import Array

try:
    from bosdi.circulax import OsdiComponentGroup
except ImportError:
    OsdiComponentGroup = None


def _assemble_osdi_group(
    y: Array,
    group,
    alpha: float,
    dt: float,
    *,
    residual_only: bool = False,
) -> tuple[Array, Array, Array]:
    """Evaluate one OSDI group via bosdi and return ``(f_l, q_l, j_eff)``."""
    try:
        from osdi_jax import osdi_eval, osdi_residual_eval
    except ImportError as _bosdi_err:
        raise ImportError(
            "OSDI support requires the 'bosdi' package, which could not be imported. "
            "Install circulax with the 'verilog-a' extra, or install bosdi directly. "
            "Note: bosdi is not available on all platforms (e.g. Windows)."
        ) from _bosdi_err

    try:
        from osdi_jax import osdi_eval_with_handle, osdi_residual_eval_with_handle
        _HAS_TIER3 = True
    except ImportError:
        _HAS_TIER3 = False

    v_all = y[group.var_indices].astype(jnp.float64)

    if residual_only and not group.use_schur_reduction:
        if _HAS_TIER3 and group.handle is not None:
            cur, chg, _ = osdi_residual_eval_with_handle(group.handle, v_all, group.states)
        else:
            cur, chg, _ = osdi_residual_eval(group.model_id, v_all, group.params, group.states)
        j_eff_stub = jnp.zeros(
            (v_all.shape[0], group.num_nodes, group.num_nodes), dtype=cur.dtype,
        )
        return cur, chg, j_eff_stub

    if _HAS_TIER3 and group.handle is not None:
        cur, cond, chg, cap, _ = osdi_eval_with_handle(group.handle, v_all, group.states)
    else:
        cur, cond, chg, cap, _ = osdi_eval(group.model_id, v_all, group.params, group.states)

    G = cond.reshape(-1, group.num_nodes, group.num_nodes)
    C = cap.reshape(-1, group.num_nodes, group.num_nodes)

    if group.use_schur_reduction:
        return _schur_reduce_osdi_stamp(
            v_all=v_all, cur=cur, chg=chg, G=G, C=C,
            alpha=alpha, dt=dt, group=group,
        )

    j_eff = G + (alpha / dt) * C + group.reg_diag
    return cur, chg, j_eff


def _schur_reduce_osdi_stamp(
    *,
    v_all: Array,
    cur: Array,
    chg: Array,
    G: Array,
    C: Array,
    alpha: float,
    dt: float,
    group,
    gmin: float = 1e-12,
) -> tuple[Array, Array, Array]:
    """Schur-reduce the per-device stamp and pad back to num_nodes."""
    N = G.shape[0]
    T = group.num_pins
    I = group.num_nodes - T

    G_TT = G[:, :T, :T]
    G_TI = G[:, :T, T:]
    G_IT = G[:, T:, :T]
    G_II = G[:, T:, T:]
    C_TT = C[:, :T, :T]
    C_TI = C[:, :T, T:]
    C_IT = C[:, T:, :T]
    C_II = C[:, T:, T:]
    cur_T = cur[:, :T]
    cur_I = cur[:, T:]
    chg_T = chg[:, :T]
    chg_I = chg[:, T:]

    eye_I = jnp.eye(I, dtype=G.dtype)

    G_II_reg = G_II + gmin * eye_I
    rhs_dc = jnp.concatenate(
        [G_IT, cur_I[..., None], chg_I[..., None]], axis=-1
    )
    sol_dc = jnp.linalg.solve(G_II_reg, rhs_dc)
    X_dc = sol_dc[..., :T]
    cur_back = sol_dc[..., T]
    chg_back = sol_dc[..., T + 1]

    cur_eff_T = cur_T - jnp.einsum("nij,nj->ni", G_TI, cur_back)
    chg_eff_T = chg_T - jnp.einsum("nij,nj->ni", G_TI, chg_back)

    v_T = v_all[:, :T]
    v_I = v_all[:, T:]
    v_I_pred = cur_back - jnp.einsum("nij,nj->ni", X_dc, v_T)

    a_over_dt = alpha / dt
    A_TT = G_TT + a_over_dt * C_TT
    A_TI = G_TI + a_over_dt * C_TI
    A_IT = G_IT + a_over_dt * C_IT
    A_II = G_II + a_over_dt * C_II
    A_II_reg = A_II + gmin * eye_I
    X_jac = jnp.linalg.solve(A_II_reg, A_IT)
    j_eff_T = A_TT - A_TI @ X_jac

    n = group.num_nodes
    j_padded = jnp.zeros((N, n, n), dtype=j_eff_T.dtype)
    j_padded = j_padded.at[:, :T, :T].set(j_eff_T)
    j_padded = j_padded.at[:, T:, :T].set(X_dc)
    j_padded = j_padded.at[:, T:, T:].set(jnp.broadcast_to(eye_I, (N, I, I)))

    cur_padded = jnp.concatenate([cur_eff_T, v_I - v_I_pred], axis=-1)
    chg_padded = jnp.concatenate([chg_eff_T, jnp.zeros_like(v_I)], axis=-1)
    return cur_padded, chg_padded, j_padded


def _assemble_osdi_gc_separate(
    y: Array,
    group,
) -> tuple[Array, Array]:
    """Return separate G (conductance) and C (capacitance) for one OSDI group.

    Unlike ``_assemble_osdi_group`` (which returns combined j_eff = G + C/dt),
    this splits G and C so callers like AC sweep can form Y(jω) = G + jωC.
    """
    try:
        from osdi_jax import osdi_eval
    except ImportError as _bosdi_err:
        raise ImportError(
            "OSDI support requires the 'bosdi' package."
        ) from _bosdi_err

    try:
        from osdi_jax import osdi_eval_with_handle
        _HAS_TIER3 = True
    except ImportError:
        _HAS_TIER3 = False

    v_all = y[group.var_indices].astype(jnp.float64)

    if _HAS_TIER3 and group.handle is not None:
        _, cond, _, cap, _ = osdi_eval_with_handle(group.handle, v_all, group.states)
    else:
        _, cond, _, cap, _ = osdi_eval(group.model_id, v_all, group.params, group.states)

    n = group.num_nodes
    G = cond.reshape(-1, n, n)
    C = cap.reshape(-1, n, n)

    if group.use_schur_reduction:
        T = group.num_pins
        I = n - T
        gmin = 1e-12
        eye_I = jnp.eye(I, dtype=G.dtype)
        G_II_reg = G[:, T:, T:] + gmin * eye_I
        G_schur = G[:, :T, :T] - G[:, :T, T:] @ jnp.linalg.solve(G_II_reg, G[:, T:, :T])
        C_schur = C[:, :T, :T] - C[:, :T, T:] @ jnp.linalg.solve(G_II_reg, C[:, T:, :T])
        g_padded = jnp.zeros_like(G)
        c_padded = jnp.zeros_like(C)
        g_padded = g_padded.at[:, :T, :T].set(G_schur)
        c_padded = c_padded.at[:, :T, :T].set(C_schur)
        g_padded = g_padded.at[:, T:, T:].set(jnp.broadcast_to(eye_I, (G.shape[0], I, I)))
        return g_padded.reshape(-1) + group.reg_diag.reshape(-1), c_padded.reshape(-1)

    return (G + group.reg_diag).reshape(-1), C.reshape(-1)


def _is_osdi(group) -> bool:
    return OsdiComponentGroup is not None and isinstance(group, OsdiComponentGroup)


# ---------------------------------------------------------------------------
# Fixed time-delay support
# ---------------------------------------------------------------------------
#
# Delayed reads interpolate a per-instance query time against the accepted-
# step history buffer (``hist_t``/``hist_y``, written by circuit_diffeq.py's
# ``_circuit_loop``). Delay ``tau`` may vary per instance within a group (e.g.
# waveguides of different lengths), so the interpolation itself must be
# vmapped over the instance axis -- computing one shared delayed vector and
# then slicing it (as ``y_guess[group.var_indices]`` does for the undelayed
# read) would only be correct if every instance shared the same tau.


def _interp_delayed(hist_t: Array, hist_cols: Array, tau: Array, idx: Array, t1: float) -> Array:
    """Per-instance delayed read via ``jnp.interp``, vmapped over ``idx``'s leading axis.

    Args:
        hist_t: Ascending accepted-step sample times, shape ``(max_steps+1,)``
            (inf-padded tail past the current step count).
        hist_cols: Sample values to interpolate, shape ``(max_steps+1, M)``.
        tau: Per-instance delay, shape ``(N,)``.
        idx: Per-instance column indices into ``hist_cols``, shape ``(N, width)``.
        t1: Current evaluation time.

    Returns:
        Delayed values, shape ``(N, width)`` -- the same layout as
        ``y_guess[idx]`` for the undelayed read.

    """

    def _read_one(tau_i: Array, idx_i: Array) -> Array:
        tq = t1 - tau_i
        cols = hist_cols[:, idx_i]  # (max_steps+1, width)
        return jax.vmap(lambda c: jnp.interp(tq, hist_t, c), in_axes=1)(cols)

    return jax.vmap(_read_one)(tau, idx)


def _group_tau(group, params, dt: float) -> Array:
    """Per-instance tau, guarded against delays shorter than the current step."""
    tau = jax.vmap(group.tau_func)(params)
    msg = f"circulax delay: group '{group.name}' has tau < dt; delayed components require tau >= dt."
    return eqxi.error_if(tau, tau < dt, msg)


def _real_hist_locs(group, params, t1: float, dt: float, hist_t: Array, hist_y: Array) -> Array:
    """Per-instance delayed local-var vector, real system layout."""
    tau = _group_tau(group, params, dt)
    return _interp_delayed(hist_t, hist_y, tau, group.var_indices, t1)


def _complex_hist_locs(group, params, t1: float, dt: float, hist_t: Array, hist_y: Array, half_size: int) -> Array:
    """Per-instance delayed local-var vector, complex (unrolled real/imag) layout."""
    tau = _group_tau(group, params, dt)
    hist_r = _interp_delayed(hist_t, hist_y[:, :half_size], tau, group.var_indices, t1)
    hist_i = _interp_delayed(hist_t, hist_y[:, half_size:], tau, group.var_indices, t1)
    return hist_r + 1j * hist_i


def _real_physics(v: Array, p: Array, group, t1: float) -> tuple[Array, Array]:
    return group.physics_func(y=v, args=p, t=t1)


def _real_physics_hist(v: Array, p: Array, h: Array, group, t1: float) -> tuple[Array, Array]:
    return group.physics_func(y=v, args=p, t=t1, hist=h)


def _complex_physics(vr: Array, vi: Array, p: Array, group, t1: float) -> tuple[Array, Array, Array, Array]:
    v = vr + 1j * vi
    f, q = group.physics_func(y=v, args=p, t=t1)
    return f.real, f.imag, q.real, q.imag


def _complex_physics_hist(vr: Array, vi: Array, p: Array, h: Array, group, t1: float) -> tuple[Array, Array, Array, Array]:
    v = vr + 1j * vi
    f, q = group.physics_func(y=v, args=p, t=t1, hist=h)
    return f.real, f.imag, q.real, q.imag


def _primal_and_jac_real(f, v: Array, p: Array) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    """Compute f(v,p) and its Jacobian w.r.t. v in a single forward sweep.

    Sweeps n unit tangents via ``jax.jvp``, extracting the primal from the
    first sweep rather than computing it separately.  Jacobian is returned in
    ``(n_eqs, n_vars)`` shape to match ``jax.jacfwd`` convention.
    """
    n = v.shape[0]
    g = lambda v_: f(v_, p)  # close over p; differentiate w.r.t. v only
    (f_vals, q_vals), (dfs, dqs) = jax.vmap(lambda e: jax.jvp(g, (v,), (e,)))(jnp.eye(n))
    return (f_vals[0], q_vals[0]), (dfs.T, dqs.T)


def _primal_and_jac_real_hist(f, v: Array, p: Array, h: Array) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    """Same as :func:`_primal_and_jac_real`, with a fixed (non-differentiated) delayed-read ``h``."""
    n = v.shape[0]
    g = lambda v_: f(v_, p, h)  # close over p, h; differentiate w.r.t. v only
    (f_vals, q_vals), (dfs, dqs) = jax.vmap(lambda e: jax.jvp(g, (v,), (e,)))(jnp.eye(n))
    return (f_vals[0], q_vals[0]), (dfs.T, dqs.T)


def _primal_and_jac_complex(
    f, vr: Array, vi: Array, p: Array
) -> tuple[
    tuple[Array, Array, Array, Array],
    tuple[Array, Array, Array, Array],
    tuple[Array, Array, Array, Array],
]:
    """Compute f(vr,vi,p) and its Jacobian w.r.t. (vr, vi) in two forward sweeps.

    Mirrors ``jax.jacfwd(f, argnums=(0, 1))``: sweeps unit tangents for vr
    then vi, extracting the primal from the first sweep.  Each Jacobian block
    is returned in ``(n_eqs, n_vars)`` shape.
    """
    n = vr.shape[0]
    zeros_vr = jnp.zeros_like(vr)
    zeros_vi = jnp.zeros_like(vi)
    g = lambda vr_, vi_: f(vr_, vi_, p)  # close over p; differentiate w.r.t. vr, vi only
    (fr_s, fi_s, qr_s, qi_s), (dfr_r, dfi_r, dqr_r, dqi_r) = jax.vmap(lambda e: jax.jvp(g, (vr, vi), (e, zeros_vi)))(jnp.eye(n))
    _, (dfr_i, dfi_i, dqr_i, dqi_i) = jax.vmap(lambda e: jax.jvp(g, (vr, vi), (zeros_vr, e)))(jnp.eye(n))
    primal = (fr_s[0], fi_s[0], qr_s[0], qi_s[0])
    jac_r = (dfr_r.T, dfi_r.T, dqr_r.T, dqi_r.T)
    jac_i = (dfr_i.T, dfi_i.T, dqr_i.T, dqi_i.T)
    return primal, jac_r, jac_i


def _primal_and_jac_complex_hist(
    f, vr: Array, vi: Array, p: Array, h: Array
) -> tuple[
    tuple[Array, Array, Array, Array],
    tuple[Array, Array, Array, Array],
    tuple[Array, Array, Array, Array],
]:
    """Same as :func:`_primal_and_jac_complex`, with a fixed (non-differentiated) delayed-read ``h``."""
    n = vr.shape[0]
    zeros_vr = jnp.zeros_like(vr)
    zeros_vi = jnp.zeros_like(vi)
    g = lambda vr_, vi_: f(vr_, vi_, p, h)  # close over p, h; differentiate w.r.t. vr, vi only
    (fr_s, fi_s, qr_s, qi_s), (dfr_r, dfi_r, dqr_r, dqi_r) = jax.vmap(lambda e: jax.jvp(g, (vr, vi), (e, zeros_vi)))(jnp.eye(n))
    _, (dfr_i, dfi_i, dqr_i, dqi_i) = jax.vmap(lambda e: jax.jvp(g, (vr, vi), (zeros_vr, e)))(jnp.eye(n))
    primal = (fr_s[0], fi_s[0], qr_s[0], qi_s[0])
    jac_r = (dfr_r.T, dfi_r.T, dqr_r.T, dqi_r.T)
    jac_i = (dfr_i.T, dfi_i.T, dqr_i.T, dqi_i.T)
    return primal, jac_r, jac_i


def assemble_system_real(
    y_guess: Array,
    component_groups: dict,
    t1: float,
    dt: float,
    source_scale: float = 1.0,
    alpha: float = 1.0,
    hist_t: Array | None = None,
    hist_y: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Assemble the residual vectors and effective Jacobian values for a real system.

    For each component group, evaluates the physics at ``t1`` and computes the
    forward-mode Jacobian via ``jax.jacfwd``. The effective Jacobian combines
    the resistive and reactive contributions as ``J_eff = df/dy + (alpha/dt) * dq/dy``,
    where ``alpha=1`` recovers Backward Euler and ``alpha=3/2`` (uniform step) gives BDF2.

    Components are processed in sorted key order to ensure a deterministic
    non-zero layout in the sparse Jacobian, which is required for the
    factorisation step.

    Args:
        y_guess: Current state vector of shape ``(sys_size,)``.
        component_groups: Compiled component groups returned by
            :func:`compile_netlist`, keyed by group name.
        t1: Time at which the system is being evaluated.
        dt: Timestep duration, used to scale the reactive Jacobian block.
        source_scale: Multiplicative scale applied to source amplitudes
            (components whose ``amplitude_param`` is set).  Use ``1.0``
            for a standard evaluation and values in ``(0, 1)`` during
            DC homotopy source stepping.
        alpha: Jacobian scaling factor for the reactive block.  Use ``1.0``
            for Backward Euler, the variable-step BDF2 ``α₀`` coefficient for
            BDF2, or ``1/γ`` for SDIRK3 stages.

    Returns:
        A three-tuple ``(total_f, total_q, jac_vals)`` where:

        - **total_f** — assembled resistive residual, shape ``(sys_size,)``.
        - **total_q** — assembled reactive residual, shape ``(sys_size,)``.
        - **jac_vals** — concatenated non-zero values of the effective Jacobian
            in group-sorted order, ready to be passed to the sparse linear solver.

    """
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)
    vals_list = []

    for k in sorted(component_groups.keys()):
        group = component_groups[k]

        if _is_osdi(group):
            f_l, q_l, j_eff = _assemble_osdi_group(y_guess, group, alpha, dt)
            total_f = total_f.at[group.eq_indices].add(f_l)
            total_q = total_q.at[group.eq_indices].add(q_l)
            vals_list.append(j_eff.reshape(-1))
            continue

        if group.is_fdomain:
            # F-domain component: evaluate admittance at f=0 (DC).
            v_locs = y_guess[group.var_indices]
            Y_mats = jax.vmap(lambda p: group.physics_func(0.0, p))(group.params)
            Y_real = Y_mats.real  # (N, n_ports, n_ports)
            f_l = jnp.einsum("nij,nj->ni", Y_real, v_locs)  # (N, n_ports)
            total_f = total_f.at[group.eq_indices].add(f_l)
            vals_list.append(Y_real.reshape(-1))  # Jacobian = Y at DC
            continue

        v_locs = y_guess[group.var_indices]

        ap = group.amplitude_param
        params = (
            eqx.tree_at(lambda p, _ap=ap: getattr(p, _ap), group.params, getattr(group.params, ap) * source_scale)
            if ap
            else group.params
        )

        # Direct combined bypass — calls _fast_combined ONCE per device per Newton
        # iteration instead of routing through vmap(jvp(fast_physics, v, eye[i]))
        # which fires the custom JVP n times.  Only active when the VA emitter
        # produced a combined_fn (i.e. group.combined_func is not None).
        if group.combined_func is not None:
            f_l, q_l, df_l, dq_l = jax.vmap(
                lambda v, p: group.combined_func(v, p, t1)
            )(v_locs, params)
            total_f = total_f.at[group.eq_indices].add(f_l)
            total_q = total_q.at[group.eq_indices].add(q_l)
            j_eff = df_l + (alpha / dt) * dq_l
            vals_list.append(j_eff.reshape(-1))
            continue

        if group.has_delay and hist_t is not None:
            hist_locs = _real_hist_locs(group, params, t1, dt, hist_t, hist_y)
            physics_at_t1 = functools.partial(_real_physics_hist, group=group, t1=t1)
            (f_l, q_l), (df_l, dq_l) = jax.vmap(
                functools.partial(_primal_and_jac_real_hist, physics_at_t1)
            )(v_locs, params, hist_locs)
        else:
            physics_at_t1 = functools.partial(_real_physics, group=group, t1=t1)
            (f_l, q_l), (df_l, dq_l) = jax.vmap(functools.partial(_primal_and_jac_real, physics_at_t1))(v_locs, params)

        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)
        j_eff = df_l + (alpha / dt) * dq_l
        vals_list.append(j_eff.reshape(-1))

    return total_f, total_q, jnp.concatenate(vals_list)


def assemble_gc_real(
    y_guess: Array,
    component_groups: dict,
) -> tuple[Array, Array]:
    """Return separate G and C COO value arrays at the linearisation point.

    Mirrors :func:`assemble_system_real` but returns ``df/dy`` (conductance) and
    ``dq/dy`` (capacitance) separately instead of combining them as
    ``G + C/dt``.  Frequency-domain groups contribute zero-filled blocks so that
    the returned arrays align with the static COO index arrays produced by
    ``_build_index_arrays`` (which includes all groups).

    Args:
        y_guess: Linearisation point (DC operating point), shape ``(num_vars,)``.
        component_groups: Compiled component groups from :func:`compile_netlist`.

    Returns:
        A two-tuple ``(G_vals, C_vals)`` of real-valued 1-D JAX arrays.  Both
        have the same length as the concatenated ``jac_rows``/``jac_cols`` COO
        index arrays from ``_build_index_arrays``.

    """
    g_vals_list = []
    c_vals_list = []

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        n_entries = int(jnp.array(group.jac_rows).reshape(-1).shape[0])

        if _is_osdi(group):
            g_sep, c_sep = _assemble_osdi_gc_separate(y_guess, group)
            g_vals_list.append(g_sep.reshape(-1).astype(y_guess.dtype))
            c_vals_list.append(c_sep.reshape(-1).astype(y_guess.dtype))
            continue

        if group.is_fdomain:
            # Fdomain groups are re-evaluated per-frequency in ac_sweep.
            # Emit zero blocks so COO alignment with _build_index_arrays is preserved.
            g_vals_list.append(jnp.zeros(n_entries, dtype=y_guess.dtype))
            c_vals_list.append(jnp.zeros(n_entries, dtype=y_guess.dtype))
            continue

        v_locs = y_guess[group.var_indices]

        # Direct combined bypass for VA components with combined_fn.
        if group.combined_func is not None:
            _, _, df_l, dq_l = jax.vmap(
                lambda v, p: group.combined_func(v, p, 0.0)
            )(v_locs, group.params)
            g_vals_list.append(df_l.reshape(-1))
            c_vals_list.append(dq_l.reshape(-1))
            continue

        physics_at_dc = functools.partial(_real_physics, group=group, t1=0.0)

        (_, _), (df_l, dq_l) = jax.vmap(functools.partial(_primal_and_jac_real, physics_at_dc))(v_locs, group.params)

        g_vals_list.append(df_l.reshape(-1))
        c_vals_list.append(dq_l.reshape(-1))

    return jnp.concatenate(g_vals_list), jnp.concatenate(c_vals_list)


def assemble_gc_complex(
    y_guess: Array,
    component_groups: dict,
) -> tuple[Array, Array]:
    """Return separate complex G and C COO value arrays at the linearisation point.

    Complex counterpart of :func:`assemble_gc_real`.  The input state vector is
    in unrolled block format (real parts followed by imaginary parts, shape
    ``(2 * num_vars,)``).  Returns **complex128** COO value arrays aligned with
    the **N-sized** (not 2N) index arrays from ``_build_index_arrays(...,
    is_complex=False)``, suitable for building an N x N complex admittance
    matrix in the AC sweep.

    The complex Jacobian is recovered from the four real Jacobian blocks via
    the Wirtinger formula::

        J_complex = (1/2)(J_RR + J_II) + (j/2)(J_IR - J_RI)

    For holomorphic functions this simplifies to ``J_RR + j * J_IR``; for
    real-valued functions it reduces to ``J_RR + 0j``.

    """
    half = y_guess.shape[0] // 2
    y_real, y_imag = y_guess[:half], y_guess[half:]

    g_vals_list: list[Array] = []
    c_vals_list: list[Array] = []

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        n_entries = int(jnp.array(group.jac_rows).reshape(-1).shape[0])

        if _is_osdi(group):
            g_sep, c_sep = _assemble_osdi_gc_separate(y_real, group)
            g_vals_list.append(g_sep.reshape(-1).astype(jnp.complex128))
            c_vals_list.append(c_sep.reshape(-1).astype(jnp.complex128))
            continue

        if group.is_fdomain:
            g_vals_list.append(jnp.zeros(n_entries, dtype=jnp.complex128))
            c_vals_list.append(jnp.zeros(n_entries, dtype=jnp.complex128))
            continue

        v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]

        if group.combined_func is not None:
            _, _, df_l, dq_l = jax.vmap(lambda v, p: group.combined_func(v, p, 0.0))(v_r, group.params)
            g_vals_list.append(df_l.reshape(-1).astype(jnp.complex128))
            c_vals_list.append(dq_l.reshape(-1).astype(jnp.complex128))
            continue

        physics_split = functools.partial(_complex_physics, group=group, t1=0.0)

        (_, _, _, _), (dfr_r, dfi_r, dqr_r, dqi_r), (dfr_i, dfi_i, dqr_i, dqi_i) = jax.vmap(
            functools.partial(_primal_and_jac_complex, physics_split)
        )(v_r, v_i, group.params)

        g_complex = 0.5 * (dfr_r + dfi_i) + 0.5j * (dfi_r - dfr_i)
        c_complex = 0.5 * (dqr_r + dqi_i) + 0.5j * (dqi_r - dqr_i)

        g_vals_list.append(g_complex.reshape(-1))
        c_vals_list.append(c_complex.reshape(-1))

    return jnp.concatenate(g_vals_list), jnp.concatenate(c_vals_list)


def assemble_gc_complex_2n(
    y_guess: Array,
    component_groups: dict,
) -> tuple[list[Array], list[Array]]:
    """Return four-block G and C COO value arrays for the 2N×2N real-block system.

    Unlike :func:`assemble_gc_complex` which collapses the four Jacobian blocks
    into N×N complex matrices via the Wirtinger formula, this function keeps the
    blocks separate — returning ``[RR, RI, IR, II]`` for both G and C.  This is
    required for circuits containing non-holomorphic operations (e.g.
    ``jnp.real()``) where the Wirtinger N×N system does not fully capture the
    conjugate field response.

    Returns:
        ``(G_blocks, C_blocks)`` where each is a list of four JAX arrays
        ``[RR, RI, IR, II]``.  Each array contains the COO values aligned with
        the N-sized index arrays from ``_build_index_arrays(..., is_complex=False)``.

    """
    half = y_guess.shape[0] // 2
    y_real, y_imag = y_guess[:half], y_guess[half:]

    g_blocks: list[list[Array]] = [[], [], [], []]
    c_blocks: list[list[Array]] = [[], [], [], []]

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        n_entries = int(jnp.array(group.jac_rows).reshape(-1).shape[0])

        if _is_osdi(group):
            g_sep, c_sep = _assemble_osdi_gc_separate(y_real, group)
            flat_g = g_sep.reshape(-1)
            flat_c = c_sep.reshape(-1)
            zeros = jnp.zeros_like(flat_g)
            g_blocks[0].append(flat_g)
            g_blocks[1].append(zeros)
            g_blocks[2].append(zeros)
            g_blocks[3].append(zeros)
            c_blocks[0].append(flat_c)
            c_blocks[1].append(zeros)
            c_blocks[2].append(zeros)
            c_blocks[3].append(zeros)
            continue

        if group.is_fdomain:
            zeros = jnp.zeros(n_entries, dtype=jnp.float64)
            for blk in g_blocks:
                blk.append(zeros)
            for blk in c_blocks:
                blk.append(zeros)
            continue

        v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]

        if group.combined_func is not None:
            _, _, df_l, dq_l = jax.vmap(lambda v, p: group.combined_func(v, p, 0.0))(v_r, group.params)
            flat_g = df_l.reshape(-1)
            flat_c = dq_l.reshape(-1)
            zeros = jnp.zeros_like(flat_g)
            g_blocks[0].append(flat_g)
            g_blocks[1].append(zeros)
            g_blocks[2].append(zeros)
            g_blocks[3].append(zeros)
            c_blocks[0].append(flat_c)
            c_blocks[1].append(zeros)
            c_blocks[2].append(zeros)
            c_blocks[3].append(zeros)
            continue

        physics_split = functools.partial(_complex_physics, group=group, t1=0.0)

        (_, _, _, _), (dfr_r, dfi_r, dqr_r, dqi_r), (dfr_i, dfi_i, dqr_i, dqi_i) = jax.vmap(
            functools.partial(_primal_and_jac_complex, physics_split)
        )(v_r, v_i, group.params)

        g_blocks[0].append(dfr_r.reshape(-1))
        g_blocks[1].append(dfr_i.reshape(-1))
        g_blocks[2].append(dfi_r.reshape(-1))
        g_blocks[3].append(dfi_i.reshape(-1))
        c_blocks[0].append(dqr_r.reshape(-1))
        c_blocks[1].append(dqr_i.reshape(-1))
        c_blocks[2].append(dqi_r.reshape(-1))
        c_blocks[3].append(dqi_i.reshape(-1))

    return [jnp.concatenate(b) for b in g_blocks], [jnp.concatenate(b) for b in c_blocks]


def assemble_residual_only_real(
    y_guess: Array,
    component_groups: dict,
    t1: float,
    dt: float,
    hist_t: Array | None = None,
    hist_y: Array | None = None,
) -> tuple[Array, Array]:
    """Assemble the residual vectors for a real system, without computing the Jacobian.

    Cheaper than :func:`assemble_system_real` as it performs only primal
    evaluations. Used inside the frozen-Jacobian Newton loop where the
    Jacobian has already been factored and only the residual needs to be
    recomputed at each iteration.

    Args:
        y_guess: Current state vector of shape ``(sys_size,)``.
        component_groups: Compiled component groups returned by
            :func:`compile_netlist`, keyed by group name.
        t1: Time at which the system is being evaluated.
        dt: Present for signature symmetry with :func:`assemble_system_real`
            so the two functions are interchangeable at call sites. Only
            used to guard ``tau >= dt`` for groups with delayed components.
        hist_t: Accepted-step sample times for delayed reads, or ``None``
            if the circuit has no delayed components.
        hist_y: Accepted-step state samples for delayed reads, or ``None``.

    Returns:
        A two-tuple ``(total_f, total_q)`` where both arrays have shape
        ``(sys_size,)`` and ``dtype`` matching ``y_guess.dtype``.

    """
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)

    for k in sorted(component_groups.keys()):
        group = component_groups[k]

        if _is_osdi(group):
            f_l, q_l, _ = _assemble_osdi_group(
                y_guess, group, alpha=1.0, dt=1.0, residual_only=True,
            )
            total_f = total_f.at[group.eq_indices].add(f_l)
            total_q = total_q.at[group.eq_indices].add(q_l)
            continue

        if group.is_fdomain:
            # F-domain groups have no time-domain physics; their contribution is
            # added directly in the frequency domain by the HB solver.
            continue

        v = y_guess[group.var_indices]

        if group.has_delay and hist_t is not None:
            hist_locs = _real_hist_locs(group, group.params, t1, dt, hist_t, hist_y)
            f_l, q_l = jax.vmap(functools.partial(_real_physics_hist, group=group, t1=t1))(v, group.params, hist_locs)
        else:
            physics_at_t1 = functools.partial(_real_physics, group=group, t1=t1)
            f_l, q_l = jax.vmap(physics_at_t1)(v, group.params)

        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)

    return total_f, total_q


def assemble_system_complex(
    y_guess: Array,
    component_groups: dict,
    t1: float,
    dt: float,
    source_scale: float = 1.0,
    alpha: float = 1.0,
    hist_t: Array | None = None,
    hist_y: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Assemble the residual vectors and effective Jacobian values for an unrolled complex system.

    The complex state vector is stored in unrolled (block) format: the first
    half of ``y_guess`` holds the real parts of all node voltages/states, the
    second half holds the imaginary parts. This avoids JAX's limited support
    for complex-valued sparse linear solvers by keeping all arithmetic real.

    The Jacobian is split into four real blocks — RR, RI, IR, II — representing
    the partial derivatives of the real and imaginary residual components with
    respect to the real and imaginary state components respectively. The blocks
    are concatenated in RR→RI→IR→II order to match the sparsity index layout
    produced during compilation.

    Args:
        y_guess: Unrolled state vector of shape ``(2 * num_vars,)``, where
            ``y_guess[:num_vars]`` are real parts and ``y_guess[num_vars:]``
            are imaginary parts.
        component_groups: Compiled component groups returned by
            :func:`compile_netlist`, keyed by group name.
        t1: Time at which the system is being evaluated.
        dt: Timestep duration, used to scale the reactive Jacobian blocks.
        source_scale: Multiplicative scale applied to source amplitudes
            (components whose ``amplitude_param`` is set).  Use ``1.0``
            for a standard evaluation and values in ``(0, 1)`` during
            DC homotopy source stepping.
        alpha: Jacobian scaling factor for the reactive blocks.  Use ``1.0``
            for Backward Euler, the variable-step BDF2 ``α₀`` coefficient for
            BDF2, or ``1/γ`` for SDIRK3 stages.

    Returns:
        A three-tuple ``(total_f, total_q, jac_vals)`` where:

        - **total_f** — assembled resistive residual in unrolled format,
            shape ``(2 * num_vars,)``.
        - **total_q** — assembled reactive residual in unrolled format,
            shape ``(2 * num_vars,)``.
        - **jac_vals** — concatenated non-zero values of the four effective
            Jacobian blocks (RR, RI, IR, II) in group-sorted order.

    """
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real, y_imag = y_guess[:half_size], y_guess[half_size:]

    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    total_q = jnp.zeros(sys_size, dtype=jnp.float64)

    vals_blocks: list[list[Array]] = [[], [], [], []]

    for k in sorted(component_groups.keys()):
        group = component_groups[k]

        if _is_osdi(group):
            f_l, q_l, j_eff = _assemble_osdi_group(y_guess[:half_size], group, alpha, dt)
            total_f = total_f.at[group.eq_indices].add(f_l)
            total_q = total_q.at[group.eq_indices].add(q_l)
            vals_blocks[0].append(j_eff.reshape(-1))
            vals_blocks[1].append(jnp.zeros(j_eff.size, dtype=jnp.float64))
            vals_blocks[2].append(jnp.zeros(j_eff.size, dtype=jnp.float64))
            vals_blocks[3].append(jnp.zeros(j_eff.size, dtype=jnp.float64))
            continue

        if group.is_fdomain:
            # F-domain component: evaluate admittance at f=0 (DC) — complex circuit path.
            v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]
            v_c = v_r + 1j * v_i  # (N, n_ports) complex
            Y_mats = jax.vmap(lambda p: group.physics_func(0.0, p))(group.params)
            i_c = jnp.einsum("nij,nj->ni", Y_mats, v_c)  # (N, n_ports) complex
            idx_r, idx_i = group.eq_indices, group.eq_indices + half_size
            total_f = total_f.at[idx_r].add(i_c.real).at[idx_i].add(i_c.imag)
            # Jacobian blocks: dI/dVr = Y.real, dI/dVi = -Y.imag (by Cauchy-Riemann)
            # For general complex Y: dIr/dVr = Yr, dIr/dVi = -Yi, dIi/dVr = Yi, dIi/dVi = Yr
            Yr = Y_mats.real  # (N, n_ports, n_ports)
            Yi = Y_mats.imag
            vals_blocks[0].append(Yr.reshape(-1))  # RR: dIr/dVr
            vals_blocks[1].append((-Yi).reshape(-1))  # RI: dIr/dVi
            vals_blocks[2].append(Yi.reshape(-1))  # IR: dIi/dVr
            vals_blocks[3].append(Yr.reshape(-1))  # II: dIi/dVi
            continue

        v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]

        ap = group.amplitude_param
        params = (
            eqx.tree_at(lambda p, _ap=ap: getattr(p, _ap), group.params, getattr(group.params, ap) * source_scale)
            if ap
            else group.params
        )

        if group.has_delay and hist_t is not None:
            hist_locs = _complex_hist_locs(group, params, t1, dt, hist_t, hist_y, half_size)
            physics_split = functools.partial(_complex_physics_hist, group=group, t1=t1)
            (fr, fi, qr, qi), (dfr_r, dfi_r, dqr_r, dqi_r), (dfr_i, dfi_i, dqr_i, dqi_i) = jax.vmap(
                functools.partial(_primal_and_jac_complex_hist, physics_split)
            )(v_r, v_i, params, hist_locs)
        else:
            physics_split = functools.partial(_complex_physics, group=group, t1=t1)
            (fr, fi, qr, qi), (dfr_r, dfi_r, dqr_r, dqi_r), (dfr_i, dfi_i, dqr_i, dqi_i) = jax.vmap(
                functools.partial(_primal_and_jac_complex, physics_split)
            )(v_r, v_i, params)

        idx_r, idx_i = group.eq_indices, group.eq_indices + half_size
        total_f = total_f.at[idx_r].add(fr).at[idx_i].add(fi)
        total_q = total_q.at[idx_r].add(qr).at[idx_i].add(qi)

        vals_blocks[0].append((dfr_r + (alpha / dt) * dqr_r).reshape(-1))  # RR
        vals_blocks[1].append((dfr_i + (alpha / dt) * dqr_i).reshape(-1))  # RI
        vals_blocks[2].append((dfi_r + (alpha / dt) * dqi_r).reshape(-1))  # IR
        vals_blocks[3].append((dfi_i + (alpha / dt) * dqi_i).reshape(-1))  # II

    all_vals = jnp.concatenate([jnp.concatenate(b) for b in vals_blocks])
    return total_f, total_q, all_vals


def assemble_residual_only_complex(
    y_guess: Array,
    component_groups: dict,
    t1: float,
    dt: float,
    hist_t: Array | None = None,
    hist_y: Array | None = None,
) -> tuple[Array, Array]:
    """Assemble the residual vectors for an unrolled complex system, without computing the Jacobian.

    The complex counterpart of :func:`assemble_residual_only_real`. The state
    vector is expected in unrolled block format (real parts followed by imaginary
    parts) matching the layout used by :func:`assemble_system_complex`.

    Args:
        y_guess: Unrolled state vector of shape ``(2 * num_vars,)``.
        component_groups: Compiled component groups returned by
            :func:`compile_netlist`, keyed by group name.
        t1: Time at which the system is being evaluated.
        dt: Present for signature symmetry with :func:`assemble_system_complex`
            so the two functions are interchangeable at call sites. Only used
            to guard ``tau >= dt`` for groups with delayed components.
        hist_t: Accepted-step sample times for delayed reads, or ``None``
            if the circuit has no delayed components.
        hist_y: Accepted-step state samples for delayed reads, or ``None``.

    Returns:
        A two-tuple ``(total_f, total_q)`` where both arrays have shape
        ``(2 * num_vars,)`` and ``dtype`` matching ``y_guess.dtype``.

    """
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real, y_imag = y_guess[:half_size], y_guess[half_size:]

    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)

    for k in sorted(component_groups.keys()):
        group = component_groups[k]

        if _is_osdi(group):
            f_l, q_l, _ = _assemble_osdi_group(
                y_guess[:half_size], group, alpha=1.0, dt=1.0, residual_only=True,
            )
            total_f = total_f.at[group.eq_indices].add(f_l)
            total_q = total_q.at[group.eq_indices].add(q_l)
            continue

        if group.is_fdomain:
            # F-domain groups have no time-domain physics; their contribution is
            # added directly in the frequency domain by the HB solver.
            continue

        v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]

        if group.has_delay and hist_t is not None:
            hist_locs = _complex_hist_locs(group, group.params, t1, dt, hist_t, hist_y, half_size)
            physics_split = functools.partial(_complex_physics_hist, group=group, t1=t1)
            fr, fi, qr, qi = jax.vmap(physics_split)(v_r, v_i, group.params, hist_locs)
        else:
            physics_split = functools.partial(_complex_physics, group=group, t1=t1)
            fr, fi, qr, qi = jax.vmap(physics_split)(v_r, v_i, group.params)

        idx_r, idx_i = group.eq_indices, group.eq_indices + half_size
        total_f = total_f.at[idx_r].add(fr).at[idx_i].add(fi)
        total_q = total_q.at[idx_r].add(qr).at[idx_i].add(qi)

    return total_f, total_q
