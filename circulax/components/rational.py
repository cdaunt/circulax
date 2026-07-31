"""Factory functions for creating circulax components from vfitax SSModel."""
from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from circulax.components.base_component import CircuitComponent, _extract_param


def rational_component(
    ss: Any,
    name: str = "RationalModel",
    z0: complex = 50.0,
    holomorphic: bool = True,
) -> type[CircuitComponent]:
    """Create a time-domain component from a vfitax SSModel.

    Maps the state-space transfer function H(s) = C diag(1/(s-A)) B + D + s E
    onto circulax's DAE formulation F(y) + dQ/dt = 0:

        Port i:  f[port_i] = (C @ x + D @ v)_i    q[port_i] = (E @ v)_i
        State j: f[x_j]    = -(A_j x_j + (B @ v)_j)   q[x_j]    = x_j

    The resulting component works in AC sweep, harmonic balance, and transient
    with zero solver changes. The SS arrays are stored as JAX-traceable Equinox
    fields, enabling differentiation through the model.

    Args:
        ss: SSModel from vfitax (A, B, C, D, E arrays).
        name: Class name for the generated component.
        z0: Reference impedance (stored for documentation; not used in stamp).
        holomorphic: Whether to use the fast N×N Wirtinger AC path.

    Returns:
        A CircuitComponent subclass with Nc ports and Nc*N states.

    Note:
        Currently requires ``is_complex=True`` in ``analyze_circuit`` and
        ``setup_ac_sweep``, even for electrical (real-valued) circuits. A future
        real-pair block form (``_ss_to_real_pairs``) would halve the system size
        for circuits with conjugate pole pairs.
    """
    A = np.asarray(ss.A, dtype=np.complex128)
    B = np.asarray(ss.B, dtype=np.complex128)
    C = np.asarray(ss.C, dtype=np.complex128)
    D = np.asarray(ss.D, dtype=np.complex128)
    E = np.asarray(ss.E, dtype=np.complex128)

    Nc = D.shape[0]
    n_states = A.shape[0]
    N = n_states // Nc

    min_pole_mag = float(np.min(np.abs(A[:N].real)))
    if min_pole_mag < 1e-14:
        raise ValueError(
            f"SSModel has a pole at or near the origin (min |Re(pole)| = {min_pole_mag:.2e}). "
            f"This would create a singular DC stamp."
        )

    ports = tuple(f"p{i+1}" for i in range(Nc))
    states = tuple(f"x{j}" for j in range(n_states))
    n_p = Nc

    _param_names = ("ss_A", "ss_B", "ss_C", "ss_D", "ss_E")

    def _fast_physics(
        vars_vec: jax.Array,
        params: Any,
        t: float,
        hist: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        ss_A = _extract_param(params, "ss_A")
        ss_B = _extract_param(params, "ss_B")
        ss_C = _extract_param(params, "ss_C")
        ss_D = _extract_param(params, "ss_D")
        ss_E = _extract_param(params, "ss_E")

        v = vars_vec[:n_p].astype(jnp.complex128)
        x = vars_vec[n_p:].astype(jnp.complex128)

        i_port = ss_C @ x + ss_D @ v
        f_state = -(ss_A * x + (ss_B @ v))
        f_vals = jnp.concatenate([i_port, f_state])

        q_port = ss_E @ v
        q_state = x
        q_vals = jnp.concatenate([q_port, q_state])

        return f_vals, q_vals

    def _invoke_physics(
        self: CircuitComponent,
        signals: Any,
        s: Any,
        t: float,
        params: Any,
        hist: Any = None,
    ) -> tuple[dict, dict]:
        ss_A = _extract_param(params, "ss_A")
        ss_B = _extract_param(params, "ss_B")
        ss_C = _extract_param(params, "ss_C")
        ss_D = _extract_param(params, "ss_D")
        ss_E = _extract_param(params, "ss_E")

        v = jnp.array([getattr(signals, p) for p in ports], dtype=jnp.complex128)
        x = jnp.array([getattr(s, st) for st in states], dtype=jnp.complex128)

        i_port = ss_C @ x + ss_D @ v
        f_state = -(ss_A * x + (ss_B @ v))

        f_dict = {p: i_port[i] for i, p in enumerate(ports)}
        f_dict.update({st: f_state[j] for j, st in enumerate(states)})

        q_port = ss_E @ v
        q_dict = {p: q_port[i] for i, p in enumerate(ports)}
        q_dict.update({st: x[j] for j, st in enumerate(states)})

        return f_dict, q_dict

    namespace: dict[str, Any] = {
        "__annotations__": {
            "ss_A": jnp.ndarray,
            "ss_B": jnp.ndarray,
            "ss_C": jnp.ndarray,
            "ss_D": jnp.ndarray,
            "ss_E": jnp.ndarray,
        },
        "ports": ports,
        "states": states,
        "_fast_physics": staticmethod(_fast_physics),
        "_invoke_physics": _invoke_physics,
        "_uses_time": False,
        "_holomorphic": holomorphic,
        "amplitude_param": "",
        "_has_init_arg": False,
        "_has_hist_arg": False,
        "_has_delay": False,
        "_static_param_names": (),
        "_diff_param_names": _param_names,
        "ss_A": eqx.field(default_factory=lambda: jnp.array(A)),
        "ss_B": eqx.field(default_factory=lambda: jnp.array(B)),
        "ss_C": eqx.field(default_factory=lambda: jnp.array(C)),
        "ss_D": eqx.field(default_factory=lambda: jnp.array(D)),
        "ss_E": eqx.field(default_factory=lambda: jnp.array(E)),
    }

    cls = type(name, (CircuitComponent,), namespace)
    return cls


def rational_fdomain_component(
    ss: Any,
    name: str = "RationalFdomainModel",
) -> type[CircuitComponent]:
    """Create a frequency-domain oracle component from a vfitax SSModel.

    Evaluates H(j2*pi*f) directly. Works in AC sweep and harmonic balance
    but not transient. Used as a test reference for rational_component.

    Args:
        ss: SSModel from vfitax.
        name: Class name for the generated component.

    Returns:
        A CircuitComponent subclass with _is_fdomain = True.

    """
    A_val = jnp.array(np.asarray(ss.A, dtype=np.complex128))
    B_val = jnp.array(np.asarray(ss.B, dtype=np.complex128))
    C_val = jnp.array(np.asarray(ss.C, dtype=np.complex128))
    D_val = jnp.array(np.asarray(ss.D, dtype=np.complex128))
    E_val = jnp.array(np.asarray(ss.E, dtype=np.complex128))

    Nc = D_val.shape[0]
    ports = tuple(f"p{i+1}" for i in range(Nc))

    def _fast_physics(f: float, args: Any) -> jnp.ndarray:
        ss_A = _extract_param(args, "ss_A")
        ss_B = _extract_param(args, "ss_B")
        ss_C = _extract_param(args, "ss_C")
        ss_D = _extract_param(args, "ss_D")
        ss_E = _extract_param(args, "ss_E")
        s = 1j * 2.0 * jnp.pi * f
        resolvent = 1.0 / (s - ss_A)
        return ss_C @ (resolvent[:, None] * ss_B) + ss_D + s * ss_E

    @classmethod
    def solver_call(cls, f: float, args: Any) -> jnp.ndarray:
        return cls._fast_physics(f, args)

    namespace: dict[str, Any] = {
        "__annotations__": {
            "ss_A": jnp.ndarray,
            "ss_B": jnp.ndarray,
            "ss_C": jnp.ndarray,
            "ss_D": jnp.ndarray,
            "ss_E": jnp.ndarray,
        },
        "ports": ports,
        "states": (),
        "_is_fdomain": True,
        "_uses_time": False,
        "_fast_physics": staticmethod(_fast_physics),
        "solver_call": solver_call,
        "ss_A": eqx.field(default_factory=lambda: A_val),
        "ss_B": eqx.field(default_factory=lambda: B_val),
        "ss_C": eqx.field(default_factory=lambda: C_val),
        "ss_D": eqx.field(default_factory=lambda: D_val),
        "ss_E": eqx.field(default_factory=lambda: E_val),
    }

    cls = type(name, (CircuitComponent,), namespace)
    return cls
