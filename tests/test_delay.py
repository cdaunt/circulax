"""Tests for fixed time-delay support (see gdsfactory/circulax#2).

Verifies:
  - A delay-line component's transient output matches an analytically
    time-shifted, attenuated/phase-shifted copy of its (simulated) input --
    including two instances of *different* lengths in the same component
    group, exercising the per-instance-varying-tau vmap path.
  - Gradients w.r.t. a delay-driving parameter (``length_um``) match finite
    differences through the checkpointed adjoint.
  - The ``tau >= dt`` guard rejects delays shorter than the step size.
  - Adaptive step-size controllers are rejected for delayed circuits (v1
    requires ConstantStepSize).
"""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from circulax.compiler import compile_netlist
from circulax.components.electronic import Resistor
from circulax.components.photonic import OpticalDelayLine, OpticalSourcePulse
from circulax.solvers import analyze_circuit, setup_transient

_C_UM_PER_S = 2.99792458e14  # speed of light, um/s

# Pulse delayed well past the sigmoid's floor so the DC operating point (which
# has no notion of delay history) is ~0 and doesn't leak into the comparison.
_PULSE_DELAY = 2.0e-12
_PULSE_RISE = 0.1e-12
_N_GROUP = 4.0


def _tau_to_length_um(tau: float, n_group: float = _N_GROUP) -> float:
    return tau * _C_UM_PER_S / n_group


def _phase(length_um: float, neff: float = 2.4, wavelength_nm: float = 1310.0) -> float:
    return 2.0 * jnp.pi * neff * (length_um / wavelength_nm) * 1000.0


def _analytic_source(t):
    return jax.nn.sigmoid((t - _PULSE_DELAY) / _PULSE_RISE)


@pytest.fixture
def two_delay_lines_netlist():
    """Source driving two OpticalDelayLine instances of different lengths, each to its own load."""
    tau1, tau2 = 1e-12, 2e-12
    length1, length2 = _tau_to_length_um(tau1), _tau_to_length_um(tau2)

    models_map = {
        "source": OpticalSourcePulse,
        "delay": OpticalDelayLine,
        "resistor": Resistor,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "I1": {
                "component": "source",
                "settings": {"power": 1.0, "phase": 0.0, "delay": _PULSE_DELAY, "rise": _PULSE_RISE},
            },
            "WG1": {"component": "delay", "settings": {"length_um": length1, "n_group": _N_GROUP, "loss_dB_cm": 0.0}},
            "WG2": {"component": "delay", "settings": {"length_um": length2, "n_group": _N_GROUP, "loss_dB_cm": 0.0}},
            "R1": {"component": "resistor", "settings": {"R": 1.0}},
            "R2": {"component": "resistor", "settings": {"R": 1.0}},
        },
        "connections": {
            "GND,p1": ("I1,p2", "R1,p2", "R2,p2"),
            "I1,p1": ("WG1,p1", "WG2,p1"),
            "WG1,p2": "R1,p1",
            "WG2,p2": "R2,p1",
        },
    }
    return net_dict, models_map, tau1, tau2, length1, length2


def test_delay_line_matches_analytic_shift(two_delay_lines_netlist):
    """Two delay-line instances of different lengths in one group each match their own tau.

    This exercises the case the design explicitly calls out: because tau
    varies per instance, the delayed read must vmap the interpolation itself
    rather than compute one shared delayed vector and slice it.
    """
    net_dict, models_map, tau1, tau2, length1, length2 = two_delay_lines_netlist

    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    linear_strat = analyze_circuit(groups, sys_size, backend="dense", is_complex=True)
    y0 = linear_strat.solve_dc(groups, jnp.zeros(sys_size * 2, dtype=jnp.float64))
    run_transient = setup_transient(groups, linear_strat)

    t0, t1, dt0 = 0.0, 10e-12, 1e-14
    ts = jnp.linspace(t0, t1, 400)
    sol = run_transient(
        t0=t0, t1=t1, dt0=dt0, y0=y0, saveat=diffrax.SaveAt(ts=ts), max_steps=2000, throw=True,
    )

    def complex_at(idx):
        return sol.ys[:, idx] + 1j * sol.ys[:, idx + sys_size]

    v_out1 = complex_at(port_map["WG1,p2"])
    v_out2 = complex_at(port_map["WG2,p2"])

    T1 = jnp.exp(-1j * _phase(length1))
    T2 = jnp.exp(-1j * _phase(length2))
    expected1 = T1 * _analytic_source(ts - tau1)
    expected2 = T2 * _analytic_source(ts - tau2)

    # Loose-ish tolerance: dt0=1e-14 resolves the 0.1ps rise time with ~10
    # steps, so some linear-interpolation error against the analytic sigmoid
    # is expected; a broken delay (wrong tau, ignored per-instance variation,
    # etc.) would miss by order-1, not by this discretization noise.
    assert jnp.max(jnp.abs(v_out1 - expected1)) < 5e-4
    assert jnp.max(jnp.abs(v_out2 - expected2)) < 5e-4

    # The two instances must actually differ -- catches a bug where tau is
    # silently shared across the group instead of read per-instance.
    assert jnp.max(jnp.abs(v_out1 - v_out2)) > 0.5


def test_delay_line_gradient_matches_fd():
    """grad(loss)/d(length_um) through the delay read matches central finite differences."""
    tau = 1e-12
    length_um0 = _tau_to_length_um(tau)

    models_map = {
        "source": OpticalSourcePulse,
        "delay": OpticalDelayLine,
        "resistor": Resistor,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "I1": {
                "component": "source",
                "settings": {"power": 1.0, "phase": 0.0, "delay": _PULSE_DELAY, "rise": _PULSE_RISE},
            },
            "WG1": {"component": "delay", "settings": {"length_um": length_um0, "n_group": _N_GROUP, "loss_dB_cm": 0.0}},
            "R1": {"component": "resistor", "settings": {"R": 1.0}},
        },
        "connections": {
            "GND,p1": ("I1,p2", "R1,p2"),
            "I1,p1": "WG1,p1",
            "WG1,p2": "R1,p1",
        },
    }
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    linear_strat = analyze_circuit(groups, sys_size, backend="dense", is_complex=True)
    y0 = linear_strat.solve_dc(groups, jnp.zeros(sys_size * 2, dtype=jnp.float64))
    run_transient = setup_transient(groups, linear_strat)

    idx_out = port_map["WG1,p2"]
    t0, t1, dt0 = 0.0, 5e-12, 1e-14
    ts = jnp.array([4.5e-12])  # well past tau + rise, in the settled region

    def loss_fn(length_um):
        new_params = eqx.tree_at(lambda p: p.length_um, groups["delay"].params, jnp.array([length_um]))
        new_group = eqx.tree_at(lambda g: g.params, groups["delay"], new_params)
        new_groups = dict(groups)
        new_groups["delay"] = new_group

        sol = run_transient(
            t0=t0, t1=t1, dt0=dt0, y0=y0, saveat=diffrax.SaveAt(ts=ts), max_steps=2000, throw=True,
            args=(new_groups, sys_size),
        )
        v_out = sol.ys[0, idx_out] + 1j * sol.ys[0, idx_out + sys_size]
        return jnp.abs(v_out) ** 2

    grad_val = jax.grad(loss_fn)(length_um0)

    h = 1e-4 * length_um0
    fd = (loss_fn(length_um0 + h) - loss_fn(length_um0 - h)) / (2 * h)

    assert grad_val == pytest.approx(fd, rel=0.1)


def test_delay_shorter_than_dt_raises():
    """tau < dt must be rejected -- the history buffer can't resolve it."""
    models_map = {
        "source": OpticalSourcePulse,
        "delay": OpticalDelayLine,
        "resistor": Resistor,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "I1": {"component": "source", "settings": {"power": 1.0, "delay": _PULSE_DELAY, "rise": _PULSE_RISE}},
            # length chosen so tau (~1e-16 s) << dt0 (1e-14 s) below.
            "WG1": {"component": "delay", "settings": {"length_um": 0.001, "n_group": _N_GROUP}},
            "R1": {"component": "resistor", "settings": {"R": 1.0}},
        },
        "connections": {
            "GND,p1": ("I1,p2", "R1,p2"),
            "I1,p1": "WG1,p1",
            "WG1,p2": "R1,p1",
        },
    }
    groups, sys_size, _port_map = compile_netlist(net_dict, models_map)
    linear_strat = analyze_circuit(groups, sys_size, backend="dense", is_complex=True)
    y0 = linear_strat.solve_dc(groups, jnp.zeros(sys_size * 2, dtype=jnp.float64))
    run_transient = setup_transient(groups, linear_strat)

    with pytest.raises(Exception, match="tau < dt"):
        run_transient(
            t0=0.0, t1=5e-12, dt0=1e-14, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.array([4e-12])), max_steps=1000, throw=True,
        )


def test_delay_requires_constant_step_size(two_delay_lines_netlist):
    """Adaptive step-size controllers are rejected for delayed circuits (v1 limitation)."""
    net_dict, models_map, *_ = two_delay_lines_netlist
    groups, sys_size, _port_map = compile_netlist(net_dict, models_map)
    linear_strat = analyze_circuit(groups, sys_size, backend="dense", is_complex=True)
    y0 = linear_strat.solve_dc(groups, jnp.zeros(sys_size * 2, dtype=jnp.float64))
    run_transient = setup_transient(groups, linear_strat)

    with pytest.raises(ValueError, match="ConstantStepSize"):
        run_transient(
            t0=0.0, t1=5e-12, dt0=1e-14, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.array([4e-12])),
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            max_steps=1000, throw=True,
        )


def test_undelayed_circuit_unaffected(simple_lrc_netlist):
    """A circuit with no delayed components must not allocate/thread a history buffer."""
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, _port_map = compile_netlist(net_dict, models_map)
    assert all(not getattr(g, "has_delay", False) for g in groups.values())

    linear_strat = analyze_circuit(groups, sys_size, backend="dense")
    y0 = linear_strat.solve_dc(groups, jnp.zeros(sys_size, dtype=jnp.float64))
    run_transient = setup_transient(groups, linear_strat)

    sol = run_transient(
        t0=0.0, t1=1e-8, dt0=1e-10, y0=y0, saveat=diffrax.SaveAt(ts=jnp.linspace(0, 1e-8, 20)),
        max_steps=5000, throw=True,
    )
    assert jnp.all(jnp.isfinite(sol.ys))
