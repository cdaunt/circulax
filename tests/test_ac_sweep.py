"""Tests for setup_ac_sweep: correctness, JIT, vmap, and passivity."""

import jax
import jax.numpy as jnp
import pytest

from circulax.compiler import compile_netlist
from circulax.solvers import analyze_circuit, setup_ac_sweep

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_R = 50.0  # series resistor (ohms)
_C = 1e-9  # shunt capacitor (farads)
_Z0 = 50.0  # reference impedance


@pytest.fixture
def rc_netlist():
    """Single-port parallel RC circuit: R and C both shunt to GND.

    Port is at the shared node R1,p1 = C1,p1.  The admittance seen from the
    port is Y_circuit = 1/R + jωC, so the analytical S11 is:

        Y_total = 1/Z0 + 1/R + jωC
        S11 = (2/Z0) / Y_total - 1
    """
    from circulax.components.electronic import Capacitor, Resistor

    models_map = {
        "resistor": Resistor,
        "capacitor": Capacitor,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "R1": {"component": "resistor", "settings": {"R": _R}},
            "C1": {"component": "capacitor", "settings": {"C": _C}},
        },
        "connections": {
            "R1,p1": "C1,p1",  # port node: R1,p1 == C1,p1
            "R1,p2": "GND,p1",
            "C1,p2": "GND,p1",
        },
    }
    return net_dict, models_map


@pytest.fixture
def rc_setup(rc_netlist):
    """Compiled RC circuit with DC solution and run_ac callable."""
    net_dict, models_map = rc_netlist
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))
    port_nodes = [pmap["R1,p1"]]
    run_ac = setup_ac_sweep(groups, num_vars, port_nodes, z0=_Z0)
    return run_ac, y_dc


def _analytical_s11(freqs: jnp.ndarray, R: float = _R, C: float = _C, z0: float = _Z0) -> jnp.ndarray:
    """Analytical S11 for the parallel-RC test circuit.

    Y_circuit = 1/R + jωC, so:
        Y_total = 1/z0 + 1/R + jωC
        S11 = (2/z0) / Y_total - 1
    """
    omega = 2.0 * jnp.pi * freqs
    Y_total = 1.0 / z0 + 1.0 / R + 1j * omega * C
    V_port = (2.0 / z0) / Y_total
    return V_port - 1.0


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

_FREQS = jnp.logspace(6, 10, 50)  # 1 MHz to 10 GHz


def test_ac_sweep_shapes(rc_setup):
    """run_ac returns (N_freqs, N_ports, N_ports) complex array."""
    run_ac, y_dc = rc_setup
    S = run_ac(y_dc, _FREQS)
    assert S.shape == (len(_FREQS), 1, 1)
    assert jnp.iscomplexobj(S)


def test_ac_sweep_finite(rc_setup):
    """All S-parameter values are finite."""
    run_ac, y_dc = rc_setup
    S = run_ac(y_dc, _FREQS)
    assert jnp.isfinite(jnp.abs(S)).all()


def test_ac_sweep_s11_analytical(rc_setup):
    """S11 matches the analytical RC formula to within 1e-6."""
    run_ac, y_dc = rc_setup
    S = run_ac(y_dc, _FREQS)
    S11 = S[:, 0, 0]
    S11_ref = _analytical_s11(_FREQS)
    assert jnp.allclose(S11, S11_ref, atol=1e-6), f"Max error: {jnp.max(jnp.abs(S11 - S11_ref)):.2e}"


def test_ac_sweep_passivity(rc_setup):
    """Passive circuit satisfies |S11| <= 1 at all frequencies."""
    run_ac, y_dc = rc_setup
    S = run_ac(y_dc, _FREQS)
    assert (jnp.abs(S[:, 0, 0]) <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# Limiting cases
# ---------------------------------------------------------------------------


def test_ac_sweep_dc_limit(rc_netlist):
    """At f→0, S11 → (R_parallel - Z0)/(R_parallel + Z0) (capacitor becomes open).

    For parallel RC, at DC the capacitor is open so Y_circuit = 1/R only.
    """
    net_dict, models_map = rc_netlist
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))
    run_ac = setup_ac_sweep(groups, num_vars, [pmap["R1,p1"]], z0=_Z0)

    freqs_low = jnp.array([1e-3])  # essentially DC
    S = run_ac(y_dc, freqs_low)
    S11_dc = S[0, 0, 0]
    # At DC: Y_total = 1/Z0 + 1/R, V_port = (2/Z0)/(1/Z0 + 1/R), S11 = V_port - 1
    Y_total_dc = 1.0 / _Z0 + 1.0 / _R
    expected = (2.0 / _Z0) / Y_total_dc - 1.0  # = 0.0 for R=Z0=50
    assert jnp.abs(S11_dc - expected) < 1e-4


def test_ac_sweep_matched_load():
    """Pure shunt resistive load R = Z0 gives S11 = 0 at all frequencies."""
    from circulax.components.electronic import Resistor

    # Two identical shunt resistors in parallel = R/2.  Set R = 2*Z0 so the
    # parallel combination equals Z0, giving a matched load (S11 = 0).
    R_each = 2.0 * _Z0  # parallel combination = Z0
    models_map = {"resistor": Resistor, "ground": lambda: 0}
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "R1": {"component": "resistor", "settings": {"R": R_each}},
            "R2": {"component": "resistor", "settings": {"R": R_each}},
        },
        "connections": {
            "R1,p1": "R2,p1",  # port node shared between R1 and R2
            "R1,p2": "GND,p1",
            "R2,p2": "GND,p1",
        },
    }
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))
    run_ac = setup_ac_sweep(groups, num_vars, [pmap["R1,p1"]], z0=_Z0)

    freqs = jnp.logspace(3, 9, 20)
    S = run_ac(y_dc, freqs)
    assert jnp.allclose(jnp.abs(S[:, 0, 0]), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Per-port and per-frequency z0
# ---------------------------------------------------------------------------


def test_z0_per_port_matches_scalar(rc_netlist):
    """z0 as a length-1 array gives the same result as the scalar."""
    net_dict, models_map = rc_netlist
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))
    port_nodes = [pmap["R1,p1"]]

    run_scalar = setup_ac_sweep(groups, num_vars, port_nodes, z0=_Z0)
    run_array = setup_ac_sweep(groups, num_vars, port_nodes, z0=jnp.array([_Z0]))

    S_scalar = run_scalar(y_dc, _FREQS)
    S_array = run_array(y_dc, _FREQS)
    assert jnp.allclose(S_scalar, S_array, atol=1e-10)


def test_z0_per_port_analytical(rc_netlist):
    """Per-port z0 gives the correct analytical S11 for a different impedance."""
    net_dict, models_map = rc_netlist
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))

    z0_alt = 75.0
    run_ac = setup_ac_sweep(groups, num_vars, [pmap["R1,p1"]], z0=jnp.array([z0_alt]))
    S = run_ac(y_dc, _FREQS)
    S11_ref = _analytical_s11(_FREQS, z0=z0_alt)
    assert jnp.allclose(S[:, 0, 0], S11_ref, atol=1e-6)


def test_renormalize_roundtrip(rc_netlist):
    """Renormalize from z0=50 to z0=75, then back to z0=50 gives the original."""
    from circulax.solvers import renormalize

    net_dict, models_map = rc_netlist
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))

    run_ac = setup_ac_sweep(groups, num_vars, [pmap["R1,p1"]], z0=_Z0)
    S_50 = run_ac(y_dc, _FREQS)

    S_75 = renormalize(S_50, z0_from=50.0, z0_to=75.0)
    S_back = renormalize(S_75, z0_from=75.0, z0_to=50.0)
    assert jnp.allclose(S_50, S_back, atol=1e-10), f"Roundtrip error: {jnp.max(jnp.abs(S_50 - S_back)):.2e}"


def test_renormalize_analytical(rc_netlist):
    """Renormalizing S(z0=50) to z0=75 matches direct solve at z0=75."""
    from circulax.solvers import renormalize

    net_dict, models_map = rc_netlist
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))

    S_50 = setup_ac_sweep(groups, num_vars, [pmap["R1,p1"]], z0=50.0)(y_dc, _FREQS)
    S_75_renorm = renormalize(S_50, z0_from=50.0, z0_to=75.0)

    S_75_direct = setup_ac_sweep(groups, num_vars, [pmap["R1,p1"]], z0=75.0)(y_dc, _FREQS)
    assert jnp.allclose(S_75_renorm, S_75_direct, atol=1e-6), (
        f"Max error: {jnp.max(jnp.abs(S_75_renorm - S_75_direct)):.2e}"
    )


# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------


def test_ac_sweep_jit(rc_setup):
    """jax.jit(run_ac) matches the eager result."""
    run_ac, y_dc = rc_setup
    S_eager = run_ac(y_dc, _FREQS)
    S_jit = jax.jit(run_ac)(y_dc, _FREQS)
    assert jnp.allclose(S_eager, S_jit, atol=1e-10)


# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------


def test_ac_sweep_vmap_over_ydc(rc_netlist):
    """jax.vmap(run_ac, in_axes=(0, None)) batches correctly over y_dc."""
    net_dict, models_map = rc_netlist
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))
    run_ac = setup_ac_sweep(groups, num_vars, [pmap["R1,p1"]], z0=_Z0)

    # Batch of 3 identical DC points
    y_dc_batch = jnp.stack([y_dc, y_dc, y_dc])
    S_batch = jax.vmap(run_ac, in_axes=(0, None))(y_dc_batch, _FREQS)

    assert S_batch.shape == (3, len(_FREQS), 1, 1)
    S_single = run_ac(y_dc, _FREQS)
    assert jnp.allclose(S_batch[0], S_single, atol=1e-10)
    assert jnp.allclose(S_batch[1], S_single, atol=1e-10)


# ---------------------------------------------------------------------------
# F-domain component
# ---------------------------------------------------------------------------


def test_ac_sweep_fdomain():
    """S11 with an f-domain resistor matches the analytical skin-effect formula."""
    from circulax.s_transforms import fdomain_component

    @fdomain_component(ports=("p1", "p2"))
    def SkinResistor(f: float, R0: float = 1.0, a: float = 0.1):
        """Skin-effect resistor: Z(f) = R0 + a*sqrt(|f|)."""
        Z = R0 + a * jnp.sqrt(jnp.abs(f) + 1e-30)
        Y = 1.0 / Z
        return jnp.array([[Y, -Y], [-Y, Y]], dtype=jnp.complex128)

    R0, a = 25.0, 1e-5
    from circulax.components.electronic import Resistor

    models_map = {"skin_r": SkinResistor, "resistor": Resistor, "ground": lambda: 0}
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "SR1": {"component": "skin_r", "settings": {"R0": R0, "a": a}},
            # Huge shunt resistor to create the port node without affecting the result
            "Rbig": {"component": "resistor", "settings": {"R": 1e15}},
        },
        "connections": {
            "SR1,p1": "Rbig,p1",  # port node
            "SR1,p2": "GND,p1",
            "Rbig,p2": "GND,p1",
        },
    }
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))
    run_ac = setup_ac_sweep(groups, num_vars, [pmap["SR1,p1"]], z0=_Z0)

    freqs = jnp.logspace(3, 9, 30)
    S = run_ac(y_dc, freqs)
    S11 = S[:, 0, 0]

    # Analytical S11 for skin-effect load
    Z_circuit = R0 + a * jnp.sqrt(jnp.abs(freqs) + 1e-30)
    Y_total = 1.0 / _Z0 + 1.0 / Z_circuit
    S11_ref = (2.0 / _Z0) / Y_total - 1.0

    assert jnp.allclose(S11, S11_ref, atol=1e-6), f"Max fdomain error: {jnp.max(jnp.abs(S11 - S11_ref)):.2e}"


# ---------------------------------------------------------------------------
# Ground node guard
# ---------------------------------------------------------------------------


def test_ac_sweep_ground_node_raises(rc_netlist):
    """Passing node 0 (ground) as a port raises ValueError."""
    net_dict, models_map = rc_netlist
    groups, num_vars, _ = compile_netlist(net_dict, models_map)
    with pytest.raises(ValueError, match="ground"):
        setup_ac_sweep(groups, num_vars, [0], z0=_Z0)


# ---------------------------------------------------------------------------
# Complex (photonic) AC sweep
# ---------------------------------------------------------------------------

_OPT_Z0 = 1.0


@pytest.fixture
def waveguide_netlist():
    """Two-port lossless waveguide circuit for complex AC tests.

    The waveguide S-matrix is [[0, T], [T, 0]] where T = exp(-j*phi).
    No sources — pure passive photonic component.
    """
    from circulax.components.photonic import OpticalWaveguide

    models_map = {
        "waveguide": OpticalWaveguide,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "WG1": {
                "component": "waveguide",
                "settings": {
                    "length_um": 100.0,
                    "loss_dB_cm": 0.0,
                    "neff": 2.4,
                    "n_group": 4.0,
                    "center_wavelength_nm": 1310.0,
                    "wavelength_nm": 1310.0,
                },
            },
        },
        "connections": {
            "WG1,p2": "GND,p1",
        },
        "ports": {"in": "WG1,p1"},
    }
    return net_dict, models_map


@pytest.fixture
def waveguide_complex_setup(waveguide_netlist):
    """Compiled complex waveguide circuit with DC solution and run_ac callable."""
    net_dict, models_map = waveguide_netlist
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars, is_complex=True)
    sys_size = num_vars * 2
    y_dc = solver.solve_dc(groups, jnp.zeros(sys_size))
    port_nodes = [pmap["WG1,p1"]]
    run_ac = setup_ac_sweep(groups, num_vars, port_nodes, z0=_OPT_Z0, is_complex=True)
    return run_ac, y_dc


def test_complex_ac_sweep_shapes(waveguide_complex_setup):
    """Complex AC run_ac returns (N_freqs, N_ports, N_ports) complex array."""
    run_ac, y_dc = waveguide_complex_setup
    freqs = jnp.logspace(6, 10, 20)
    S = run_ac(y_dc, freqs)
    assert S.shape == (len(freqs), 1, 1)
    assert jnp.iscomplexobj(S)


def test_complex_ac_sweep_finite(waveguide_complex_setup):
    """All S-parameter values are finite."""
    run_ac, y_dc = waveguide_complex_setup
    freqs = jnp.logspace(6, 10, 20)
    S = run_ac(y_dc, freqs)
    assert jnp.isfinite(jnp.abs(S)).all()


def test_complex_ac_sweep_passivity(waveguide_complex_setup):
    """Passive photonic circuit satisfies |S11| <= 1 at all frequencies."""
    run_ac, y_dc = waveguide_complex_setup
    freqs = jnp.logspace(6, 10, 50)
    S = run_ac(y_dc, freqs)
    assert (jnp.abs(S[:, 0, 0]) <= 1.0 + 1e-6).all()


def test_complex_ac_sweep_jit(waveguide_complex_setup):
    """jax.jit(run_ac) matches the eager result for complex circuits."""
    run_ac, y_dc = waveguide_complex_setup
    freqs = jnp.logspace(6, 10, 20)
    S_eager = run_ac(y_dc, freqs)
    S_jit = jax.jit(run_ac)(y_dc, freqs)
    assert jnp.allclose(S_eager, S_jit, atol=1e-10)


def test_complex_sp_via_circuit():
    """Circuit.sp() works for complex-valued (photonic) circuits."""
    from circulax.circuit import compile_circuit
    from circulax.components.photonic import OpticalWaveguide

    models_map = {
        "waveguide": OpticalWaveguide,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "WG1": {
                "component": "waveguide",
                "settings": {
                    "length_um": 100.0,
                    "loss_dB_cm": 0.0,
                },
            },
        },
        "connections": {
            "WG1,p2": "GND,p1",
        },
        "ports": {"in": "WG1,p1"},
    }
    circuit = compile_circuit(net_dict, models_map, is_complex=True)
    freqs = jnp.logspace(6, 10, 10)
    S = circuit.sp(ports="in", freqs=freqs, z0=_OPT_Z0)
    assert S.shape == (len(freqs), 1, 1)
    assert jnp.iscomplexobj(S)
    assert jnp.isfinite(jnp.abs(S)).all()


def test_ac_deprecation_warning():
    """Circuit.ac() emits a DeprecationWarning and delegates to sp()."""
    import warnings

    from circulax.circuit import compile_circuit
    from circulax.components.photonic import OpticalWaveguide

    models_map = {"waveguide": OpticalWaveguide, "ground": lambda: 0}
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "WG1": {"component": "waveguide", "settings": {"length_um": 100.0, "loss_dB_cm": 0.0}},
        },
        "connections": {"WG1,p2": "GND,p1"},
        "ports": {"in": "WG1,p1"},
    }
    circuit = compile_circuit(net_dict, models_map, is_complex=True)
    freqs = jnp.logspace(6, 10, 5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        S_ac = circuit.ac(ports="in", freqs=freqs, z0=_OPT_Z0)
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "sp()" in str(w[0].message)
    S_sp = circuit.sp(ports="in", freqs=freqs, z0=_OPT_Z0)
    assert jnp.allclose(S_ac, S_sp, atol=1e-12)


# ---------------------------------------------------------------------------
# Non-holomorphic (2N block) AC sweep
# ---------------------------------------------------------------------------


def test_non_holomorphic_matches_holomorphic_for_waveguide(waveguide_complex_setup):
    """For a holomorphic circuit, holomorphic=False gives the same S-params."""
    run_ac, y_dc = waveguide_complex_setup
    freqs = jnp.logspace(6, 10, 20)
    S_wirtinger = run_ac(y_dc, freqs)

    from circulax.components.photonic import OpticalWaveguide

    models_map = {"waveguide": OpticalWaveguide, "ground": lambda: 0}
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "WG1": {
                "component": "waveguide",
                "settings": {"length_um": 100.0, "loss_dB_cm": 0.0, "neff": 2.4, "n_group": 4.0,
                              "center_wavelength_nm": 1310.0, "wavelength_nm": 1310.0},
            },
        },
        "connections": {"WG1,p2": "GND,p1"},
        "ports": {"in": "WG1,p1"},
    }
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars, is_complex=True)
    sys_size = num_vars * 2
    y_dc_2 = solver.solve_dc(groups, jnp.zeros(sys_size))
    run_ac_2n = setup_ac_sweep(groups, num_vars, [pmap["WG1,p1"]], z0=_OPT_Z0, holomorphic=False)
    S_2n = run_ac_2n(y_dc_2, freqs)

    assert S_2n.shape == S_wirtinger.shape
    assert jnp.allclose(S_2n, S_wirtinger, atol=1e-6), (
        f"Max complex error: {jnp.max(jnp.abs(S_2n - S_wirtinger)):.2e}"
    )


def test_non_holomorphic_via_circuit():
    """Circuit.sp(holomorphic=False) works for complex circuits."""
    from circulax.circuit import compile_circuit
    from circulax.components.photonic import OpticalWaveguide

    models_map = {"waveguide": OpticalWaveguide, "ground": lambda: 0}
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "WG1": {"component": "waveguide", "settings": {"length_um": 100.0, "loss_dB_cm": 0.0}},
        },
        "connections": {"WG1,p2": "GND,p1"},
        "ports": {"in": "WG1,p1"},
    }
    circuit = compile_circuit(net_dict, models_map, is_complex=True)
    freqs = jnp.logspace(6, 10, 10)
    S = circuit.sp(ports="in", freqs=freqs, z0=_OPT_Z0, holomorphic=False)
    assert S.shape == (len(freqs), 1, 1)
    assert jnp.iscomplexobj(S)
    assert jnp.isfinite(jnp.abs(S)).all()

    S_std = circuit.sp(ports="in", freqs=freqs, z0=_OPT_Z0)
    assert jnp.allclose(S, S_std, atol=1e-6), (
        f"Max complex error: {jnp.max(jnp.abs(S - S_std)):.2e}"
    )


def test_non_holomorphic_lossy_waveguide():
    """2N path matches Wirtinger for a lossy waveguide where |S11| < 1."""
    from circulax.components.photonic import OpticalWaveguide

    models_map = {"waveguide": OpticalWaveguide, "ground": lambda: 0}
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "WG1": {
                "component": "waveguide",
                "settings": {"length_um": 500.0, "loss_dB_cm": 3.0, "neff": 2.4, "n_group": 4.0,
                              "center_wavelength_nm": 1310.0, "wavelength_nm": 1310.0},
            },
        },
        "connections": {"WG1,p2": "GND,p1"},
        "ports": {"in": "WG1,p1"},
    }
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars, is_complex=True)
    sys_size = num_vars * 2
    y_dc = solver.solve_dc(groups, jnp.zeros(sys_size))
    freqs = jnp.logspace(6, 10, 20)

    run_ac_w = setup_ac_sweep(groups, num_vars, [pmap["WG1,p1"]], z0=_OPT_Z0, is_complex=True)
    S_wirtinger = run_ac_w(y_dc, freqs)

    run_ac_2n = setup_ac_sweep(groups, num_vars, [pmap["WG1,p1"]], z0=_OPT_Z0, holomorphic=False)
    S_2n = run_ac_2n(y_dc, freqs)

    assert jnp.any(jnp.abs(S_2n[:, 0, 0]) < 0.99), "Lossy waveguide should have |S11| < 1"
    assert jnp.allclose(S_2n, S_wirtinger, atol=1e-6), (
        f"Max complex error: {jnp.max(jnp.abs(S_2n - S_wirtinger)):.2e}"
    )


def test_non_holomorphic_jit(waveguide_netlist):
    """jax.jit works with the non-holomorphic AC sweep."""
    net_dict, models_map = waveguide_netlist
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars, is_complex=True)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars * 2))
    run_ac = setup_ac_sweep(groups, num_vars, [pmap["WG1,p1"]], z0=_OPT_Z0, holomorphic=False)
    freqs = jnp.logspace(6, 10, 10)
    S_eager = run_ac(y_dc, freqs)
    S_jit = jax.jit(run_ac)(y_dc, freqs)
    assert jnp.allclose(S_eager, S_jit, atol=1e-10)


# ---------------------------------------------------------------------------
# Ring modulator: holomorphic=False required for EO power modulation
# ---------------------------------------------------------------------------


def _ring_modulator_circuit():
    """Build a ring modulator circuit with non-holomorphic jnp.real()."""
    import numpy as np

    from circulax.circuit import compile_circuit
    from circulax.components.base_component import component, source
    from circulax.components.electronic import Capacitor, Resistor

    @source(ports=("p1", "p2"), states=("i_src",), holomorphic=True)
    def OpticalCW(signals, s, t, power=1.0, phase=0.0):
        amp = jnp.sqrt(power) * jnp.exp(1j * phase)
        return {"p1": s.i_src, "p2": -s.i_src, "i_src": (signals.p1 - signals.p2) - amp}, {}

    @source(ports=("p1", "p2"), states=("i_src",), holomorphic=True)
    def DCVoltage(signals, s, t, V_dc=-2.0):
        return {"p1": s.i_src, "p2": -s.i_src, "i_src": (signals.p1 - signals.p2) - V_dc}, {}

    @component(ports=("p1", "p2", "v_e"), states=("a", "i_out"), holomorphic=False)
    def RingEO(signals, s, ng=3.8, L=3.14159265e-5, gamma=0.976, alpha0=0.969,
               alpha1=0.0, f_operating=2.2904e14, f_resonance=2.2901e14, v_to_wr=0.0):
        c_val = 2.998e8
        voltage = jnp.real(signals.v_e)  # <-- non-holomorphic
        tau_e = 2 * ng * L / ((1 - gamma**2) * c_val)
        alpha_v = alpha0 + alpha1 * voltage
        tau_l = 2 * ng * L / ((1 - alpha_v**2) * c_val)
        tau = 1 / (1 / tau_e + 1 / tau_l)
        coupling = jnp.sqrt(2 / tau_e)
        delta_omega = 2 * jnp.pi * (f_operating - f_resonance) + v_to_wr * voltage
        rhs_a = -1j * coupling * signals.p1 + 1j * delta_omega * s.a - s.a / tau
        E_o = signals.p1 - 1j * coupling * s.a
        return {"p1": 0 + 0j, "p2": s.i_out, "v_e": 0 + 0j, "i_out": signals.p2 - E_o, "a": -rhs_a}, {"a": s.a}

    c = 2.998e8
    ng = 3.8
    L_ring = float(2 * np.pi * 5e-6)
    gamma = 0.976
    alpha0 = 0.969
    f_res = c / 1310e-9
    f_op = c / 1309.7e-9
    V_bias = -2.0
    R_s = 100.0
    C_j = 100e-15
    v_to_wr = 2 * np.pi * 2e9

    models_map = {
        "ground": lambda: 0, "optical_cw": OpticalCW, "dc_voltage": DCVoltage,
        "ring_eo": RingEO, "resistor": Resistor, "capacitor": Capacitor,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "OptSrc": {"component": "optical_cw", "settings": {"power": 1.0}},
            "Ring": {"component": "ring_eo", "settings": {
                "ng": ng, "L": L_ring, "gamma": gamma, "alpha0": alpha0,
                "f_operating": float(f_op), "f_resonance": float(f_res), "v_to_wr": v_to_wr}},
            "Load": {"component": "resistor", "settings": {"R": 1.0}},
            "Vsrc": {"component": "dc_voltage", "settings": {"V_dc": V_bias}},
            "Rs": {"component": "resistor", "settings": {"R": R_s}},
            "Cj": {"component": "capacitor", "settings": {"C": C_j}},
        },
        "connections": {
            "GND,p1": ("OptSrc,p2", "Load,p2", "Vsrc,p2", "Cj,p2"),
            "OptSrc,p1": "Ring,p1", "Ring,p2": "Load,p1",
            "Vsrc,p1": "Rs,p1", "Rs,p2": ("Cj,p1", "Ring,v_e"),
        },
        "ports": {"in": "Ring,p1", "out": "Ring,p2", "ve": "Ring,v_e"},
    }
    circuit = compile_circuit(net_dict, models_map, is_complex=True)
    y_dc = circuit.dc()

    # Analytic parameters for the power modulation transfer
    tau_e = 2 * ng * L_ring / ((1 - gamma**2) * c)
    tau_l = 2 * ng * L_ring / ((1 - alpha0**2) * c)
    tau = 1 / (1 / tau_e + 1 / tau_l)
    delta_omega_dc = 2 * np.pi * (float(f_op) - float(f_res)) + v_to_wr * V_bias
    f_RC = 1 / (2 * np.pi * R_s * C_j)

    return circuit, y_dc, tau, tau_l, delta_omega_dc, R_s, C_j


def test_non_holomorphic_ring_modulator_matches_analytic():
    """holomorphic=False matches analytic EO power modulation; holomorphic=True does not."""
    import numpy as np

    from circulax.solvers.assembly import assemble_gc_complex, assemble_gc_complex_2n
    from circulax.solvers.linear import GROUND_STIFFNESS, _build_index_arrays

    circuit, y_dc, tau, tau_l, delta_omega_dc, R_s, C_j = _ring_modulator_circuit()
    groups = circuit.groups
    N = circuit.sys_size
    pmap = circuit.port_map

    freqs_hz = jnp.array(np.linspace(1e9, 80e9, 10))
    omega_m = 2 * np.pi * np.array(freqs_hz)

    # ── Analytic transfer ─────────────────────────────────────────────
    inv_tau, inv_tau_l = 1 / tau, 1 / tau_l
    H_RC = 1 / (1 + 1j * omega_m * R_s * C_j)
    H_opt = (1j * omega_m + 2 * inv_tau_l) / (
        -(omega_m**2) + 1j * 2 * inv_tau * omega_m + delta_omega_dc**2 + inv_tau**2
    )
    H_analytic = np.abs(H_RC * H_opt)
    H_analytic_norm = H_analytic / H_analytic[0]
    H_analytic_dB = 20 * np.log10(H_analytic_norm)

    # ── holomorphic=False (2N block) ──────────────────────────────────
    rows_n, cols_n, gidxs_n, _ = _build_index_arrays(groups, N, is_complex=False)
    G_blocks, C_blocks = assemble_gc_complex_2n(y_dc, groups)
    rows_j = jnp.array(rows_n)
    cols_j = jnp.array(cols_n)
    gidxs_2n = jnp.concatenate([jnp.array(gidxs_n), jnp.array(gidxs_n) + N])
    offsets = jnp.array([[0, 0], [0, N], [N, 0], [N, N]])

    drive_idx = pmap["Vsrc,i_src"]
    probe_idx = pmap["out"]
    E_dc = y_dc[probe_idx] + 1j * y_dc[probe_idx + N]
    rhs_2n = jnp.zeros(2 * N, dtype=jnp.complex128).at[drive_idx].set(1.0)

    def eo_power_mod_2n(f):
        w = 2.0 * jnp.pi * f
        Y = jnp.zeros((2 * N, 2 * N), dtype=jnp.complex128)
        for k in range(4):
            ro, co = offsets[k]
            Y = Y.at[rows_j + ro, cols_j + co].add(
                (G_blocks[k] + 1j * w * C_blocks[k]).astype(jnp.complex128)
            )
        Y = Y.at[gidxs_2n, gidxs_2n].add(GROUND_STIFFNESS)
        x = jnp.linalg.solve(Y, rhs_2n)
        dy_R, dy_I = x[:N], x[N:]
        H_plus = dy_R[probe_idx] + 1j * dy_I[probe_idx]
        H_minus_conj = dy_R[probe_idx] - 1j * dy_I[probe_idx]
        return jnp.abs(jnp.conj(E_dc) * H_plus + E_dc * H_minus_conj)

    dP_2n = jax.vmap(eo_power_mod_2n)(freqs_hz)
    dP_2n_norm = np.array(dP_2n / dP_2n[0])
    dP_2n_dB = 20 * np.log10(dP_2n_norm)

    # ── holomorphic=True (N×N Wirtinger) ──────────────────────────────
    G_w, C_w = assemble_gc_complex(y_dc, groups)
    rhs_n = jnp.zeros(N, dtype=jnp.complex128).at[drive_idx].set(1.0)
    gidxs_j = jnp.array(gidxs_n)

    def eo_power_mod_wirtinger(f):
        w = 2.0 * jnp.pi * f
        Y = jnp.zeros((N, N), dtype=jnp.complex128)
        Y = Y.at[rows_j, cols_j].add(G_w + 1j * w * C_w)
        Y = Y.at[gidxs_j, gidxs_j].add(GROUND_STIFFNESS)
        x = jnp.linalg.solve(Y, rhs_n)
        return jnp.abs(jnp.conj(E_dc) * x[probe_idx] + E_dc * jnp.conj(x[probe_idx]))

    dP_w = jax.vmap(eo_power_mod_wirtinger)(freqs_hz)
    dP_w_norm = np.array(dP_w / dP_w[0])
    dP_w_dB = 20 * np.log10(dP_w_norm)

    # ── Assertions ────────────────────────────────────────────────────
    err_2n = float(jnp.max(jnp.abs(jnp.array(dP_2n_dB) - jnp.array(H_analytic_dB))))
    err_w = float(jnp.max(jnp.abs(jnp.array(dP_w_dB) - jnp.array(H_analytic_dB))))

    assert err_2n < 0.01, f"holomorphic=False should match analytic, got {err_2n:.2f} dB error"
    assert err_w > 1.0, f"holomorphic=True should NOT match power modulation, got only {err_w:.2f} dB error"


def test_holomorphic_auto_detection():
    """circuit.sp() auto-detects holomorphic=False from component flags."""
    from circulax.circuit import _infer_holomorphic

    circuit, y_dc, *_ = _ring_modulator_circuit()

    assert not _infer_holomorphic(circuit.groups), "RingEO has holomorphic=False, should infer non-holomorphic"

    freqs = jnp.array([1e9, 10e9, 40e9])
    S_auto = circuit.sp(ports="out", freqs=freqs, z0=50.0, y_dc=y_dc)
    S_explicit = circuit.sp(ports="out", freqs=freqs, z0=50.0, y_dc=y_dc, holomorphic=False)
    assert jnp.allclose(S_auto, S_explicit, atol=1e-12), "auto-detected should match explicit holomorphic=False"


def test_holomorphic_jaxpr_validation_warns():
    """compile_circuit warns when holomorphic=True component uses non-holomorphic ops in a complex circuit."""
    import warnings

    from circulax.circuit import compile_circuit
    from circulax.components.base_component import component
    from circulax.components.photonic import OpticalWaveguide

    @component(ports=("p1", "p2"), holomorphic=True)
    def _BadPhotodetector(signals, s, R=1.0):
        power = jnp.real(signals.p1 * jnp.conj(signals.p1))
        i_photo = power * R
        return {"p1": 0 + 0j, "p2": i_photo}, {}

    models_map = {
        "ground": lambda: 0,
        "waveguide": OpticalWaveguide,
        "bad_pd": _BadPhotodetector,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "WG1": {"component": "waveguide", "settings": {"neff": 2.5, "ng": 3.5, "length": 1e-4, "loss": 0.0}},
            "PD": {"component": "bad_pd", "settings": {"R": 0.8}},
        },
        "connections": {"GND,p1": ("WG1,p2", "PD,p2"), "WG1,p1": "PD,p1"},
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        compile_circuit(net_dict, models_map, is_complex=True)

    matches = [x for x in w if "non-holomorphic" in str(x.message) and "bad_pd" in str(x.message).lower()]
    assert len(matches) == 1, f"Expected 1 holomorphic warning for bad_pd, got {len(matches)}: {[str(x.message) for x in matches]}"
    assert "real" in str(matches[0].message) or "conj" in str(matches[0].message)


def test_holomorphic_jaxpr_validation_no_false_positive():
    """Electronic-only circuits do not trigger false holomorphic warnings."""
    import warnings

    from circulax.circuit import compile_circuit
    from circulax.components.electronic import Capacitor, Resistor

    models_map = {
        "ground": lambda: 0,
        "resistor": Resistor,
        "capacitor": Capacitor,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "R1": {"component": "resistor", "settings": {"R": 50.0}},
            "C1": {"component": "capacitor", "settings": {"C": 1e-12}},
        },
        "connections": {"R1,p1": "C1,p1", "R1,p2": "GND,p1", "C1,p2": "GND,p1"},
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        compile_circuit(net_dict, models_map)

    matches = [x for x in w if "non-holomorphic" in str(x.message)]
    assert len(matches) == 0, f"Unexpected holomorphic warning for real circuit: {matches}"
