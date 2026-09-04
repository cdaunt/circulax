"""Tests for rational component factory: SSModel → circulax @component."""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.compiler import compile_netlist
from circulax.components.rational import rational_component, rational_delay_component, rational_fdomain_component
from circulax.solvers import analyze_circuit, setup_ac_sweep, setup_transient

jax.config.update("jax_enable_x64", True)


_TEST_POLES = np.array([-1e9 + 1j * 6e10, -1e9 - 1j * 6e10, -5e9], dtype=np.complex128)
_TEST_R1 = np.array([[5e7 + 2e7j, -4e7 - 1e7j], [-4e7 - 1e7j, 5e7 + 2e7j]], dtype=np.complex128)
_TEST_R2 = _TEST_R1.conj()
_TEST_R3 = np.array([[1e8, -8e7], [-8e7, 1e8]], dtype=np.complex128)
_TEST_RESIDUES = [_TEST_R1, _TEST_R2, _TEST_R3]
_TEST_D = np.array([[0.02, -0.019], [-0.019, 0.02]], dtype=np.float64)


def _eval_pole_residue(s_pts, poles=_TEST_POLES, residues=_TEST_RESIDUES, D=_TEST_D):
    """Evaluate Y(s) = D + sum_k R_k/(s - p_k) from raw pole-residue data.

    Independent of the tile/kron/reshape SS packing — serves as the ground-truth
    oracle that catches packing bugs invisible to _eval_ss.
    """
    result = np.zeros((len(s_pts), D.shape[0], D.shape[1]), dtype=np.complex128)
    for i, s in enumerate(np.asarray(s_pts)):
        Y = D.astype(np.complex128).copy()
        for p, R in zip(poles, residues):
            Y = Y + R / (s - p)
        result[i] = Y
    return result


def _make_test_ss():
    """Build a simple 2-port SSModel-like object with known poles."""
    poles = np.array([-1e9 + 1j * 6e10, -1e9 - 1j * 6e10, -5e9], dtype=np.complex128)
    Nc = 2
    N = len(poles)

    R1 = np.array([[5e7 + 2e7j, -4e7 - 1e7j], [-4e7 - 1e7j, 5e7 + 2e7j]], dtype=np.complex128)
    R2 = R1.conj()
    R3 = np.array([[1e8, -8e7], [-8e7, 1e8]], dtype=np.complex128)
    residues = np.stack([R1, R2, R3], axis=-1)

    D = np.array([[0.02, -0.019], [-0.019, 0.02]], dtype=np.float64)
    E = np.zeros((Nc, Nc), dtype=np.float64)

    A_diag = np.tile(poles, Nc)
    B = np.kron(np.eye(Nc, dtype=np.complex128), np.ones((N, 1), dtype=np.complex128))
    C = residues.reshape(Nc, Nc * N)

    return SimpleNamespace(
        A=jnp.array(A_diag),
        B=jnp.array(B),
        C=jnp.array(C),
        D=jnp.array(D, dtype=jnp.complex128),
        E=jnp.array(E, dtype=jnp.complex128),
    )


class TestRationalComponent:
    """Test the rational_component factory."""

    @pytest.fixture
    def ss(self):
        return _make_test_ss()

    def test_creates_component_class(self, ss):
        Comp = rational_component(ss, name="TestComp")
        assert Comp.ports == ("p1", "p2")
        assert len(Comp.states) == 6

    def test_instantiate_and_call(self, ss):
        Comp = rational_component(ss)
        inst = Comp()
        n_vars = len(Comp.ports) + len(Comp.states)
        y = jnp.zeros(n_vars, dtype=jnp.complex128)
        f, q = inst(y)
        assert set(f.keys()) == set(Comp.ports + Comp.states)
        assert set(q.keys()) == set(Comp.ports + Comp.states)

    def test_solver_call(self, ss):
        Comp = rational_component(ss)
        n_vars = len(Comp.ports) + len(Comp.states)
        y = jnp.zeros(n_vars, dtype=jnp.complex128)
        f_vec, q_vec = Comp.solver_call(0.0, y, Comp())
        assert f_vec.shape == (n_vars,)
        assert q_vec.shape == (n_vars,)
        assert jnp.all(jnp.isfinite(f_vec))
        assert jnp.all(jnp.isfinite(q_vec))

    def test_dc_response(self, ss):
        """At DC-consistent (v, x) state, i_port = H(0) @ v from pole-residue sum."""
        Comp = rational_component(ss)
        n_p = len(Comp.ports)

        H0 = _eval_pole_residue(np.array([0.0]), _TEST_POLES, _TEST_RESIDUES, _TEST_D)[0]
        v = jnp.array([1.0 + 0j, 0.5 + 0j])
        A = np.asarray(ss.A)
        B = np.asarray(ss.B)
        x = jnp.array(-(B @ np.asarray(v)) / A)
        y = jnp.concatenate([v, x])

        f_vec, _ = Comp.solver_call(0.0, y, Comp())
        np.testing.assert_allclose(np.asarray(f_vec[n_p:]), 0.0, atol=1e-6,
                                   err_msg="State equations not zero at consistent point")
        i_expected = H0 @ np.asarray(v)
        np.testing.assert_allclose(np.asarray(f_vec[:n_p]), i_expected, rtol=1e-8,
                                   err_msg="Port current != H(0) @ v")

    def test_origin_pole_raises(self):
        bad_ss = SimpleNamespace(
            A=jnp.array([0.0 + 0j, -1e9 + 0j]),
            B=jnp.ones((2, 1), dtype=jnp.complex128),
            C=jnp.ones((1, 2), dtype=jnp.complex128),
            D=jnp.zeros((1, 1), dtype=jnp.complex128),
            E=jnp.zeros((1, 1), dtype=jnp.complex128),
        )
        with pytest.raises(ValueError, match="pole at or near the origin"):
            rational_component(bad_ss)


class TestRationalFdomainComponent:
    """Test the frequency-domain oracle."""

    @pytest.fixture
    def ss(self):
        return _make_test_ss()

    def test_creates_fdomain_class(self, ss):
        Comp = rational_fdomain_component(ss)
        assert Comp._is_fdomain
        assert Comp.ports == ("p1", "p2")

    def test_solver_call_matches_pole_residue(self, ss):
        """Fdomain component matches the independent pole-residue oracle."""
        Comp = rational_fdomain_component(ss)
        freqs = np.array([1e9, 5e9, 10e9, 20e9])
        s_pts = 1j * 2 * np.pi * freqs
        H_ref = _eval_pole_residue(s_pts)

        for i, f in enumerate(freqs):
            Y = Comp.solver_call(float(f), Comp())
            np.testing.assert_allclose(np.asarray(Y), H_ref[i], atol=1e-10)


class TestFourWayAgreement:
    """Verify rational_component, rational_fdomain_component, eval_ss, and
    the original Y data all agree in AC sweep."""

    @pytest.fixture
    def setup(self):
        from circulax.components.electronic import Resistor
        ss = _make_test_ss()
        z0 = 50.0

        Comp_td = rational_component(ss, name="RatTD", z0=z0)
        Comp_fd = rational_fdomain_component(ss, name="RatFD")

        net = {
            "instances": {
                "GND": {"component": "ground"},
                "DUT": {"component": "dut", "settings": {}},
                "R1": {"component": "resistor", "settings": {"R": z0}},
                "R2": {"component": "resistor", "settings": {"R": z0}},
            },
            "connections": {
                "R1,p2": "DUT,p1",
                "R2,p2": "DUT,p2",
                "R1,p1": "GND,p1",
                "R2,p1": "GND,p1",
            },
            "ports": {"port1": "DUT,p1", "port2": "DUT,p2"},
        }

        models_td = {"dut": Comp_td, "resistor": Resistor, "ground": lambda: 0}
        groups_td, num_vars_td, pmap_td = compile_netlist(net, models_td)
        solver_td = analyze_circuit(groups_td, num_vars_td, backend="dense", is_complex=True)
        y_dc_td = solver_td.solve_dc(groups_td, jnp.zeros(num_vars_td * 2, dtype=jnp.float64))
        port_nodes_td = [pmap_td["DUT,p1"], pmap_td["DUT,p2"]]
        run_ac_td = setup_ac_sweep(groups_td, num_vars_td, port_nodes_td, z0=z0, is_complex=True)

        models_fd = {"dut": Comp_fd, "resistor": Resistor, "ground": lambda: 0}
        groups_fd, num_vars_fd, pmap_fd = compile_netlist(net, models_fd)
        solver_fd = analyze_circuit(groups_fd, num_vars_fd, backend="dense", is_complex=True)
        y_dc_fd = solver_fd.solve_dc(groups_fd, jnp.zeros(num_vars_fd * 2, dtype=jnp.float64))
        port_nodes_fd = [pmap_fd["DUT,p1"], pmap_fd["DUT,p2"]]
        run_ac_fd = setup_ac_sweep(groups_fd, num_vars_fd, port_nodes_fd, z0=z0, is_complex=True)

        return ss, run_ac_td, y_dc_td, run_ac_fd, y_dc_fd

    def test_ac_sweep_agreement(self, setup):
        """Time-domain and fdomain components produce identical S-parameters.

        Both are embedded in the same 2-port testbench (R1=R2=z0 shunt to GND),
        so the S-params include the port termination. The key check is that
        rational_component (DAE stamp) agrees with rational_fdomain_component
        (direct H(j2*pi*f) evaluation) — any difference means the DAE mapping
        is wrong.
        """
        ss, run_ac_td, y_dc_td, run_ac_fd, y_dc_fd = setup
        freqs = jnp.logspace(7, 10.3, 30)

        S_td = run_ac_td(y_dc_td, freqs)
        S_fd = run_ac_fd(y_dc_fd, freqs)

        np.testing.assert_allclose(np.asarray(S_td), np.asarray(S_fd), atol=1e-8,
                                   err_msg="Time-domain vs fdomain component disagree")

    def test_ac_sweep_matches_pole_residue_oracle(self, setup):
        """S-params from AC sweep recover the original Y data after de-embedding port termination."""
        from circulax.s_transforms import s_to_y

        ss, run_ac_td, y_dc_td, _, _ = setup
        z0 = 50.0
        freqs = jnp.logspace(8, 10.3, 20)
        s_pts = 1j * 2 * np.pi * np.asarray(freqs)

        S_td = np.asarray(run_ac_td(y_dc_td, freqs))
        Y_ref = _eval_pole_residue(s_pts)

        for k in range(len(freqs)):
            Y_circuit = np.asarray(s_to_y(jnp.array(S_td[k]), z0=z0))
            Y_dut = Y_circuit - np.eye(2) / z0
            np.testing.assert_allclose(Y_dut, Y_ref[k], rtol=1e-6,
                                       err_msg=f"Y mismatch at f={float(freqs[k]):.2e}")

    def test_ac_sweep_finite(self, setup):
        ss, run_ac_td, y_dc_td, run_ac_fd, y_dc_fd = setup
        freqs = jnp.logspace(7, 10.3, 20)
        S = run_ac_td(y_dc_td, freqs)
        assert jnp.all(jnp.isfinite(S))
        assert S.shape == (len(freqs), 2, 2)


class TestTransientStability:
    """Verify the rational component settles to correct DC steady state."""

    def test_step_response_settles(self):
        import diffrax

        from circulax.components.electronic import Resistor, VoltageSource

        ss = _make_test_ss()
        z0 = 50.0
        Comp = rational_component(ss, name="RatTransient", z0=z0)

        net = {
            "instances": {
                "GND": {"component": "ground"},
                "V1": {"component": "vsrc", "settings": {"V": 1.0}},
                "R1": {"component": "resistor", "settings": {"R": z0}},
                "DUT": {"component": "dut", "settings": {}},
            },
            "connections": {
                "V1,p1": "R1,p1",
                "R1,p2": "DUT,p1",
                "V1,p2": "GND,p1",
                "DUT,p2": "GND,p1",
            },
        }
        models = {
            "ground": lambda: 0,
            "vsrc": VoltageSource,
            "resistor": Resistor,
            "dut": Comp,
        }
        groups, num_vars, pmap = compile_netlist(net, models)
        solver = analyze_circuit(groups, num_vars, backend="dense", is_complex=True)
        y_dc = solver.solve_dc(groups, jnp.zeros(num_vars * 2, dtype=jnp.float64))

        run_transient = setup_transient(groups, solver)
        sol = run_transient(
            t0=0.0,
            t1=10e-9,
            dt0=1e-12,
            y0=y_dc,
            saveat=diffrax.SaveAt(ts=jnp.array([9e-9])),
            max_steps=50000,
        )

        assert jnp.all(jnp.isfinite(sol.ys)), "Transient solution diverged"

        H0 = _eval_pole_residue(np.array([0.0]))[0]
        Y11_dc = H0[0, 0].real
        v1_expected = 1.0 / (1.0 + z0 * Y11_dc)

        dut_p1_node = pmap["DUT,p1"]
        v1_sim = float(sol.ys[0, dut_p1_node])
        np.testing.assert_allclose(v1_sim, v1_expected, rtol=0.01,
                                   err_msg="Transient did not hold at DC divider value")


def _measure_s_matrix(Comp, freqs, z0=50.0):
    """Evaluate S(f) from a component's solver_call Y(f) output."""
    inst = Comp()
    Nc = len(Comp.ports)
    eye = np.eye(Nc, dtype=np.complex128)
    S = np.zeros((len(freqs), Nc, Nc), dtype=np.complex128)
    for i, f in enumerate(freqs):
        Y = np.asarray(Comp.solver_call(float(f), inst))
        S[i] = (eye - z0 * Y) @ np.linalg.inv(eye + z0 * Y)
    return S


def _group_delay_from_s(S_element, freqs):
    """Compute group delay from a 1-D array of S-parameter values vs frequency."""
    phase = np.unwrap(np.angle(S_element))
    omega = 2.0 * np.pi * freqs
    return -np.gradient(phase, omega)


class TestRationalDelayComponent:
    """Test the delay+rational composite fdomain component."""

    def test_creates_fdomain_class(self):
        ss = _make_test_ss()
        tau = np.array([1e-9, 1e-9])
        Comp = rational_delay_component(ss, tau)
        assert Comp._is_fdomain
        assert Comp.ports == ("p1", "p2")

    def test_s21_group_delay_matches_tau(self):
        """S21 group delay of composite exceeds rational-only by exactly tau."""
        ss = _make_test_ss()
        z0 = 50.0
        tau_val = 1e-9
        tau = np.array([tau_val, tau_val])

        Comp_delay = rational_delay_component(ss, tau, name="RatDelay", z0=z0)
        Comp_nodelay = rational_delay_component(ss, np.array([0.0, 0.0]), name="RatNoDelay", z0=z0)

        freqs = np.linspace(1e9, 15e9, 300)

        S_delay = _measure_s_matrix(Comp_delay, freqs, z0)
        S_nodelay = _measure_s_matrix(Comp_nodelay, freqs, z0)

        gd_delay = _group_delay_from_s(S_delay[:, 0, 1], freqs)
        gd_nodelay = _group_delay_from_s(S_nodelay[:, 0, 1], freqs)
        excess = np.median(gd_delay - gd_nodelay)

        np.testing.assert_allclose(excess, tau_val, rtol=0.03,
                                   err_msg=f"S21 excess group delay {excess:.3e} != tau {tau_val:.3e}")

    def test_zero_delay_matches_rational_only(self):
        """With tau=0, the composite should match rational_fdomain_component."""
        ss = _make_test_ss()
        tau = np.array([0.0, 0.0])

        Comp_delay = rational_delay_component(ss, tau, name="RatDelayZero")
        Comp_fd = rational_fdomain_component(ss, name="RatFD0")

        freqs = [1e9, 5e9, 20e9]
        for f in freqs:
            Y_delay = Comp_delay.solver_call(f, Comp_delay())
            Y_fd = Comp_fd.solver_call(f, Comp_fd())
            np.testing.assert_allclose(np.asarray(Y_delay), np.asarray(Y_fd), atol=1e-10)

    def test_asymmetric_delay(self):
        """Asymmetric delays: S11 excess GD = tau1, S22 excess GD = tau2."""
        ss = _make_test_ss()
        z0 = 50.0
        tau1, tau2 = 1e-9, 2e-9
        tau = np.array([tau1, tau2])

        Comp_delay = rational_delay_component(ss, tau, name="AsymDelay", z0=z0)
        Comp_nodelay = rational_delay_component(ss, np.array([0.0, 0.0]), name="AsymNoDelay", z0=z0)

        freqs = np.linspace(1e9, 15e9, 300)
        S_delay = _measure_s_matrix(Comp_delay, freqs, z0)
        S_nodelay = _measure_s_matrix(Comp_nodelay, freqs, z0)

        gd_s11 = _group_delay_from_s(S_delay[:, 0, 0], freqs)
        gd_s11_ref = _group_delay_from_s(S_nodelay[:, 0, 0], freqs)
        excess_s11 = np.median(gd_s11 - gd_s11_ref)
        np.testing.assert_allclose(excess_s11, tau1, rtol=0.05,
                                   err_msg=f"S11 excess GD {excess_s11:.3e} != tau1 {tau1:.3e}")

        gd_s22 = _group_delay_from_s(S_delay[:, 1, 1], freqs)
        gd_s22_ref = _group_delay_from_s(S_nodelay[:, 1, 1], freqs)
        excess_s22 = np.median(gd_s22 - gd_s22_ref)
        np.testing.assert_allclose(excess_s22, tau2, rtol=0.05,
                                   err_msg=f"S22 excess GD {excess_s22:.3e} != tau2 {tau2:.3e}")

        gd_s21 = _group_delay_from_s(S_delay[:, 0, 1], freqs)
        gd_s21_ref = _group_delay_from_s(S_nodelay[:, 0, 1], freqs)
        excess_s21 = np.median(gd_s21 - gd_s21_ref)
        np.testing.assert_allclose(excess_s21, (tau1 + tau2) / 2, rtol=0.05,
                                   err_msg=f"S21 excess GD {excess_s21:.3e} != (tau1+tau2)/2")
