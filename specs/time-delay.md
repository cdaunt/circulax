# Frequency-Domain Delay Line for AC/HB Analysis

## Overview

| Property | Value |
|----------|-------|
| Description | Pure group-delay element for AC sweep and harmonic balance |
| SPICE Equivalent | Ideal lossless transmission line (frequency domain only) |
| Approach | `exp(-j 2π f τ)` on S-matrix off-diagonals, converted to Y via `s_to_y` |
| Analysis types | AC sweep, harmonic balance |
| Transient | Not supported — fdomain components raise at `setup_transient()` time |
| Status | Complete (branch `feat-time-delay`) |

## Problem Statement

Photonic circuits require modeling signal propagation delay through waveguides.
For AC and harmonic-balance analysis, delay is a frequency-domain concept: a
delay of `τ` seconds multiplies each S-parameter off-diagonal by `exp(-j 2π f τ)`.

The key insight is that `f` in an `@fdomain_component` is the **modulation/signal
frequency** (e.g. 1 GHz for an AC sweep), not the optical carrier frequency
(~229 THz at 1310 nm). Carrier-phase effects (`neff`, `wavelength_nm`) belong in
the wavelength-domain `OpticalWaveguide` and are intentionally excluded from this
component.

### Design decisions

| Decision | Rationale |
|----------|-----------|
| `@fdomain_component` decorator | Provides `Y(f)` admittance matrices stamped into `Y_total` at each frequency point — no netlist expansion |
| No carrier phase | `f` = modulation frequency, not optical carrier; `neff`/`wavelength_nm` are wavelength-domain concepts |
| `s_to_y` conversion | Reuses existing S→Y infrastructure; consistent with `OpticalWaveguide` |
| No transient support | Fdomain components are AC/HB-only by design; `setup_transient()` raises `RuntimeError` if any group has `is_fdomain=True` |

---

## Specification

### Component: `delay_line_fdomain`

```python
@fdomain_component(ports=("p1", "p2"))
def delay_line_fdomain(
    f: float,
    length_um: float = 100.0,
    loss_dB_cm: float = 1.0,
    n_group: float = 4.0,
) -> jnp.ndarray:
    c_um_per_s = 2.99792458e14
    loss_val = loss_dB_cm * (length_um / 10000.0)
    T_mag = 10.0 ** (-loss_val / 20.0)
    tau = (length_um * n_group) / c_um_per_s
    T = T_mag * jnp.exp(-1j * 2.0 * jnp.pi * f * tau)
    S = jnp.array([[0.0, T], [T, 0.0]], dtype=jnp.complex128)
    return s_to_y(S)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `f` | Modulation frequency in Hz (supplied by the AC/HB solver) | — |
| `length_um` | Waveguide length in micrometres | `100.0` |
| `loss_dB_cm` | Propagation loss in dB/cm | `1.0` |
| `n_group` | Group refractive index; sets delay via `τ = length_um · n_group / c` | `4.0` |

### S-parameter round-trip

For an fdomain Y-matrix component, the AC sweep solver stamps `Y_fdomain(f)` directly
into the nodal admittance matrix. The S→Y→stamp→solve→S round-trip yields:

```
S21 = T_mag · exp(-j · 2π · f · τ)
```

Note: no factor-of-2 on `T_mag`. The `s_to_y` Y-matrix stamp gives `S21 = T` (identity
round-trip), unlike VCVS-style components where the constraint structure gives `S21 = 2T`.

### Usage

```python
from circulax import compile_netlist
from circulax.components.photonic import delay_line_fdomain
from circulax.solvers import analyze_circuit, setup_ac_sweep

models_map = {"delay": delay_line_fdomain, "ground": lambda: 0}
net_dict = {
    "instances": {
        "GND": {"component": "ground"},
        "WG1": {"component": "delay", "settings": {"length_um": 500.0, "n_group": 4.0}},
    },
    "connections": {},
    "ports": {"in": "WG1,p1", "out": "WG1,p2"},
}

groups, num_vars, pmap = compile_netlist(net_dict, models_map)
solver = analyze_circuit(groups, num_vars, backend="dense", is_complex=True)
y_dc = solver.solve_dc(groups, jnp.zeros(num_vars * 2, dtype=jnp.float64))

port_nodes = [pmap["WG1,p1"], pmap["WG1,p2"]]
run_ac = setup_ac_sweep(groups, num_vars, port_nodes, z0=1.0, is_complex=True)
S = run_ac(y_dc, jnp.linspace(1e9, 100e9, 50))
```

---

## Implementation Files

| File | Change |
|------|--------|
| `circulax/components/photonic.py` | `delay_line_fdomain` component; removed `OpticalDelayLine` |
| `circulax/components/__init__.py` | Exports `delay_line_fdomain` |
| `tests/test_delay.py` | `test_delay_line_fdomain_ac_sweep` — validates S21 magnitude and phase against analytic formula |

### Removed files / code (old transient delay machinery)

The following infrastructure was removed as part of the pivot from transient
history-buffer delays to the cleaner fdomain approach:

| File | Removed |
|------|---------|
| `circulax/components/base_component.py` | `hist` argument detection, `@Component.delay` decorator, `_has_delay` flag, `tau_of`, `_tau_cell`, `_register_delay`, `_resolve_tau`, `has_hist_arg` |
| `circulax/compiler.py` | `has_delay` and `tau_func` fields on `ComponentGroup` |
| `circulax/solvers/assembly.py` | `_interp_delayed`, `_group_tau`, `min_active_tau`, `assemble_delay_gc_real/complex/complex_2n`, delayed physics/Jacobian wrappers, `hist_t`/`hist_y` params on all `assemble_*` functions |
| `circulax/solvers/circuit_diffeq.py` | `CircuitState.hist_t`/`hist_y` fields, buffer allocation, seed/write/clamp logic, `record_history`/`min_tau` params |
| `circulax/solvers/transient.py` | `has_delay` detection, `min_tau` computation, `record_history` kwarg, delay warning callback, `hist_t`/`hist_y` params on helper functions |
| `circulax/solvers/ac_sweep.py` | `assemble_delay_gc_*` imports, `has_delay`/`delay_data` variables, per-frequency delay-phase contribution loops |
| `tests/test_delay.py` | 8 old transient delay tests (`test_delay_line_matches_analytic_shift`, gradient tests, adaptive tests, `OpticalDelayLine` AC sweep test, etc.) |

---

## Test Matrix

| Test | Verification |
|------|-------------|
| `test_delay_line_fdomain_ac_sweep` | S21 magnitude matches `T_mag` within 1e-6; S21 phase matches `-2πfτ` within 1e-6 rad |

---

## Known Limitations

### 1. No transient delay support — **By design**

Fdomain components cannot be used in transient simulation. `setup_transient()` raises
`RuntimeError` if any group has `is_fdomain=True`. Transient delay would require
re-implementing the history-buffer machinery that was removed, or upstream diffrax DDE
support.

### 2. Near-lossless conditioning — **Inherited from `s_to_y`**

When `T_mag ≈ 1`, the S-matrix eigenvalue approaches -1 and `(I + S)` becomes
near-singular, causing `s_to_y` to produce large Y-matrix entries. This is the same
limitation as `OpticalWaveguide` and other S-matrix-based components. Workaround: use
a small nonzero `loss_dB_cm`.

### 3. No state-dependent delays — **By design**

Delay `τ` is computed from component parameters (`length_um`, `n_group`), not from the
circuit state. State-dependent delays would require DDE machinery.

---

## Verification

All tests pass. No regressions in the full suite.

```bash
pytest tests/test_delay.py -v   # 1 passed (3.7s)
pytest tests/ -v                # 269 passed, 16 skipped, 0 failures (115s)
```
