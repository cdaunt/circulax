# Baseband Envelope Simulation — Carrier-Relative Spectral Effects

## Goal

Enable circulax to simulate how photonic component transfer functions filter
modulation sidebands differently depending on where they fall on the
wavelength-domain response — capturing sideband asymmetry, dispersion penalty,
and intermodulation that the current two-axis model misses.

## Context

### The gap in circulax today

Circulax treats the wavelength axis and modulation frequency axis as independent:

- **Wavelength axis**: S-parameters evaluated at a single λ (or vmapped over λ).
  Components like `OpticalWaveguide` have `wavelength_nm` as a parameter.
- **Modulation axis**: AC sweep / transient runs over modulation frequencies
  (0–40 GHz). The `@fdomain_component` returns Y(f_mod) at each modulation
  frequency.

These two axes do not interact. When you vmap an AC sweep over wavelength, you get
independent small-signal responses — each wavelength is a separate universe. The
modulation sidebands at f_c ± f_mod all see the same S-parameter value evaluated at
a single λ.

### What physics this misses

In a real photonic device, a modulated optical signal creates frequency components at
f_c + f_mod (upper sideband) and f_c − f_mod (lower sideband). Each sideband sees a
different point on the device's spectral transfer function. This coupling between the
wavelength-domain shape and the modulation signal is the mechanism behind:

| Effect | Physical origin |
|--------|----------------|
| Single-sideband filtering | Ring resonance slope attenuates one sideband more than the other |
| Chromatic dispersion penalty | Waveguide dispersion applies different phase shifts to upper/lower sidebands |
| RF link gain from resonance slope | Mach-Zehnder or ring modulator biased on the steepest part of the transfer function |
| Intermodulation distortion | Nonlinear (curved) transfer function around carrier generates mixing products |
| Wavelength-dependent group delay | Modulation envelope distorted by dispersion slope (GVD) |

These effects are critical for RF-photonic link design, microwave photonic filters,
and high-bandwidth modulator optimization — all areas where circulax's
differentiability could enable gradient-based design.

### The baseband envelope approach

The established solution (Ye et al. 2019, Ullrick et al. 2023) uses Complex Vector
Fitting (CVF) to fit the wavelength-domain S-parameters as a baseband rational model:

```
S_b(f_mod) = S(f_c + f_mod)
```

The key identity: evaluating the baseband model at modulation frequency f_mod is
equivalent to evaluating the passband model at f_c + f_mod. So an AC sweep in
baseband naturally captures how the spectral shape around the carrier filters the
modulation signal.

The resulting state-space ODE operates entirely in baseband with complex-valued
signals (the "complex envelope"):

```
dx_b/dt = A · x_b + B · a_b
b_b     = C · x_b + D · a_b
```

where x_b(t), a_b(t), b_b(t) are all complex. The carrier f_c does not appear in
the ODE — it only enters when recovering the physical signal:
`y(t) = Re{ y_b(t) · exp(j·2π·f_c·t) }`.

### Relationship to existing work

| Technique | What it removes | Status in circulax |
|-----------|----------------|--------------------|
| Delay de-embedding | Propagation delay τ → fewer poles | Done ([vector-fitting.md](vector-fitting.md)) |
| Baseband envelope | Carrier frequency f_c → sidebands see spectral shape | This spec (planned) |

The two techniques are orthogonal and composable. For a long waveguide feeding a ring
resonator, you would want both: delay de-embedding for the waveguide propagation, and
baseband envelope for the ring's spectral filtering of the modulation signal.

---

## Design decisions to make

### 1. Complex envelope as the signal representation

All optical signals become complex-valued baseband envelopes. The solver integrates
complex ODEs where:
- Real parts of A eigenvalues → decay rates (resonance linewidths, loss)
- Imaginary parts of A eigenvalues → detuning from carrier (GHz scale, not THz)

This means the ODE timestep is set by the modulation bandwidth, not the optical
period — a ~10^4× efficiency gain over passband simulation.

**Implication for circulax**: the `is_complex=True` path already exists and handles
complex state vectors. The infrastructure is in place; the question is how to
populate the state-space matrices from wavelength-domain S-parameter data.

### 2. CVF vs AAA for baseband fitting

Standard VF enforces conjugate pole pairs so that h(t) is real-valued. Baseband
impulse responses are inherently complex (the spectrum is not symmetric about f = 0),
so the conjugate constraint must be dropped.

Options:
- **Complex Vector Fitting (CVF)**: Modified VF without conjugate pairing. Existing
  toolbox (Spina et al. 2021, UGent). Half the poles of standard VF.
- **AAA**: Already has no conjugate constraint — works on complex data natively. The
  vfitax AAA implementation may work as-is for baseband fitting without modification.
  Needs validation.

If AAA works for baseband fitting, the entire pipeline (delay de-embed → baseband
shift → AAA fit → DAE stamp) would use a single fitting engine.

### 3. Carrier shift as a parameter

Ullrick et al. showed that shifting the carrier by Δf only modifies the A matrix:

```
A → A − j·2π·Δf·I
```

Stability and passivity are preserved (only imaginary parts of eigenvalues change).
This means a single fitted model can simulate any WDM channel within the fitted
bandwidth — the carrier wavelength becomes a differentiable parameter of the
component without refitting.

**Implication**: `rational_component` could accept an optional `f_carrier_shift`
parameter that shifts the A matrix diagonal. This would be a JAX-traceable operation,
enabling gradient-based optimization over carrier wavelength.

### 4. Interaction with delay de-embedding

For a device with both propagation delay and spectral features (e.g., waveguide +
ring), the pipeline would be:

```
S(f_optical) at passband
  → extract delay τ from phase slope
  → de-embed delay: S'(f_optical) = P·S·P
  → shift to baseband: S_b(f_mod) = S'(f_c + f_mod)
  → fit S_b with AAA → SSModel (few poles, baseband)
  → realize as rational_component (complex envelope ODE)
  → re-embed delay as separate delay element (fdomain or transient)
```

The de-embedding happens in the passband coordinate before the baseband shift,
because the delay is a passband phenomenon (propagation through the physical
waveguide).

---

## Acceptance criteria

- [ ] Baseband rational model of a ring resonator reproduces sideband asymmetry:
      S_b(+f_mod) ≠ S_b(−f_mod) for carrier detuned from resonance center
- [ ] AC sweep of baseband component matches wavelength-domain S-parameters
      evaluated at f_c + f_mod (the defining identity)
- [ ] Carrier shift Δf produces correct wavelength-shifted response without refitting
- [ ] Transient simulation of modulated signal through ring resonator shows envelope
      distortion consistent with resonance slope filtering
- [ ] Carrier shift parameter is differentiable (jax.grad produces finite gradient)
- [ ] Combined delay + baseband pipeline: long waveguide + ring resonator uses ≤10
      poles total (delay de-embedded, spectral shape fit in baseband)
- [ ] Passivity of baseband model preserved under carrier shift

---

## Components

| Component | Description | Depends on |
|-----------|-------------|------------|
| Baseband S-param shift | `shift_to_baseband(S, freqs, f_carrier) → S_b, freqs_b` — frequency translation of S-parameter data | — |
| CVF / AAA baseband fit | Fit S_b with rational model (no conjugate constraint). Validate whether vfitax AAA works as-is or needs modification | Baseband shift |
| `rational_baseband_component` | Component factory that creates complex-envelope ODE from baseband SSModel. Accepts `f_carrier_shift` as differentiable parameter | CVF/AAA fit |
| Carrier shift mechanics | `A → A − j·2π·Δf·I` in the component, JAX-traceable | `rational_baseband_component` |
| Delay + baseband pipeline | Compose delay de-embedding (passband) with baseband fit. New `fit_with_delay_baseband` entry point in vfitax | Baseband fit, delay de-embedding |
| Complex envelope testbench | Source/probe components that inject/measure complex envelope signals. Conversion to/from passband for visualization | `rational_baseband_component` |

---

## Delegation map

| Component | File scope | Constraints |
|-----------|------------|-------------|
| Baseband S-param shift | `vfitax/sparam.py` | Pure numpy, no JAX dependency |
| CVF / AAA baseband fit | `vfitax/sparam.py` or `vfitax/cvf.py` | Validate AAA first; only implement CVF if AAA fails on asymmetric spectra |
| `rational_baseband_component` | `circulax/components/rational.py` | Extend existing factory pattern; reuse `is_complex=True` path |
| Carrier shift | `circulax/components/rational.py` | Must be eqx.field for differentiability |
| Delay + baseband pipeline | `vfitax/sparam.py` | Compose with existing `fit_with_delay` |
| Complex envelope testbench | `circulax/components/sources.py` or new file | Must work with existing `setup_ac_sweep` and `setup_transient` |

---

## Implementation notes

### AAA may already work for baseband

The vfitax AAA implementation (`aaa_scalar`) operates on complex-valued functions
with no conjugate-pair enforcement. The post-processing in `_collect_poles` does
enforce conjugate pairing — this would need to be made optional for baseband fitting.
Everything else (Loewner matrix, SVD, barycentric evaluation, residue identification)
should work unchanged.

### The `is_complex=True` path handles complex state vectors

Circulax already has the 2N real-block assembly path for complex-valued circuits.
Baseband components would use this path naturally. The real-pair optimization
(`_ss_to_real_pairs`) noted as future work in `vector-fitting.md` is irrelevant for
baseband models since the poles are genuinely non-conjugate.

### Passivity under carrier shift

The Hamiltonian matrix transforms as `M → M − j·2π·Δf·I`, which translates
eigenvalues along the imaginary axis. Passivity violations (purely imaginary
Hamiltonian eigenvalues) are mapped bijectively — they translate in frequency but
neither appear nor disappear. A model that is passive at f_c is passive at
f_c + Δf. This means passivity enforcement needs to run only once, at the reference
carrier.

### Sign convention

Optical and RF communities use opposite Fourier conventions (e^{+jωt} vs e^{−jωt}).
Circulax uses the RF convention (s = j·2π·f). The baseband shift direction and
carrier shift sign must be consistent with this. Verify against the convention in
`s_transforms.py` before implementing.

### What this does NOT cover

- **Nonlinear modulation**: electro-optic modulation (Pockels effect, carrier
  injection) is a nonlinear process that creates sidebands. This spec covers the
  linear filtering of those sidebands by passive photonic components, not the
  sideband generation itself. Nonlinear modulators would need a separate `@source`
  or `@component` that generates complex envelope signals.
- **Multi-carrier simulation**: WDM with multiple carriers simultaneously present
  requires either separate baseband envelopes per channel (no inter-channel effects)
  or a wideband passband simulation. This spec covers single-carrier or
  carrier-shifted simulation.
- **Kerr / four-wave mixing**: nonlinear inter-channel effects are out of scope.

---

## Key references

1. Y. Ye, D. Spina, D. Deschrijver, W. Bogaerts, T. Dhaene, "Time-domain compact
   macromodeling of linear photonic circuits via complex vector fitting," Photonics
   Research 7(7):771, 2019.
2. T. Ullrick, D. Spina, W. Bogaerts, T. Dhaene, "Wideband parametric baseband
   macromodeling of linear and passive photonic circuits via complex vector fitting,"
   Scientific Reports 13:15407, 2023.
3. D. Spina, T. Dhaene, "Complex Vector Fitting Toolbox," Electronics Letters
   57(10):404, 2021.
4. Y. Nakatsukasa, O. Sete, L.N. Trefethen, "The AAA Algorithm for Rational
   Approximation," SIAM J. Sci. Comput. 40(3):A1494, 2018.
5. S. Grivet-Talocia, B. Gustavsen, Passive Macromodeling: Theory and Applications,
   Wiley, 2016.
