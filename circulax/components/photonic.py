"""Photonic components for optical circuit simulation.

All wavelength parameters are in nanometres and power in watts.
"""

import jax.nn as jnn
import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, States, component, source
from circulax.s_transforms import fdomain_component, s_to_y

# ===========================================================================
# Passive Optical Components (S-Matrix based)
# ===========================================================================


@component(ports=("p1", "p2"), holomorphic=True)
def OpticalWaveguide(
    signals: Signals,
    s: States,
    length_um: float = 100.0,
    loss_dB_cm: float = 1.0,
    neff: float = 2.4,
    n_group: float = 4.0,
    center_wavelength_nm: float = 1310.0,
    wavelength_nm: float = 1310.0,
) -> PhysicsReturn:
    """Single-mode waveguide with first-order dispersion and propagation loss.

    The effective index is linearised around ``center_wavelength_nm`` using the
    group index to approximate dispersion. Phase and loss are combined into a
    complex transmission coefficient ``T``, from which the 2×2 S-matrix and
    corresponding Y-matrix are derived.

    Args:
        signals: Field amplitudes at input (``p1``) and output (``p2``).
        s: Unused.
        length_um: Waveguide length in micrometres. Defaults to ``100.0``.
        loss_dB_cm: Propagation loss in dB/cm. Defaults to ``1.0``.
        neff: Effective refractive index at ``center_wavelength_nm``. Defaults to ``2.4``.
        n_group: Group refractive index, used to compute the dispersion slope. Defaults to ``4.0``.
        center_wavelength_nm: Reference wavelength for dispersion expansion in nm. Defaults to ``1310.0``.
        wavelength_nm: Operating wavelength in nm. Defaults to ``1310.0``.

    """
    d_lam = wavelength_nm - center_wavelength_nm
    slope = (neff - n_group) / center_wavelength_nm
    n_eff_disp = neff + slope * d_lam

    # Phase calculation
    phi = 2.0 * jnp.pi * n_eff_disp * (length_um / wavelength_nm) * 1000.0

    # Loss calculation
    loss_val = loss_dB_cm * (length_um / 10000.0)
    T_mag = 10.0 ** (-loss_val / 20.0)

    # S-Matrix construction
    T = T_mag * jnp.exp(-1j * phi)
    S = jnp.array([[0.0, T], [T, 0.0]], dtype=jnp.complex128)

    # Convert to Admittance (Y)
    Y = s_to_y(S)

    # Calculate Currents (I = Y * V)
    # Note: Explicit complex cast ensures JAX treats the interaction as complex
    v_vec = jnp.array([signals.p1, signals.p2], dtype=jnp.complex128)
    i_vec = Y @ v_vec

    return {"p1": i_vec[0], "p2": i_vec[1]}, {}


@component(ports=("p1", "p2"), states=("i_p2",))
def OpticalDelayLine(
    signals: Signals,
    s: States,
    hist: Signals,
    length_um: float = 100.0,
    loss_dB_cm: float = 1.0,
    neff: float = 2.4,
    n_group: float = 4.0,
    wavelength_nm: float = 1310.0,
) -> PhysicsReturn:
    """Waveguide modelled as an explicit time-of-flight delay line (real group delay).

    Unlike :func:`OpticalWaveguide` -- which is a steady-state S-matrix stamp
    only valid for CW/frequency-domain analysis -- this component enforces
    the actual transient envelope delay: the output field at ``p2`` is the
    input field at ``p1`` from ``tau = length_um * n_group / c`` seconds ago,
    attenuated and phase-shifted, via a VCVS-style constraint (same pattern as
    :func:`TunableBeamSplitter`). ``p1`` carries no self-stamp and must be
    driven by a connected source/component, matching that same convention.

    See ``circulax.solvers.assembly`` for how ``hist`` (the delayed local
    state) is computed -- a fixed-size accepted-step history buffer read via
    ``jnp.interp``, vmapped per-instance since ``tau`` may vary across
    instances of different length. Adaptive step-size controllers (e.g.
    ``PIDController``) are supported (see gdsfactory/circulax#2): the
    proposed step is automatically clamped to the smallest active ``tau``
    across the circuit. ``ConstantStepSize`` remains unclamped -- an
    explicit ``dt0`` larger than ``tau`` still raises.

    Args:
        signals: Field amplitudes at input (``p1``) and output (``p2``).
        s: Branch current state variable ``i_p2``, carrying the VCVS current
            injected at the output port.
        hist: This instance's own port fields at ``t - tau``, injected by the
            solver. Only ``hist.p1`` is used.
        length_um: Waveguide length in micrometres. Defaults to ``100.0``.
        loss_dB_cm: Propagation loss in dB/cm. Defaults to ``1.0``.
        neff: Effective refractive index, used for the carrier phase shift
            over the delay. Defaults to ``2.4``.
        n_group: Group refractive index; sets the propagation delay via
            ``tau = length_um * n_group / c``. Defaults to ``4.0``.
        wavelength_nm: Operating wavelength in nm. Defaults to ``1310.0``.

    """
    phi = 2.0 * jnp.pi * neff * (length_um / wavelength_nm) * 1000.0
    loss_val = loss_dB_cm * (length_um / 10000.0)
    T_mag = 10.0 ** (-loss_val / 20.0)
    T = T_mag * jnp.exp(-1j * phi)

    constraint = signals.p2 - T * hist.p1

    return {"p1": 0.0, "p2": s.i_p2, "i_p2": constraint}, {}


@OpticalDelayLine.delay
def _optical_delay_line_tau(length_um: float = 100.0, n_group: float = 4.0) -> float:
    """Group delay ``tau = length / v_group`` in seconds (``length_um`` in micrometres)."""
    c_um_per_s = 2.99792458e14  # speed of light, um/s
    return (length_um * n_group) / c_um_per_s


@fdomain_component(ports=("p1", "p2"))
def delay_line_fdomain(
    f: float,
    length_um: float = 100.0,
    loss_dB_cm: float = 1.0,
    n_group: float = 4.0,
) -> jnp.ndarray:
    """Frequency-domain delay line for AC and harmonic-balance analysis.

    Models a pure transmission-line group delay: ``S21 = T_mag * exp(-j 2pi f tau)``
    where ``tau = length_um * n_group / c`` and ``T_mag`` accounts for
    propagation loss.  The S-matrix is converted to a Y-matrix via
    :func:`s_to_y`.

    This is a frequency-domain-only component (``f`` = modulation / signal
    frequency, e.g. 1 GHz for an AC sweep).  Carrier-phase effects
    (``neff``, ``wavelength_nm``) belong in the wavelength-domain
    :func:`OpticalWaveguide` and are intentionally excluded here.

    This component is AC/HB-only; it cannot be used in transient simulation
    (fdomain components raise at transient setup time).

    Args:
        f: Modulation frequency in Hz (supplied by the AC/HB solver).
        length_um: Waveguide length in micrometres. Defaults to ``100.0``.
        loss_dB_cm: Propagation loss in dB/cm. Defaults to ``1.0``.
        n_group: Group refractive index; sets the propagation delay via
            ``tau = length_um * n_group / c``. Defaults to ``4.0``.

    """
    c_um_per_s = 2.99792458e14
    loss_val = loss_dB_cm * (length_um / 10000.0)
    T_mag = 10.0 ** (-loss_val / 20.0)
    tau = (length_um * n_group) / c_um_per_s
    T = T_mag * jnp.exp(-1j * 2.0 * jnp.pi * f * tau)
    S = jnp.array([[0.0, T], [T, 0.0]], dtype=jnp.complex128)
    return s_to_y(S)


@component(ports=("grating", "waveguide"), holomorphic=True)
def Grating(
    signals: Signals,
    s: States,
    center_wavelength_nm: float = 1310.0,
    peak_loss_dB: float = 0.0,
    bandwidth_1dB: float = 20.0,
    wavelength_nm: float = 1310.0,
) -> PhysicsReturn:
    """Grating coupler with Gaussian wavelength-dependent insertion loss.

    Loss increases quadratically with detuning from ``center_wavelength_nm``,
    approximating the Gaussian spectral response of a typical grating coupler.
    Transmission is clipped to ``0.9999`` to keep the Y-matrix well-conditioned.

    Args:
        signals: Field amplitudes at the grating (``grating``) and waveguide (``waveguide``) ports.
        s: Unused.
        center_wavelength_nm: Peak transmission wavelength in nm. Defaults to ``1310.0``.
        peak_loss_dB: Insertion loss at peak wavelength in dB. Defaults to ``0.0``.
        bandwidth_1dB: Full 1 dB bandwidth in nm. Defaults to ``20.0``.
        wavelength_nm: Operating wavelength in nm. Defaults to ``1310.0``.

    """
    delta = wavelength_nm - center_wavelength_nm
    excess_loss = (delta / (0.5 * bandwidth_1dB)) ** 2
    loss_dB = peak_loss_dB + excess_loss

    T = 10.0 ** (-loss_dB / 20.0)
    # Numerical stability clip
    T = jnp.minimum(T, 0.9999)

    S = jnp.array([[0.0, T], [T, 0.0]], dtype=jnp.complex128)
    Y = s_to_y(S)

    v_vec = jnp.array([signals.grating, signals.waveguide], dtype=jnp.complex128)
    i_vec = Y @ v_vec

    return {"grating": i_vec[0], "waveguide": i_vec[1]}, {}


@component(ports=("p1", "p2", "p3"), holomorphic=True)
def Splitter(signals: Signals, s: States, split_ratio: float = 0.5) -> PhysicsReturn:
    """Lossless asymmetric optical splitter (Y-junction) with a configurable power split ratio.

    The S-matrix is constructed to be unitary, with the cross-port coupling
    carrying a ``j`` phase shift to satisfy energy conservation.

    Args:
        signals: Field amplitudes at the input (``p1``) and two output ports
            (``p2``, ``p3``).
        s: Unused.
        split_ratio: Fraction of input power routed to ``p2``. The remaining
            ``1 - split_ratio`` is routed to ``p3``. Defaults to ``0.5``
            (50/50 splitter).

    """
    r = jnp.sqrt(split_ratio)
    tc = jnp.sqrt(1.0 - split_ratio)

    S = jnp.array([[0.0, r, 1j * tc], [r, 0.0, 0.0], [1j * tc, 0.0, 0.0]], dtype=jnp.complex128)

    Y = s_to_y(S)

    v_vec = jnp.array([signals.p1, signals.p2, signals.p3], dtype=jnp.complex128)
    i_vec = Y @ v_vec

    return {"p1": i_vec[0], "p2": i_vec[1], "p3": i_vec[2]}, {}


@component(ports=("p1", "p2", "p3", "p4"), holomorphic=True)
def DirectionalCoupler(
    signals: Signals,
    s: States,
    coupling: float = 0.5,
) -> PhysicsReturn:
    """4-port directional coupler (2×2 beamsplitter) with configurable power split ratio.

    Ports p1/p2 are the input ports; p3/p4 are the output ports.
    Signal entering p1 exits at p3 (bar, amplitude t) and p4 (cross, amplitude jk).
    Signal entering p2 exits at p3 (cross, jk) and p4 (bar, t).

    S-matrix::

        S = [[0,    0,    t,    jk  ],
             [0,    0,    jk,   t   ],
             [t,    jk,   0,    0   ],
             [jk,   t,    0,    0   ]]

    where t = sqrt(1 - coupling), k = sqrt(coupling).

    .. note::
        ``coupling`` is clipped to (1e-6, 1−1e-6) to prevent ``I + S`` from becoming
        singular at the degenerate all-through (coupling=0) and all-cross (coupling=1)
        operating points where the S-matrix has an eigenvalue of −1.

    Args:
        signals: Field amplitudes at all four ports.
        s: Unused.
        coupling: Power fraction routed to the cross port (p4 for p1 input, p3 for p2 input).
            Defaults to ``0.5`` (50/50 splitter).

    """
    coupling = jnp.clip(coupling, 1e-6, 1.0 - 1e-6)
    t = jnp.sqrt(1.0 - coupling)
    k = jnp.sqrt(coupling)

    S = jnp.array(
        [
            [0.0,   0.0,   t,      1j * k],
            [0.0,   0.0,   1j * k, t     ],
            [t,     1j * k, 0.0,   0.0   ],
            [1j * k, t,    0.0,    0.0   ],
        ],
        dtype=jnp.complex128,
    )
    Y = s_to_y(S)
    v_vec = jnp.array([signals.p1, signals.p2, signals.p3, signals.p4], dtype=jnp.complex128)
    i_vec = Y @ v_vec
    return {"p1": i_vec[0], "p2": i_vec[1], "p3": i_vec[2], "p4": i_vec[3]}, {}


# ===========================================================================
# Optical Sources
# ===========================================================================


@component(ports=("p1", "p2"), states=("i_src",), holomorphic=True)
def OpticalSource(signals: Signals, s: States, power: float = 1.0, phase: float = 0.0) -> PhysicsReturn:
    """Ideal CW optical source for DC and small-signal AC analysis.

    Enforces a fixed complex field amplitude ``sqrt(power) * exp(j * phase)``
    across its ports, analogous to an ideal voltage source in electrical circuits.

    Args:
        signals: Field amplitudes at the positive (``p1``) and negative (``p2``) ports.
        s: Source current state variable ``i_src``.
        power: Output optical power in watts. Defaults to ``1.0``.
        phase: Output field phase in radians. Defaults to ``0.0``.

    """
    v_val = jnp.sqrt(power) * jnp.exp(1j * phase)
    constraint = (signals.p1 - signals.p2) - v_val

    return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}


@source(ports=("p1", "p2"), states=("i_src",), holomorphic=True)
def OpticalSourcePulse(
    signals: Signals,
    s: States,
    t: float,
    power: float = 1.0,
    phase: float = 0.0,
    delay: float = 0.2e-9,
    rise: float = 0.05e-9,
) -> PhysicsReturn:
    """Time-dependent optical source with a sigmoid turn-on profile.

    Field amplitude ramps smoothly from zero to ``sqrt(power)`` around
    ``delay``, with the steepness of the transition controlled by ``rise``.
    Suitable for transient simulations of optical pulse propagation.

    Args:
        signals: Field amplitudes at the positive (``p1``) and negative (``p2``) ports.
        s: Source current state variable ``i_src``.
        t: Current simulation time in seconds.
        power: Peak output optical power in watts. Defaults to ``1.0``.
        phase: Output field phase in radians. Defaults to ``0.0``.
        delay: Turn-on delay in seconds. Defaults to ``0.2e-9``.
        rise: Sigmoid rise time constant in seconds. Defaults to ``0.05e-9``.

    """
    val = jnp.sqrt(power) * jnn.sigmoid((t - delay) / rise)
    v_val = val * jnp.exp(1j * phase)

    constraint = (signals.p1 - signals.p2) - v_val

    return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}


# ===========================================================================
# Tunable Active Components
# ===========================================================================


@component(ports=("p1", "p2", "p3", "p4"), states=("i_top", "i_bot"), holomorphic=True)
def TunableBeamSplitter(
    signals: Signals,
    s: States,
    theta: float = jnp.pi / 4,
) -> PhysicsReturn:
    """Ideal lossless 2×2 directional coupler with a variable coupling angle.

    Implements a tunable beam splitter using a Modified Nodal Analysis (MNA)
    voltage-controlled voltage source (VCVS) stamp, which avoids the
    near-singular ``(I + S)`` matrix that arises when applying ``s_to_y`` to
    a lossless transmission-only S-matrix.

    The transmission relations enforced are::

        E_p3 = cos(theta) * E_p1 + j * sin(theta) * E_p2
        E_p4 = j * sin(theta) * E_p1 + cos(theta) * E_p2

    Special cases:

    * ``theta = 0``       — bar state: ``p1 → p3``, ``p2 → p4`` (no coupling).
    * ``theta = pi/4``    — 50/50 beamsplitter (default).
    * ``theta = pi/2``    — cross state: ``p1 → p4``, ``p2 → p3`` (full coupling).

    Two internal state variables ``i_top`` and ``i_bot`` carry the branch
    currents at the output ports ``p3`` and ``p4`` respectively.  The input
    ports ``p1`` and ``p2`` contribute no self-stamp; they must be driven by
    connected sources or loads to avoid floating nodes.

    Gradient flow is exact via ``jax.grad`` for all values of ``theta``
    when using the ``"dense"`` solver backend.

    Args:
        signals: Field amplitudes at all four ports.
        s: Branch current state variables ``i_top`` and ``i_bot``.
        theta: Coupling angle in radians.  Controls the power split ratio:
            ``P_cross / P_total = sin²(theta)``.  Defaults to ``pi/4`` (50/50).

    """
    c = jnp.cos(theta).astype(jnp.complex128)
    js = (1j * jnp.sin(theta)).astype(jnp.complex128)

    # Voltage constraints (VCVS): enforce transmission relations
    eq_top = signals.p3 - c * signals.p1 - js * signals.p2
    eq_bot = signals.p4 - js * signals.p1 - c * signals.p2

    return {
        "p1": 0.0,        # no self-stamp on input ports
        "p2": 0.0,
        "p3": s.i_top,    # VCVS branch current injected at output p3
        "p4": s.i_bot,    # VCVS branch current injected at output p4
        "i_top": eq_top,  # constraint: E_p3 = cos(θ)·E_p1 + j·sin(θ)·E_p2
        "i_bot": eq_bot,  # constraint: E_p4 = j·sin(θ)·E_p1 + cos(θ)·E_p2
    }, {}
