"""Base class and decorators for defining JAX-compatible circuit components.

Circuit components are defined as plain Python functions decorated with
:func:`component` or :func:`source`, which compile them into
:class:`CircuitComponent` subclasses — Equinox modules whose parameters are
JAX-traceable leaves. The resulting classes expose two entry points:

- ``__call__`` — a debug-friendly instance method that accepts port voltages
  and state values as keyword arguments and returns the physics dicts directly.
- ``solver_call`` — a class method used by the transient solver that operates
  on flat JAX arrays and a parameter container, and is compatible with
  ``jax.vmap`` and ``jax.jacfwd``.

Example::

    @component(ports=("p1", "p2"))
    def Resistor(signals: Signals, s: States, R: float = 1.0):
        i = (signals.p1 - signals.p2) / R
        return {"p1": i, "p2": -i}, {}


    r = Resistor(R=100.0)
    f, q = r(p1=1.0, p2=0.0)
"""

import inspect
from collections import namedtuple
from typing import Any, ClassVar, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

PhysicsReturn = tuple[dict[str, Array], dict[str, Array]]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class Signals(Protocol):
    """Protocol representing the port voltage signals passed to a component's physics function.

    Attributes are accessed by port name (e.g. ``signals.p1``), backed by a
    namedtuple constructed from the component's ``ports`` declaration.
    """

    def __getattr__(self, name: str) -> Any: ...


@runtime_checkable
class States(Protocol):
    """Protocol representing the internal state variables passed to a component's physics function.

    Attributes are accessed by state name (e.g. ``s.i_L``), backed by a
    namedtuple constructed from the component's ``states`` declaration.
    """

    def __getattr__(self, name: str) -> Any: ...


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class CircuitComponent(eqx.Module):
    """Base class for all JAX-compatible circuit components.

    Subclasses are not written by hand — they are generated at import time by
    the :func:`component` and :func:`source` decorators, which inspect the
    decorated function's signature to populate the class variables and wire up
    the two physics entry points.

    Class Variables:
        ports: Ordered tuple of port names, e.g. ``("p1", "p2")``.
        states: Ordered tuple of internal state variable names,
            e.g. ``("i_L",)``. Empty for purely algebraic components.
        _uses_time: ``True`` for components decorated with :func:`source`
            whose physics function accepts a ``t`` argument.
        _VarsType_P: Namedtuple type for unpacking port voltages from a
            flat array. ``None`` if the component has no ports.
        _VarsType_S: Namedtuple type for unpacking state variables from a
            flat array. ``None`` if the component has no states.
        _n_ports: Number of ports, cached to avoid repeated ``len`` calls
            in the hot path.
        _fast_physics: Static closure over the user-defined physics function,
            compatible with ``jax.vmap`` and ``jax.jacfwd``. Signature is
            ``(vars_vec, params, t) -> (f_vec, q_vec)``.
    """

    ports: ClassVar[tuple[str, ...]] = ()
    states: ClassVar[tuple[str, ...]] = ()

    _uses_time: ClassVar[bool] = False
    _is_fdomain: ClassVar[bool] = False
    _holomorphic: ClassVar[bool] = False

    _VarsType_P: ClassVar[Any] = None
    _VarsType_S: ClassVar[Any] = None
    _n_ports: ClassVar[int] = 0

    _fast_physics: ClassVar[Any] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialise namedtuple types and cached port count for each new subclass."""
        super().__init_subclass__(**kwargs)
        if cls.ports:
            cls._VarsType_P = namedtuple("Ports", cls.ports)  # noqa: PYI024
        if cls.states:
            cls._VarsType_S = namedtuple("States", cls.states)  # noqa: PYI024
        cls._n_ports = len(cls.ports)

    def __call__(
        self,
        t: Any = 0.0,
        y: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Evaluate the component physics (debug entry point).

        Accepts inputs either as a flat state vector ``y`` or as individual
        keyword arguments keyed by port and state name. A heuristic detects
        whether the first positional argument is a time value or a state vector,
        allowing the shorthand ``component(y)`` in addition to the explicit
        ``component(t=0.0, y=y)``.

        Args:
            t: Simulation time, or a state vector when called as
                ``component(y)`` without an explicit ``y`` keyword.
            y: Flat state vector of shape ``(n_ports + n_states,)``. When
                provided, port voltages are taken from ``y[:n_ports]`` and
                state values from ``y[n_ports:]``. Mutually exclusive with
                ``**kwargs``.
            **kwargs: Port voltages and state values by name, e.g.
                ``p1=1.0, p2=0.0``. Used when ``y`` is not provided.
                Missing names default to ``0.0``.

        Returns:
            A two-tuple ``(f, q)`` of dicts mapping port/state names to their
            resistive (``f``) and reactive (``q``) contributions respectively.

        """
        hist = kwargs.pop("hist", None)

        if y is None and not kwargs:
            is_scalar = isinstance(t, (int, float)) or (hasattr(t, "shape") and t.shape == ())
            if not is_scalar:
                y = t
                t = 0.0

        if y is not None:
            n_p = self._n_ports
            signals = self._VarsType_P(*y[:n_p]) if self._VarsType_P else ()
            s = self._VarsType_S(*y[n_p:]) if self._VarsType_S else ()
        else:

            def _get_args(names: tuple[str, ...]) -> list[Any]:
                return [kwargs.get(name, 0.0) for name in names]

            signals = self._VarsType_P(*_get_args(self.ports)) if self._VarsType_P else ()
            s = self._VarsType_S(*_get_args(self.states)) if self._VarsType_S else ()

        return self._invoke_physics(signals, s, t, self, hist)

    @classmethod
    def solver_call(
        cls,
        t: float,
        y: jax.Array,
        args: Any,
        hist: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate the component physics (solver entry point).

        Thin wrapper around the static ``_fast_physics`` closure. Called by
        the transient solver inside ``jax.vmap`` across all instances in a
        component group, and differentiated via ``jax.jacfwd`` to assemble
        the system Jacobian.

        Args:
            t: Current simulation time.
            y: Flat state vector of shape ``(n_ports + n_states,)`` containing
                port voltages followed by state variable values.
            args: Parameter container for this instance. May be a dict
                ``{"R": 100.0}`` or an object (e.g. the component instance
                itself) whose attributes match the parameter names. Must not
                be a raw scalar.
            hist: Delayed port-voltage vector of shape ``(n_ports,)``, i.e.
                this instance's own ports evaluated at ``t - tau`` rather than
                ``t``. Only consumed by components declaring a ``hist``
                argument (see :func:`component`/:func:`source`); ``None``
                otherwise.

        Returns:
            A two-tuple ``(f_vec, q_vec)`` of JAX arrays, each of shape
            ``(n_ports + n_states,)``, containing the resistive and reactive
            contributions for every port and state variable.

        """
        return cls._fast_physics(y, args, t, hist)

    # -----------------------------------------------------------------------
    # Internal Dispatchers (wired up by decorator)
    # -----------------------------------------------------------------------
    def physics(self, *args: Any, **kwargs: Any) -> tuple[dict, dict]:
        """Raw physics dispatch; overridden by the decorator-generated subclass."""
        raise NotImplementedError

    def _invoke_physics(
        self,
        signals: Any,
        s: Any,
        t: float,
        params: Any,
        hist: Any = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Trampoline used by ``__call__`` to dispatch to the user physics function.

        Overridden by the decorator-generated subclass with a closure that
        extracts named parameters from ``params`` and forwards them as keyword
        arguments to the original decorated function.

        Args:
            signals: Namedtuple of port voltages.
            s: Namedtuple of state variable values.
            t: Current simulation time.
            params: Parameter container (instance or dict).
            hist: Delayed port-voltage namedtuple/tuple, or ``None``. Only
                consumed by components declaring a ``hist`` argument.

        Returns:
            A two-tuple ``(f, q)`` of physics dicts.

        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helper: Parameter Extraction
# ---------------------------------------------------------------------------
def _extract_param(container: Any, name: str) -> Any:
    """Extract a named parameter from either a dict or an object.

    Args:
        container: A dict or any object with named attributes.
        name: The parameter name to look up.

    Returns:
        The parameter value.

    """
    if isinstance(container, dict):
        return container[name]
    return getattr(container, name)


# ---------------------------------------------------------------------------
# Holomorphic validation via jaxpr inspection
# ---------------------------------------------------------------------------
_NON_HOLOMORPHIC_PRIMITIVES = frozenset({"real", "imag", "conj", "abs"})


def _jaxpr_has_non_holomorphic(jaxpr: Any) -> set[str]:
    """Walk a jaxpr and return names of non-holomorphic primitives on traced variables."""
    from jax._src.core import ClosedJaxpr, Jaxpr, Literal

    found: set[str] = set()
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in _NON_HOLOMORPHIC_PRIMITIVES:
            has_traced_complex_input = any(
                not isinstance(v, Literal) and hasattr(v.aval, "dtype") and jnp.issubdtype(v.aval.dtype, jnp.complexfloating)
                for v in eqn.invars
            )
            if has_traced_complex_input:
                found.add(eqn.primitive.name)
        for v in eqn.params.values():
            if isinstance(v, Jaxpr):
                found |= _jaxpr_has_non_holomorphic(v)
            elif isinstance(v, ClosedJaxpr):
                found |= _jaxpr_has_non_holomorphic(v.jaxpr)
    return found


# ---------------------------------------------------------------------------
# The Builder
# ---------------------------------------------------------------------------
def _build_component(  # noqa: C901
    fn: Any,
    ports: tuple[str, ...],
    states: tuple[str, ...],
    *,
    uses_time: bool,
    amplitude_param: str = "",
    setup_fn: Any = None,
    differentiable_params: "tuple[str, ...] | None" = None,
    port_aliases: "dict[str, str | tuple[str, ...]] | None" = None,
    holomorphic: bool = False,
) -> type[CircuitComponent]:
    """Compile a physics function into a :class:`CircuitComponent` subclass.

    Inspects ``fn``'s signature, validates it against the declared ``ports``
    and ``states``, performs a dry-run with default values to catch errors
    early, then builds two closures:

    - ``_fast_physics`` — a static function ``(vars_vec, params, t)`` suitable
      for ``jax.vmap`` / ``jax.jacfwd``, used by the solver.
    - ``_invoke_physics`` — a bound method used by the debug ``__call__`` path.

    Args:
        fn: The decorated physics function. Its signature must begin with
            ``(signals, s)`` for :func:`component` or ``(signals, s, t)``
            for :func:`source`, followed by any number of keyword-only
            parameters with defaults.
        ports: Ordered tuple of port names matching the netlist connections.
        states: Ordered tuple of internal state variable names.
        uses_time: ``True`` when compiling a :func:`source` component whose
            physics function accepts a time argument ``t``.
        amplitude_param: Name of the parameter representing the source
            amplitude for DC homotopy stepping.  Empty string for passives.
        port_aliases: Optional mapping from a canonical port name (must be a
            member of ``ports``) to one or more alternate port names that may
            appear in a netlist instead (e.g. ``{"p1": "P", "p2": "N"}``).
            Consumed by ``compiler.py``'s ``_resolve_port_index`` so callers
            can wire up a netlist using either name without duplicating the
            physics.

    Returns:
        A new :class:`CircuitComponent` subclass named after ``fn``.

    Raises:
        TypeError: If the function signature does not start with the required
            reserved arguments, if any parameter lacks a default value, if a
            non-source component declares a ``t`` parameter, or if the dry-run
            raises an exception.
        ValueError: If ``port_aliases`` references a canonical name that is
            not in ``ports``.

    """
    reserved = ("signals", "s", "t") if uses_time else ("signals", "s")

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if len(params) < len(reserved):
        msg = f"Function '{fn.__name__}' must start with arguments {reserved}"
        raise TypeError(msg)
    for i, expected in enumerate(reserved):
        if params[i].name != expected:
            msg = f"Arg #{i + 1} must be '{expected}'"
            raise TypeError(msg)

    param_specs = params[len(reserved) :]

    # Optional analog-init slot. When the physics function declares ``init``
    # as the first non-reserved positional argument, the framework will
    # inject the return value of a ``@<Component>.setup``-registered
    # function there at evaluation time. The ``init`` arg is NOT a regular
    # JAX-traced parameter — it is computed from the other params each
    # call, allowing gradients to flow through the setup body and matching
    # Verilog-A's ``analog initial`` semantic.
    has_init_arg = bool(param_specs) and param_specs[0].name == "init"
    if has_init_arg:
        param_specs = param_specs[1:]

    # Optional delay-history slot. When the physics function declares ``hist``
    # as the next non-reserved positional argument (after ``init``, if
    # present), the framework injects the component's own port voltages
    # evaluated at ``t - tau`` instead of ``t``, where ``tau`` is computed by
    # a ``@<Component>.delay``-registered function. Requires ``ports`` to be
    # non-empty. See ``circulax.solvers.assembly`` for how the delayed read
    # is computed (a fixed-size accepted-step history buffer + ``jnp.interp``,
    # vmapped per-instance since ``tau`` may vary across instances).
    has_hist_arg = bool(param_specs) and param_specs[0].name == "hist"
    if has_hist_arg:
        param_specs = param_specs[1:]
        if not ports:
            msg = f"{fn.__name__}: 'hist' argument requires non-empty ports."
            raise TypeError(msg)

    if not uses_time:
        for p in param_specs:
            if p.name == "t":
                msg = "Use @source for time-dependent components."
                raise TypeError(msg)

    for p in param_specs:
        if p.default is inspect.Parameter.empty:
            msg = f"Parameter '{p.name}' must have a default."
            raise TypeError(msg)

    _dummy_P = namedtuple("Ports", ports)(*([0.0] * len(ports))) if ports else ()  # noqa: PYI024
    _dummy_S = namedtuple("States", states)(*([0.0] * len(states))) if states else ()  # noqa: PYI024
    _defaults = {p.name: p.default for p in param_specs}

    # Dry-run only validates non-injected args; ``init``/``hist`` will be
    # injected by the closure once registered via ``.setup``/``.delay`` —
    # pass structural placeholders (empty dict / zero ports) here instead.
    _dry_positional = [_dummy_P, _dummy_S]
    if uses_time:
        _dry_positional.append(0.0)
    if has_init_arg:
        _dry_positional.append({})
    if has_hist_arg:
        _dry_positional.append(_dummy_P)

    try:
        fn(*_dry_positional, **_defaults)
    except Exception as exc:
        # Bodies that index into init with concrete keys will fail the
        # placeholder dry-run; that's expected and harmless. Suppress only
        # KeyError/IndexError/AttributeError, which are the typical "init
        # was the empty placeholder" failure modes — anything else is a
        # real signature bug.
        if not (has_init_arg and isinstance(exc, (KeyError, IndexError, AttributeError, TypeError))):
            raise TypeError(f"Dry-run failed: {exc}") from exc

    n_p = len(ports)
    full_keys = ports + states
    _param_names = tuple(p.name for p in param_specs)
    _user_fn = fn
    _PortsType = namedtuple("Ports", ports) if ports else None  # noqa: PYI024
    _StatesType = namedtuple("States", states) if states else None  # noqa: PYI024

    # Mutable cell for the analog-init function — populated by the
    # ``@<Component>.setup`` classmethod after class construction. The
    # closure below reads from this list so re-registration works without
    # rebuilding ``_fast_physics``.
    _setup_cell: list[Any] = [None]

    def _resolve_init(kw: dict[str, Any]) -> Any:
        """Run the registered setup fn with the current params, returning init.

        Called from inside ``_fast_physics`` / ``_invoke_physics`` when the
        physics signature declares an ``init`` argument. The setup body is
        re-traced each call — XLA constant-folds it for static params,
        and gradients flow through it for differentiable ones.
        """
        setup_fn = _setup_cell[0]
        if setup_fn is None:
            msg = (
                f"{fn.__name__}: physics function declares an 'init' argument "
                f"but no setup function has been registered. "
                f"Use ``@{fn.__name__}.setup`` to register one."
            )
            raise RuntimeError(msg)
        # Pass through only kwargs the setup function declares — ergonomic
        # for setups that consume a subset of the physics params.  When the
        # setup wrapper uses **kwargs (e.g. emitted _register_setup), pass
        # everything so the inner function receives the actual param values.
        setup_sig = inspect.signature(setup_fn)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in setup_sig.parameters.values()
        )
        if has_var_kw:
            return setup_fn(**kw)
        accepts = {p.name for p in setup_sig.parameters.values()}
        sub_kw = {k: v for k, v in kw.items() if k in accepts}
        return setup_fn(**sub_kw)

    # Mutable cell for the delay function — populated by the
    # ``@<Component>.delay`` classmethod after class construction, mirroring
    # ``_setup_cell`` above.
    _tau_cell: list[Any] = [None]

    def _resolve_tau(kw: dict[str, Any]) -> Any:
        """Run the registered delay fn with the current params, returning tau."""
        tau_fn = _tau_cell[0]
        if tau_fn is None:
            msg = (
                f"{fn.__name__}: physics function declares a 'hist' argument "
                f"but no delay function has been registered. "
                f"Use ``@{fn.__name__}.delay`` to register one."
            )
            raise RuntimeError(msg)
        tau_sig = inspect.signature(tau_fn)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in tau_sig.parameters.values()
        )
        if has_var_kw:
            return tau_fn(**kw)
        accepts = {p.name for p in tau_sig.parameters.values()}
        sub_kw = {k: v for k, v in kw.items() if k in accepts}
        return tau_fn(**sub_kw)

    def _build_positional(signals: Any, s: Any, t: float, init_value: Any, hist_signals: Any) -> list[Any]:
        positional = [signals, s]
        if uses_time:
            positional.append(t)
        if has_init_arg:
            positional.append(init_value)
        if has_hist_arg:
            positional.append(hist_signals)
        return positional

    if len(full_keys) == 0:
        _fast_physics = lambda v, p, t, hist=None: (jnp.zeros(0), jnp.zeros(0))  # noqa: E731
    else:

        def _fast_physics(
            vars_vec: jax.Array,
            params: Any,
            t: float,
            hist: jax.Array | None = None,
        ) -> tuple[jax.Array, jax.Array]:
            signals = _PortsType(*vars_vec[:n_p]) if _PortsType else ()
            s = _StatesType(*vars_vec[n_p:]) if _StatesType else ()
            kw = {name: _extract_param(params, name) for name in _param_names}
            init_value = _resolve_init(kw) if has_init_arg else None
            # hist is None outside the delay-aware solver path (e.g. the DC
            # operating-point solve in circulax.solvers.linear, which has no
            # history buffer yet) -- default to zero so those callers don't
            # need to know about delayed components at all.
            hist_signals = (
                _PortsType(*(hist[:n_p] if hist is not None else jnp.zeros_like(vars_vec[:n_p])))
                if has_hist_arg
                else None
            )
            positional = _build_positional(signals, s, t, init_value, hist_signals)
            f_dict, q_dict = _user_fn(*positional, **kw)
            f_vals = [f_dict.get(k, 0.0) for k in full_keys]
            q_vals = [q_dict.get(k, 0.0) for k in full_keys]
            return jnp.array(f_vals), jnp.array(q_vals)

    if uses_time:

        def _invoke_physics(
            self: CircuitComponent,
            signals: Any,
            s: Any,
            t: float,
            params: Any,
            hist: Any = None,
        ) -> tuple[dict, dict]:
            kw = {name: _extract_param(params, name) for name in _param_names}
            init_value = _resolve_init(kw) if has_init_arg else None
            hist_signals = _PortsType(*(hist if hist is not None else [0.0] * n_p)) if has_hist_arg else None
            positional = _build_positional(signals, s, t, init_value, hist_signals)
            return _user_fn(*positional, **kw)
    else:

        def _invoke_physics(
            self: CircuitComponent,
            signals: Any,
            s: Any,
            t: float,
            params: Any,
            hist: Any = None,
        ) -> tuple[dict, dict]:
            kw = {name: _extract_param(params, name) for name in _param_names}
            init_value = _resolve_init(kw) if has_init_arg else None
            hist_signals = _PortsType(*(hist if hist is not None else [0.0] * n_p)) if has_hist_arg else None
            positional = _build_positional(signals, s, t, init_value, hist_signals)
            return _user_fn(*positional, **kw)

    annotations = {p.name: (p.annotation if p.annotation is not inspect.Parameter.empty else Any) for p in param_specs}

    # Distinguish JAX-traced params (numeric scalars / arrays) from static
    # config fields (strings, None, bools). Static fields are carried on the
    # Equinox module but not stacked across instances or traced through
    # transformations — useful for absorbing GDSFactory-style "config"
    # settings like ``cross_section="strip"`` or ``width=None``.
    #
    # ``differentiable_params`` (from va_component) overrides the default
    # staticness for numeric params:
    #   - ``()``      → all numeric params are static (fastest; no jax.grad)
    #   - ``None``    → all numeric params are dynamic JAX leaves (full grad)
    #   - ``("W",)``  → only named params are dynamic; rest are static
    def _is_static(p: Any) -> bool:
        if p.default is None or isinstance(p.default, (str, bool)):
            return True
        if differentiable_params is None:
            return False
        return p.name not in differentiable_params

    _static_names: set[str] = set()
    defaults: dict[str, Any] = {}
    for p in param_specs:
        if _is_static(p):
            defaults[p.name] = eqx.field(default=p.default, static=True)
            _static_names.add(p.name)
        else:
            defaults[p.name] = p.default
    _diff_param_names_tuple = tuple(p.name for p in param_specs if p.name not in _static_names)

    def _register_setup(cls_inner: type, setup_fn: Any) -> type:
        """Register an analog-init / setup function on the component class.

        Used as ``@<Component>.setup`` decorator. The registered function
        will be called inside the JAX trace each time the component is
        evaluated, with the same parameters the physics function receives
        (filtered to those the setup function declares). Its return value
        is injected as the ``init`` argument of the physics function —
        the physics function MUST declare ``init`` as the first
        non-reserved positional argument for ``.setup`` to work.

        Returns ``cls_inner`` for decorator chaining.

        Raises:
            TypeError: physics function did not declare ``init`` arg.
            RuntimeError: ``.setup`` was already registered on this class.

        """
        if not has_init_arg:
            msg = (
                f"@{fn.__name__}.setup requires the physics function to declare "
                f"an 'init' argument as the first non-reserved positional "
                f"parameter. Add ``init`` to the signature, e.g. "
                f"``def {fn.__name__}(signals, s, init, R=1.0): ...``"
            )
            raise TypeError(msg)
        if _setup_cell[0] is not None:
            msg = (
                f"@{fn.__name__}.setup is already registered. Re-register "
                f"is intentionally rejected to catch typos; if you really "
                f"want to replace it, set ``cls._setup_fn_ref = None`` first."
            )
            raise RuntimeError(msg)
        if not callable(setup_fn):
            msg = f"@{fn.__name__}.setup expects a callable; got {type(setup_fn).__name__}"
            raise TypeError(msg)
        _setup_cell[0] = setup_fn
        cls_inner._setup_fn_ref = staticmethod(setup_fn)
        return cls_inner

    def _register_delay(cls_inner: type, tau_fn: Any) -> type:
        """Register a fixed-delay function on the component class.

        Used as ``@<Component>.delay`` decorator. The registered function is
        called with the component's own params (filtered to those it
        declares) and must return the delay ``tau`` — the physics function
        then receives ``hist`` as this instance's own port voltages at
        ``t - tau``. Requires the physics function to declare ``hist`` as a
        reserved argument (see :func:`_build_component`).

        Returns ``cls_inner`` for decorator chaining.

        Raises:
            TypeError: physics function did not declare a ``hist`` arg.
            RuntimeError: ``.delay`` was already registered on this class.

        """
        if not has_hist_arg:
            msg = (
                f"@{fn.__name__}.delay requires the physics function to declare "
                f"a 'hist' argument (after 'init', if present). Add ``hist`` to "
                f"the signature, e.g. ``def {fn.__name__}(signals, s, hist, "
                f"length_um=100.0): ...``"
            )
            raise TypeError(msg)
        if _tau_cell[0] is not None:
            msg = (
                f"@{fn.__name__}.delay is already registered. Re-register "
                f"is intentionally rejected to catch typos."
            )
            raise RuntimeError(msg)
        if not callable(tau_fn):
            msg = f"@{fn.__name__}.delay expects a callable; got {type(tau_fn).__name__}"
            raise TypeError(msg)
        _tau_cell[0] = tau_fn
        cls_inner._tau_fn_ref = staticmethod(tau_fn)
        return cls_inner

    def _tau_of(params: Any) -> Any:
        """Compute tau for a single instance's params — vmapped by assembly.py over a group."""
        kw = {name: _extract_param(params, name) for name in _param_names}
        return _resolve_tau(kw)

    namespace = {
        "__annotations__": annotations,
        "ports": ports,
        "states": states,
        "_fast_physics": staticmethod(_fast_physics),
        "_invoke_physics": _invoke_physics,
        "_uses_time": uses_time,
        "_holomorphic": holomorphic,
        "amplitude_param": amplitude_param,
        "_has_init_arg": has_init_arg,
        "_setup_fn_ref": None,
        "setup": classmethod(_register_setup),
        "_has_hist_arg": has_hist_arg,
        "_has_delay": has_hist_arg,
        "_tau_fn_ref": None,
        "delay": classmethod(_register_delay),
        "tau_of": staticmethod(_tau_of),
        # Expose static/diff param name splits for _install_custom_jvp.
        "_static_param_names": tuple(sorted(_static_names)),
        "_diff_param_names": _diff_param_names_tuple,
        **defaults,
    }

    cls = type(fn.__name__, (CircuitComponent,), namespace)
    cls.__doc__ = fn.__doc__

    if port_aliases:
        normalized_aliases: dict[str, tuple[str, ...]] = {}
        for canonical, raw in port_aliases.items():
            if canonical not in ports:
                msg = f"{fn.__name__}: port_aliases key {canonical!r} is not in ports {ports}"
                raise ValueError(msg)
            normalized_aliases[canonical] = (raw,) if isinstance(raw, str) else tuple(raw)
        cls._sanitized_to_raw_ports = normalized_aliases

    return cls


def component(
    ports: tuple[str, ...] = (),
    states: tuple[str, ...] = (),
    amplitude_param: str = "",
    port_aliases: "dict[str, str | tuple[str, ...]] | None" = None,
    holomorphic: bool = False,
) -> Any:
    """Decorator for defining a time-independent circuit component.

    Compiles the decorated physics function into a :class:`CircuitComponent`
    subclass. The function must begin with ``(signals, s)`` followed by any
    number of parameters with defaults, which become JAX-traceable Equinox
    fields on the resulting class.

    Args:
        ports: Ordered tuple of port names. Must match the connection keys
            used in the netlist.
        states: Ordered tuple of internal state variable names. State
            variables are appended to the solver's state vector after the
            node voltages.
        amplitude_param: Name of the parameter that represents the source
            amplitude (e.g. ``"I"`` for a current source). When non-empty,
            DC homotopy solvers will scale this parameter during stepping.
            Leave empty (default) for passive components.
        port_aliases: Optional mapping from a canonical port name to one or
            more alternate names a netlist may use instead, e.g.
            ``{"p1": "P", "p2": "N"}``. Lets integrators (e.g. Mosaic/kfnetlist)
            reuse this component's physics under their own port-naming
            convention without redefining it.
        holomorphic: If ``True``, the component's physics function is
            holomorphic (complex-differentiable), enabling the faster N×N
            Wirtinger AC solve.  Defaults to ``False`` (safe: the full 2N×2N
            real-block system is always correct).  Set to ``True`` for
            components whose physics uses only holomorphic operations — if
            *every* component in a circuit is holomorphic, the AC sweep
            uses the faster N×N path automatically.

    Returns:
        A decorator that accepts a physics function and returns a
        :class:`CircuitComponent` subclass.

    Example::

        @component(ports=("p1", "p2"))
        def Resistor(signals: Signals, s: States, R: float = 1.0):
            i = (signals.p1 - signals.p2) / R
            return {"p1": i, "p2": -i}, {}

    """
    return lambda fn: _build_component(
        fn, ports, states, uses_time=False, amplitude_param=amplitude_param, port_aliases=port_aliases, holomorphic=holomorphic
    )


def source(
    ports: tuple[str, ...] = (),
    states: tuple[str, ...] = (),
    amplitude_param: str = "",
    port_aliases: "dict[str, str | tuple[str, ...]] | None" = None,
    holomorphic: bool = False,
) -> Any:
    """Decorator for defining a time-dependent circuit component.

    Identical to :func:`component` except the decorated physics function
    must accept ``t`` as its third argument (after ``signals`` and ``s``),
    and may use it to implement time-varying behaviour such as sinusoidal
    sources or delayed step functions.

    Args:
        ports: Ordered tuple of port names.
        states: Ordered tuple of internal state variable names.
        amplitude_param: Name of the parameter that represents the source
            amplitude (e.g. ``"V"`` for a voltage source). When non-empty,
            DC homotopy solvers will scale this parameter during stepping.
            Leave empty (default) for time-dependent components that are not
            primary excitation sources.
        port_aliases: Optional mapping from a canonical port name to one or
            more alternate names a netlist may use instead. See
            :func:`component`.
        holomorphic: If ``True``, the component's physics is holomorphic.
            Defaults to ``False`` (safe).  See :func:`component`.

    Returns:
        A decorator that accepts a physics function and returns a
        :class:`CircuitComponent` subclass.

    Example::

        @source(ports=("p1", "p2"), states=("i_src",))
        def VoltageSource(signals: Signals, s: States, t: float, V: float = 1.0):
            constraint = (signals.p1 - signals.p2) - V
            return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}

    """
    return lambda fn: _build_component(
        fn, ports, states, uses_time=True, amplitude_param=amplitude_param, port_aliases=port_aliases, holomorphic=holomorphic
    )
