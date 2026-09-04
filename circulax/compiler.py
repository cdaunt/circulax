"""Compiles the group into ComponentGroups and organizes the node index."""

import dataclasses
import inspect
from collections import defaultdict
from functools import cache, wraps
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import kfnetlist as kfnl

from circulax.components.base_component import PhysicsReturn, Signals
from circulax.netlist import build_net_map_kfnetlist, sax_to_kfnetlist

try:
    from bosdi.circulax import _BOSDI_AVAILABLE, OsdiComponentGroup, OsdiModelDescriptor
except ImportError:
    OsdiComponentGroup = None  # type: ignore[assignment,misc]
    OsdiModelDescriptor = None  # type: ignore[assignment]
    _BOSDI_AVAILABLE = False


def ensure_time_signature(model_func: callable) -> callable:
    """Wraps a model function to ensure it accepts a 't' keyword argument.

    If the original model doesn't take 't', the wrapper swallows it.
    """
    sig = inspect.signature(model_func)

    if "t" in sig.parameters or "**kwargs" in str(sig):
        return model_func

    # Wrapper for static models
    @wraps(model_func)
    def time_aware_wrapper(signal_states: Signals, params: dict, t: float | None = None) -> PhysicsReturn:  # noqa: ARG001
        return model_func(signal_states, params)

    return time_aware_wrapper


class ComponentGroup(eqx.Module):
    """Represents a BATCH of identical components (e.g., ALL Resistors).
    Optimized for jax.vmap in the solver.
    """  # noqa: D205

    name: str = eqx.field(static=True)
    physics_func: callable = eqx.field(static=True)

    # BATCHED PARAMETERS:
    # This is the batched component instance (e.g. a Resistor where self.R is an array).
    # It acts as 'self' when passed to the physics function via vmap.
    params: Any

    # BATCHED STATE INDICES:
    # Shape (N, num_vars_per_component) - Used to gather 'v' from y0
    var_indices: jnp.ndarray

    # BATCHED EQUATION INDICES:
    # Shape (N, num_eqs_per_component) - Used to scatter 'i' into residual
    eq_indices: jnp.ndarray

    # FLATTENED JACOBIAN INDICES:
    # Shape (N * num_vars * num_vars,) - Concatenated for BCOO construction
    jac_rows: jnp.ndarray
    jac_cols: jnp.ndarray

    index_map: dict[str, int] | None = eqx.field(static=True, default=None)
    is_fdomain: bool = eqx.field(static=True, default=False)
    amplitude_param: str = eqx.field(static=True, default="")
    combined_func: Any = eqx.field(static=True, default=None)
    holomorphic: bool = eqx.field(static=True, default=False)

    # Fixed time-delay support (see circulax.solvers.assembly). ``has_delay``
    # gates the delay-history lookup so undelayed groups pay no cost;
    # ``tau_func(params) -> tau`` computes a single instance's delay, vmapped
    # over the group's batched params by the assembly layer.
    has_delay: bool = eqx.field(static=True, default=False)
    tau_func: Any = eqx.field(static=True, default=None)


def get_model_width(func: callable) -> int:
    """Determines the size of the 'vars' vector expected by the model."""
    sig = inspect.signature(func)
    if "vars" not in sig.parameters:
        msg = f"{func.__name__} missing 'vars' argument"
        raise ValueError(msg)
    default_val = sig.parameters["vars"].default
    if default_val is inspect.Parameter.empty:
        msg = f"{func.__name__} 'vars' must have a default (e.g. jnp.zeros(3))"
        raise ValueError(msg)
    return len(default_val)


# --- Main Compiler ---


def merge_dicts(dict_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Merges a list of dictionaries into a single dictionary."""
    merged = {}
    for d in dict_list:
        merged.update(d)
    return merged


@cache
def _get_default_params_cached(func: callable) -> dict[str, Any]:
    sig = inspect.signature(func)

    if "params" not in sig.parameters:
        msg = f"{func.__name__} missing 'params' argument"
        raise ValueError(msg)

    default_params = sig.parameters["params"].default

    if default_params is inspect.Parameter.empty:
        msg = f"{func.__name__} 'params' must have a default value (e.g. {{'R': 100.0}})"
        raise ValueError(msg)

    return default_params


def get_default_params(func: callable) -> dict[str, Any]:
    """Return a copy so callers can't mutate the cache."""
    return dict(_get_default_params_cached(func))


def solve_connectivity(connections: dict) -> dict:  # noqa: C901
    """Resolves Port-to-Port connections into a Port-to-NodeID map.

    Example: {"R1,p1": "V1,p1"} -> {"R1,p1": 1, "V1,p1": 1}
    """
    parent = {}

    def find(i: int):  # noqa: ANN202
        if i not in parent:
            parent[i] = i
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # 1. Process all connections
    for src, targets in connections.items():
        # Ensure 'targets' is a list
        if not isinstance(targets, (list, tuple)):
            targets = [targets]

        # Link Source to all Targets
        for tgt in targets:
            union(src, tgt)

    # 2. Assign Node IDs
    # We reserve ID 0 for Ground (any group containing "GND")
    groups = {}
    node_map = {}

    # Identify the root for "GND" if it exists
    gnd_roots = {find(k) for k in parent if "GND" in k}

    node_counter = 1

    for port in parent:
        root = find(port)

        if root in gnd_roots:
            node_id = 0
        else:
            if root not in groups:
                groups[root] = node_counter
                node_counter += 1
            node_id = groups[root]

        node_map[port] = node_id

    return node_map, node_counter


def compile_netlist(  # noqa: C901, PLR0912, PLR0915
    netlist: dict | kfnl.Netlist,
    models_map: dict,
    *,
    params_map: dict[str, dict[str, str]] | None = None,
) -> tuple[dict, int, dict]:
    """Compile a netlist into batched, vectorized component groups ready for simulation.

    Accepts either a ``kfnetlist.Netlist`` (preferred) or a SAX-format dict
    (auto-converted via :func:`sax_to_kfnetlist`).

    Returns:
        ``(compiled_groups, sys_size, port_to_node_map)``

    """
    _RESERVED = frozenset({"ground"})

    from circulax.s_transforms import _normalize_model

    def _maybe_normalize(k: str, v: Any) -> Any:
        if OsdiModelDescriptor is not None and isinstance(v, OsdiModelDescriptor):
            return v
        return _normalize_model(v, name=k)

    models_map = {k: _maybe_normalize(k, v) for k, v in models_map.items() if k not in _RESERVED}

    # --- 1. Normalize to kfnetlist.Netlist ---
    settings_override: dict[str, dict[str, Any]] = {}
    if isinstance(netlist, dict):
        netlist, settings_override = sax_to_kfnetlist(netlist)

    port_to_node_map, num_nodes = build_net_map_kfnetlist(netlist)

    def _port_candidates(comp_cls: type, port: str) -> tuple[str, ...]:
        aliases = getattr(comp_cls, "_sanitized_to_raw_ports", {})
        raw_aliases = tuple(raw for raw in aliases.get(port, ()) if raw != port)
        return (port, *raw_aliases)

    def _resolve_port_index(comp_cls: type, inst_name: str, port: str) -> int:
        candidates = _port_candidates(comp_cls, port)
        for candidate in candidates:
            key = f"{inst_name},{candidate}"
            if key in port_to_node_map:
                idx = port_to_node_map[key]
                port_to_node_map.setdefault(f"{inst_name},{port}", idx)
                return idx
        expected = f"{inst_name},{port}"
        if len(candidates) > 1:
            expected += f" (aliases: {', '.join(f'{inst_name},{p}' for p in candidates[1:])})"
        msg = f"Port '{port}' on '{inst_name}' is unconnected.\nYour netlist connections must include '{expected}'"
        raise ValueError(msg)

    # Buckets: Key = (comp_type_name, tree_structure), Value = list of instances
    buckets = defaultdict(list)
    # Separate bucket for OSDI instances (keyed by comp_type only)
    osdi_buckets: dict[str, list] = defaultdict(list)
    sys_size = num_nodes

    # --- 2. Process Instances ---
    for name, inst in netlist.instances.items():
        comp_type = inst.component

        if comp_type == "ground" or name == "GND":
            continue

        if comp_type not in models_map:
            msg = f"Model '{comp_type}' not found for '{name}'"
            raise ValueError(msg)

        comp_cls = models_map[comp_type]
        settings = settings_override.get(name, inst.settings or {})

        # Apply parameter name mapping (cell settings → model fields).
        if params_map and comp_type in params_map:
            rename = params_map[comp_type]
            settings = {rename.get(k, k): v for k, v in settings.items()}

        # OSDI components use a descriptor object instead of an Equinox class.
        if OsdiModelDescriptor is not None and isinstance(comp_cls, OsdiModelDescriptor):
            if not _BOSDI_AVAILABLE:
                raise ImportError(
                    f"Component '{name}' uses an OSDI model but the bosdi runtime "
                    "(osdi_loader) is not available. Install circulax[verilog-a] to "
                    "enable OSDI support. Note: OSDI is not available on all platforms."
                )
            port_indices = []
            for port in comp_cls.ports:
                key = f"{name},{port}"
                if key in port_to_node_map:
                    port_indices.append(port_to_node_map[key])
                else:
                    msg = f"Port '{port}' on '{name}' is unconnected.\nYour netlist connections must include '{key}'"
                    raise ValueError(msg)
            osdi_buckets[comp_type].append(
                {
                    "params_dict": comp_cls.make_instance(settings),
                    "ports": port_indices,
                    "name": name,
                }
            )
            continue

        # GDSFactory netlists carry geometry settings that don't appear on
        # the simulation model (e.g. ``dy``/``dx`` on a ``coupler_strip``
        # instance, or ``allow_min_radius_violation`` on a ``bend_euler``).
        # Filter to fields the model actually declares, and drop ``None``
        # values (GDSFactory convention: ``None`` means "use the default").
        known_fields = {f.name for f in dataclasses.fields(comp_cls)}
        settings = {k: v for k, v in settings.items() if v is not None and k in known_fields}

        # A. Create Equinox Object
        try:
            comp_obj = comp_cls(**settings)
        except TypeError as e:
            msg = f"Settings error for {name}: {e}"
            raise TypeError(msg)  # noqa: B904

        # B. Get Port Indices
        port_indices = []
        for port in comp_cls.ports:
            port_indices.append(_resolve_port_index(comp_cls, name, port))

        # Group by Type AND Structure (to handle static field differences)
        structure = jax.tree.structure(comp_obj)
        buckets[(comp_type, structure)].append(
            {
                "obj": comp_obj,
                "ports": port_indices,
                "num_states": len(comp_cls.states),
                "name": name,
            }
        )

    compiled_groups = {}

    # Helper to generate unique names for split groups
    type_counts = defaultdict(int)
    for ctype, _ in buckets:
        type_counts[ctype] += 1
    type_counters = defaultdict(int)

    for (comp_type, _), items in buckets.items():
        comp_cls = models_map[comp_type]

        # Generate Group Name
        if type_counts[comp_type] > 1:
            idx = type_counters[comp_type]
            group_name = f"{comp_type}_{idx}"
            type_counters[comp_type] += 1
        else:
            group_name = comp_type

        # A. Assign Internal States
        all_var_indices = []
        for item in items:
            state_indices = []
            for s_name in comp_cls.states:
                port_to_node_map[f"{item['name']},{s_name}"] = sys_size
                state_indices.append(sys_size)
                sys_size += 1
            all_var_indices.append(item["ports"] + state_indices)

        # B. Batch Params
        instance_objects = [item["obj"] for item in items]
        batched_params = jax.tree.map(lambda *args: jnp.stack(args), *instance_objects)

        # C. Matrices
        var_indices_arr = jnp.array(all_var_indices, dtype=jnp.int32)
        width = var_indices_arr.shape[1]
        count = len(items)

        jac_rows = jnp.broadcast_to(var_indices_arr[:, :, None], (count, width, width))
        jac_cols = jnp.broadcast_to(var_indices_arr[:, None, :], (count, width, width))

        # Create Index Map for parameter updates
        index_map = {item["name"]: i for i, item in enumerate(items)}

        _combined_func = getattr(comp_cls, "_combined_fn", None) if getattr(comp_cls, "_has_combined_fn", False) else None
        compiled_groups[group_name] = ComponentGroup(
            name=group_name,
            var_indices=var_indices_arr,
            eq_indices=var_indices_arr,
            params=batched_params,
            physics_func=comp_cls.solver_call,
            jac_rows=jac_rows,
            jac_cols=jac_cols,
            index_map=index_map,
            is_fdomain=getattr(comp_cls, "_is_fdomain", False),
            amplitude_param=getattr(comp_cls, "amplitude_param", ""),
            combined_func=_combined_func,
            holomorphic=getattr(comp_cls, "_holomorphic", True),
            has_delay=getattr(comp_cls, "_has_delay", False),
            tau_func=getattr(comp_cls, "tau_of", None) if getattr(comp_cls, "_has_delay", False) else None,
        )

    # --- Process OSDI buckets (requires circulax[verilog-a] / bosdi) ---
    for comp_type, items in osdi_buckets.items():
        descriptor: OsdiModelDescriptor = models_map[comp_type]
        group_name = comp_type
        n_dev = len(items)
        n_pins = descriptor.model.num_pins
        n_nodes = descriptor.model.num_nodes  # terminals + non-collapsed internal
        n_internal = n_nodes - n_pins  # extra unknowns per instance

        params_arr = jnp.array(
            [[item["params_dict"][k] for k in descriptor.param_names] for item in items],
            dtype=jnp.float64,
        )  # (N, num_params)
        states_arr = jnp.zeros((n_dev, descriptor.model.num_states), dtype=jnp.float64)

        # Bosdi Tier-3: pre-bake params into an OsdiBatchHandle so per-Newton-iter
        # OSDI calls skip the params upload (~20–40 % faster for PSP103).
        handle = None
        try:
            import numpy as _np
            from osdi_jax import osdi_setup_batch
            handle = osdi_setup_batch(descriptor.model.id, _np.asarray(params_arr))
        except ImportError:
            pass  # older bosdi without Tier-3; legacy model_id + params path still works

        # Regularisation diagonal: 1.0 only on internal reactive-only nodes.
        # External terminal nodes never need it — the wider circuit KCL provides
        # their equations.  Internal reactive-only nodes have no equation from
        # any other component, so j_eff[i,:] would be zero without this term.
        is_internal = jnp.arange(n_nodes) >= n_pins
        is_reactive = ~jnp.array(descriptor.model.resistive_mask, dtype=bool)
        reg_mask = (is_internal & is_reactive).astype(jnp.float64)
        reg_diag = jnp.diag(reg_mask)  # (num_nodes, num_nodes)

        # Build (N, n_nodes) index array: terminal indices first, then one new
        # state slot per internal node per instance (appended to global state vector).
        all_var_idx_list = []
        for item in items:
            terminal_indices = item["ports"]
            internal_indices = []
            for _i in range(n_internal):
                internal_indices.append(sys_size)
                sys_size += 1
            all_var_idx_list.append(terminal_indices + internal_indices)

        all_var_idx = jnp.array(all_var_idx_list, dtype=jnp.int32)  # (N, n_nodes)

        jac_rows = jnp.broadcast_to(all_var_idx[:, :, None], (n_dev, n_nodes, n_nodes)).reshape(-1)
        jac_cols = jnp.broadcast_to(all_var_idx[:, None, :], (n_dev, n_nodes, n_nodes)).reshape(-1)

        compiled_groups[group_name] = OsdiComponentGroup(
            name=group_name,
            model_id=descriptor.model.id,
            num_pins=n_pins,
            num_nodes=n_nodes,
            num_params=descriptor.model.num_params,
            num_states=descriptor.model.num_states,
            params=params_arr,
            states=states_arr,
            var_indices=all_var_idx,
            eq_indices=all_var_idx,
            jac_rows=jac_rows,
            jac_cols=jac_cols,
            reg_diag=reg_diag,
            index_map={item["name"]: i for i, item in enumerate(items)},
            use_schur_reduction=getattr(descriptor, "use_schur_reduction", False),
            handle=handle,
        )

    return compiled_groups, sys_size, port_to_node_map
