# circulax — Specs Overview

**Last updated**: 2026-07-28

---

## What this repo does

Circulax is a differentiable, JAX-based circuit simulator for electronic and photonic
circuits. It formulates netlists as DAE systems (F(y) + dQ/dt = 0) and uses automatic
differentiation for gradient-based optimization and inverse design — used by circuit and
photonic-device designers who need to co-optimize topology and physical parameters.

---

## Architecture

Three-layer design, strictly separated:

1. **Physics** (`circulax/components/`) — plain functions decorated with `@component`/`@source`
   define port equations and storage terms (charge/flux) for electronic and photonic parts,
   including OSDI/Verilog-A models (`components/osdi/`, lowered via `circulax/va/`).
2. **Topology** (`circulax/compiler.py`, `circulax/netlist.py`) — compiles a SAX-format netlist
   dict into `ComponentGroup` objects: assigns node IDs, groups instances by type, batches
   parameters, and pre-computes Jacobian sparsity.
3. **Analysis** (`circulax/solvers/`) — assembles and solves the DAE system: DC operating point
   (Newton-Raphson), transient (Diffrax implicit ODE / Backward Euler), and harmonic balance.

```
compile_netlist(net_dict, models) → [ComponentGroup, ...]
    → analyze_circuit(groups) → CircuitLinearSolver
        → solve_dc()
        → setup_transient()
        → setup_harmonic_balance()
```

---

## Spec index

| Spec | Status | What it covers |
|------|--------|----------------|
| [hierarchy.md](hierarchy.md) | **Done** | Hierarchical subcircuit composition — `RecursiveNetlist` support, netlist-level flattening, `Circuit`-as-subcircuit (V1, PR #39) |
| [time-delay.md](time-delay.md) | **Done** | Time delay for AC/HB and transient — `delay_line_fdomain` + `OpticalDelayLine` with DDE history buffer (branch `feat-time-delay`) |
| [vector-fitting.md](vector-fitting.md) | **Done** | S-parameter vector fitting with delay de-embedding — `rational_component`, `rational_fdomain_component`, `rational_delay_component` (branch `feat-time-delay`) |
| [baseband-envelope.md](baseband-envelope.md) | **Planned** | Baseband envelope simulation — carrier-relative spectral effects, sideband asymmetry, CVF/AAA baseband fitting, carrier shift as differentiable parameter |
| Physics — `circulax/components/` | **Planned** | <!-- FILL --> |
| Topology — `circulax/compiler.py`, `circulax/netlist.py` | **Planned** | <!-- FILL --> |
| Analysis — `circulax/solvers/` | **Planned** | <!-- FILL --> |
| VA/OSDI integration — `circulax/va/` | **Planned** | <!-- FILL --> |

Status values: **Planned** · **In progress** · **Done** · **Needs update**

---

## Spec format

Every spec file follows this structure. When writing a new spec, use all sections. When reading a spec to execute, the **delegation map** and **acceptance criteria** tell you what to spawn and how to verify done.

### Goal
One sentence. What needs to be built or changed.

### Context
Why this exists. Constraints and decisions already made. What the execution agent must not second-guess.

### Acceptance criteria
Explicit, verifiable conditions. Run these after implementation to self-verify.
- [ ] Criterion A
- [ ] Criterion B

### Components
Breakdown into independent units. Each is a candidate for delegation to a sub-agent.

| Component | Description | Depends on |
|-----------|-------------|------------|
| A | ... | — |
| B | ... | A |

### Delegation map
Which sub-agent handles which component. Spawned in parallel where dependencies allow.

| Component | Sub-agent role | File scope | Constraints |
|-----------|---------------|------------|--------------|
| A | backend-agent | src/api/user.ts | Do not change function signatures |
| B | test-agent | test/api/user.test.ts | Only add tests, do not modify src |

### Implementation notes
Design decisions, gotchas, patterns to follow. Anything that would otherwise be invented incorrectly by a fresh agent.
