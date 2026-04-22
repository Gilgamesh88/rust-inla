# API Implementation Queue

This queue translates the API-surface inventory into a practical implementation order.

Each item is labeled as:

- `Cheap`
- `High-value`
- `Blocked`

These labels mean:

- `Cheap`: mostly wrapper or bookkeeping work, limited engine risk
- `High-value`: meaningfully improves usability or parity for the current architecture
- `Blocked`: should wait on missing engine/model/workflow support

## 1. Recommended execution order

### Already landed

These cheap object-parity wins are no longer queue items; they are already in
the wrapper:

1. `names.fixed`, `size.random`, `size.linear.predictor`, `nhyper`, `version`, `ok`
   Labels: `Cheap`, `High-value`

2. timing fields such as `cpu.intern`, `cpu.used`
   Labels: `Cheap`, `High-value`

### Queue 1: cheap and immediately useful

These are the best first additions because they improve usability without committing us to a huge API surface.

1. public link helper utilities for the current supported subset
   Labels: `Cheap`, `High-value`
   Why:
   - easy public parity win
   - good foundation for richer family support later

2. public marginal helper utilities built on the existing internal machinery
   Labels: `Medium`, `High-value`
   Why:
   - the core already has strong marginal support
   - users gain a lot from `dmarginal` / `pmarginal` / `qmarginal` style helpers

3. expose a constrained `control.fixed`
   Labels: `Cheap` to `Medium`, `High-value`
   Why:
   - fits current fixed-effect architecture
   - useful immediately

### Queue 2: best next INLA-like controls

These are the next-best public API improvements once Queue 1 is done.

4. constrained `control.compute`
   Labels: `Medium`, `High-value`
   Why:
   - maps naturally to current output profiles
   - helps users ask for exactly what they want

5. constrained `control.mode`
   Labels: `Medium`, `High-value`
   Why:
   - the engine already has warm-start and fixed-theta concepts
   - valuable for diagnostics, exact-theta replays, and future updating hooks

6. constrained `control.predictor`
   Labels: `Medium`, `High-value`
   Why:
   - aligns well with current fitted/linear-predictor outputs

7. internal-scale hyperparameter outputs
   Labels: `Medium`, `High-value`
   Why:
   - closes a real parity gap without adding a new subsystem
   - useful for diagnostics and future prior/update work

8. better `.args` and fit metadata parity
   Labels: `Cheap` to `Medium`, `High-value`
   Why:
   - improves apples-to-apples comparisons
   - easier than adding new inference functionality

### Queue 3: worthwhile, but only after the above

These are real features, but they are not the cheapest wins.

9. `summary.lincomb` and `marginals.lincomb`
   Labels: `High-value`, not `Cheap`
   Why:
   - useful and very INLA-like
   - requires new public linear-combination workflow support

10. `waic`, `dic`, residual surfaces
   Labels: `Medium` to `High`, `High-value`
   Why:
   - important model-assessment outputs
   - need family-consistent definitions and validation

11. `cpo`, `gcpo`, `po`
   Labels: `High`, `High-value`
   Why:
   - very useful in practice
   - needs a predictive-ordinate subsystem and careful validation

12. public graph/Q helper suite
   Labels: `Medium` to `High`, `High-value`
   Why:
   - becomes much more valuable once graph-driven models exist

13. full `marginals.linear.predictor` and `marginals.fitted.values`
   Labels: `High`, conditionally `High-value`
   Why:
   - scientifically useful
   - memory-heavy on large portfolios
    - should probably stay opt-in

## 2. Blocked items

These should not be near-term API goals because they are blocked on missing subsystems.

### Blocked by missing model or workflow support

- `control.group`
- `control.hazard`
- `control.lp.scale`
- `control.mix`
- `control.scopy`
- `control.pom`
- `control.sem`
- SPDE block outputs
- stack helpers
- mesh/SPDE helpers

Labels:

- `Blocked`

Why:

- they depend on model or workflow surfaces that do not exist yet

### Blocked by missing extension-hook architecture

- `rgeneric` parity
- `cgeneric` parity
- `cloglike` parity
- `rprior` parity
- `scopy` definition helpers

Labels:

- `Blocked`

Why:

- these are extension frameworks, not just convenience wrappers

### Blocked by major inference additions

- posterior sampling parity
- `joint.hyper`
- full `misc` parity
- backend control ecosystems like `pardiso`, `taucs`, `numa`

Labels:

- `Blocked`

Why:

- these are large milestones, not incremental API polish

## 3. Best near-term roadmap

If we want the best return for effort, the recommended order is:

1. marginal helper utilities
2. constrained `control.fixed`
3. constrained `control.compute`
4. constrained `control.mode`
5. constrained `control.predictor`
6. internal-scale hyperparameter outputs
7. linear-combination support
8. model-assessment outputs like `waic` / `dic`
9. predictive-ordinate outputs

## 4. Why this order is efficient

This order deliberately:

- uses the current engine before widening architecture
- improves user experience early
- avoids pretending we have SPDE/generic-model ecosystems before we do
- creates a better base for future state-update or prior-update features

## 5. Related documents

- [RINLA_API_SURFACE_INVENTORY.md](RINLA_API_SURFACE_INVENTORY.md)
- [COVERAGE_EVALUATION_2026-04-19.md](COVERAGE_EVALUATION_2026-04-19.md)
- [RINLA_PARITY_GAP_INVENTORY.md](RINLA_PARITY_GAP_INVENTORY.md)
