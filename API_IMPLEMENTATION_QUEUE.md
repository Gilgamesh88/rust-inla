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

### Phase 8: prior/control metadata reuse

These items should be treated as the next prior/control phase, not as leftover
Phase 7A fixed-effects productization. They are cheap only after the shared
metadata layer exists.

3. design shared prior/control metadata for defaults and overrides
   Labels: `Medium`, `High-value`
   Why:
   - keeps model defaults and public override semantics in one place
   - prevents one-off R-side custom-prior paths from becoming the public API

4. validate the experimental one-`iid` posterior-state update path
   Labels: `Medium`, `High-value`
   Why:
   - gives the Bayesian updating idea a narrow measurable surface
   - reuses the previous `iid` log-precision posterior as a Gaussian prior on
     the internal theta scale
   - adds a stricter reusable `rusty_update_state()` experiment for dense
     fixed-effect evidence, diagonal `iid` evidence, and the fixed-`iid` cross
     block
   - keeps broader joint-state reuse and full hyperparameter mixture reuse out
     of the first implementation, with SD ratios checked against joint refits

5. expose a constrained `control.fixed` on top of that metadata layer
   Labels: `Cheap` to `Medium`, `High-value`
   Why:
   - fits the current fixed-effect architecture once prior metadata is explicit
   - gives users the first controlled prior-override surface without committing
     to arbitrary expression/table prior parity

### Queue 2: best next INLA-like controls

These are the next-best public API improvements once Queue 1 and the Phase 8
prior/control metadata decision are done.

6. constrained `control.compute`
   Labels: `Medium`, `High-value`
   Why:
   - maps naturally to current output profiles
   - helps users ask for exactly what they want

7. constrained `control.mode`
   Labels: `Medium`, `High-value`
   Why:
   - the engine already has warm-start and fixed-theta concepts
   - valuable for diagnostics, exact-theta replays, and future updating hooks

8. constrained `control.predictor`
   Labels: `Medium`, `High-value`
   Why:
   - aligns well with current fitted/linear-predictor outputs

9. internal-scale hyperparameter outputs
   Labels: `Medium`, `High-value`
   Why:
   - closes a real parity gap without adding a new subsystem
   - useful for diagnostics and future prior/update work

10. better `.args` and fit metadata parity
   Labels: `Cheap` to `Medium`, `High-value`
   Why:
   - improves apples-to-apples comparisons
   - easier than adding new inference functionality

### Queue 3: worthwhile, but only after the above

These are real features, but they are not the cheapest wins.

11. `summary.lincomb` and `marginals.lincomb`
   Labels: `High-value`, not `Cheap`
   Why:
   - useful and very INLA-like
   - requires new public linear-combination workflow support

12. `waic`, `dic`, residual surfaces
   Labels: `Medium` to `High`, `High-value`
   Why:
   - important model-assessment outputs
   - need family-consistent definitions and validation

13. `cpo`, `gcpo`, `po`
   Labels: `High`, `High-value`
   Why:
   - very useful in practice
   - needs a predictive-ordinate subsystem and careful validation

14. public graph/Q helper suite
   Labels: `Medium` to `High`, `High-value`
   Why:
   - becomes much more valuable once graph-driven models exist

15. full `marginals.linear.predictor` and `marginals.fitted.values`
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

1. Phase 8 prior/control metadata design
2. experimental one-`iid` posterior-state update validation
3. constrained `control.fixed` on that metadata layer
4. marginal helper utilities
5. constrained `control.compute`
6. constrained `control.mode`
7. constrained `control.predictor`
8. internal-scale hyperparameter outputs
9. linear-combination support
10. model-assessment outputs like `waic` / `dic`
11. predictive-ordinate outputs

## 4. Why this order is efficient

This order deliberately:

- uses the current engine before widening architecture
- improves user experience early
- avoids pretending we have SPDE/generic-model ecosystems before we do
- creates a better base for future state-update or prior-update features
- keeps custom-prior support separate from the completed Phase 7A fixed-effects
  contract

## 5. Related documents

- [RINLA_API_SURFACE_INVENTORY.md](RINLA_API_SURFACE_INVENTORY.md)
- [COVERAGE_EVALUATION_2026-04-19.md](COVERAGE_EVALUATION_2026-04-19.md)
- [RINLA_PARITY_GAP_INVENTORY.md](RINLA_PARITY_GAP_INVENTORY.md)
