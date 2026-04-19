# R-INLA API Surface Inventory

This document inventories the public API gap between `rustyINLA` and `R-INLA`, beyond the model registry.

It focuses on:

- `control.*` surfaces
- output-object surfaces
- exported helper functions

It also classifies each area by likely implementation difficulty.

## 1. Current public package surface

### `rustyINLA`

Current exported functions:

- `rusty_inla`
- `rust_inla_run`
- `f`

Current S3 methods:

- `print.rusty_inla`
- `summary.rusty_inla`
- `predict.rusty_inla`

Current `rusty_inla()` user-facing arguments:

- `formula`
- `data`
- `family`
- `offset`
- `output_profile`

### `R-INLA`

Local export snapshot:

- total namespace exports: `410`
- `control.*` constructors: `24`
- `inla.set.control.*.default` helpers: `23`
- link helpers: `35`
- marginal helpers: `15`
- mesh/SPDE helpers: `46`
- graph/Q helpers: `17`
- generic model-definition helpers: `13`
- stack helpers: `12`
- posterior helpers: `4`

This immediately shows the difference in product surface:

- `rustyINLA` exposes a small, focused fitting API
- `R-INLA` exposes a much larger modeling, workflow, and utility ecosystem

## 2. Difficulty classes

This inventory uses four difficulty classes:

- `Low`
  Mostly wrapper or bookkeeping work. Usually does not require new inference math.
- `Medium`
  Needs meaningful work in the existing core or result-building path, but fits the current architecture.
- `High`
  Needs a new subsystem, broader API redesign, or substantial new validation work.
- `Very high`
  Large workflow surface or architecture milestone, not just an additive feature.

## 3. `control.*` surface inventory

### A. Controls that are conceptually closest to the current architecture

| Surface | Current status | Likely difficulty | Why |
| --- | --- | --- | --- |
| `control.compute` | partial concept only | `Medium` | some compute toggles map naturally to current output profiles, but R-INLA breadth is much wider |
| `control.family` | missing public API | `Medium` | fits existing family path for supported families, but would need public prior/link/input plumbing |
| `control.fixed` | missing public API | `Low` to `Medium` | closest to existing fixed-effect path |
| `control.mode` | missing public API | `Medium` | current engine already has mode and warm-start concepts |
| `control.predictor` | missing public API | `Medium` | partly overlaps with current fitted/linear-predictor output controls |
| `control.inla` | missing public API | `Medium` to `High` | some internals exist, but broad parity is much larger than the current engine surface |
| `control.link` | missing public API | `Medium` | some link behavior already exists internally, but public parity is broader |

### B. Controls that depend on missing model or wrapper surfaces

| Surface | Current status | Likely difficulty | Why |
| --- | --- | --- | --- |
| `control.group` | missing | `High` | depends on missing group-model surface |
| `control.hazard` | missing | `High` | depends on missing hazard/survival surface |
| `control.lp.scale` | missing | `High` | depends on missing `lp.scale` surface and related model support |
| `control.mix` | missing | `High` | depends on missing mixture surface |
| `control.scopy` | missing | `High` | depends on missing `scopy` support |
| `control.pom` | missing | `High` | depends on missing proportional-odds model family |
| `control.sem` | missing | `High` | depends on missing SEM likelihood surface |
| `control.lincomb` | missing | `High` | needs public linear-combination workflow plus output support |
| `control.update` | missing | `High` | meaningful parity depends on broader update/rerun workflow support |
| `control.vb` | missing | `Very high` | variational-Bayes-specific surface is outside the current engine direction |

### C. Controls tied to backend or platform ecosystems

| Surface | Current status | Likely difficulty | Why |
| --- | --- | --- | --- |
| `control.pardiso` | missing | `Very high` | backend integration milestone |
| `control.taucs` | missing | `Very high` | backend integration milestone |
| `control.numa` | missing | `Very high` | system-level backend tuning surface |
| `control.stiles` | missing | `Very high` | specialized subsystem not represented in the current architecture |
| `control.expert` | missing | `Very high` | broad expert override surface, not just one feature |
| `control.gcpo` | missing | `High` | depends on predictive-ordinate outputs and related diagnostics |
| `control.bgev` | missing | `High` | tied to missing BGEV likelihood |

### `control.*` overall read

The easiest meaningful wins are:

- `control.fixed`
- a constrained subset of `control.compute`
- a constrained subset of `control.mode`
- a constrained subset of `control.predictor`

Those are the best candidates if we ever want a more INLA-like call surface without overcommitting.

## 4. Output-object inventory

### A. Output surfaces already present in `rustyINLA`

Always present in current fits:

- `summary.fixed`
- `summary.random`
- `summary.hyperpar`
- `summary.fitted.values`
- `mlik`
- `mode`
- diagnostics-related fields

Present in `output_profile = "benchmark"`:

- `summary.linear.predictor`
- `marginals.fixed`
- `marginals.random`
- `marginals.hyperpar`
- `model.matrix`
- light `.args`

### B. Output surfaces missing in `thin` but mostly present in `benchmark`

| Output | Current status | Likely difficulty |
| --- | --- | --- |
| `summary.linear.predictor` | available in `benchmark` only | `Low` |
| `marginals.fixed` | available in `benchmark` only | `Low` |
| `marginals.random` | available in `benchmark` only | `Low` |
| `marginals.hyperpar` | available in `benchmark` only | `Low` |
| `.args` | light version only | `Low` to `Medium` |
| `model.matrix` | available in `benchmark` only | `Low` |

These are not really missing from the engine. They are mostly profile-policy decisions.

### C. Output surfaces that fit the current architecture with more work

| Output | Current status | Likely difficulty | Why |
| --- | --- | --- | --- |
| `names.fixed` | missing | `Low` | bookkeeping around current fixed-effect outputs |
| `size.random` | missing | `Low` | straightforward metadata |
| `size.linear.predictor` | missing | `Low` | straightforward metadata |
| `nhyper` | missing | `Low` | straightforward metadata |
| `version` | missing | `Low` | bookkeeping |
| `ok` | missing | `Low` | bookkeeping |
| `cpu.intern`, `cpu.used` | missing | `Low` to `Medium` | timing bookkeeping |
| `internal.summary.hyperpar` | missing | `Medium` | needs internal-scale hyperparameter summaries |
| `internal.marginals.hyperpar` | missing | `Medium` | same |
| `all.hyper` | missing | `Medium` | needs clearer hyperparameter bookkeeping/export |
| `model.random` | missing | `Medium` | richer random-component metadata surface |
| `offset.linear.predictor` | missing | `Medium` | requires cleaner predictor bookkeeping |

### D. Output surfaces that need real feature additions

| Output | Current status | Likely difficulty | Why |
| --- | --- | --- | --- |
| `summary.lincomb` / `marginals.lincomb` | missing | `High` | requires public linear-combination feature support |
| `summary.lincomb.derived` / `marginals.lincomb.derived` | missing | `High` | same plus derived-workflow semantics |
| `marginals.linear.predictor` | missing | `High` | memory-heavy observation-level marginals |
| `marginals.fitted.values` | missing | `High` | same memory wall on large portfolios |
| `joint.hyper` | missing | `High` | needs joint hyperparameter surface and export story |
| `Q` | missing | `High` | exportable precision object design, memory implications |
| `graph` | missing | `High` | public graph export design and stable representation |
| `logfile` | missing | `Medium` to `High` | logging design more than math, but still product-surface work |
| `misc` | missing | `Very high` | this is a large heterogeneous surface in R-INLA |

### E. Output surfaces that depend on missing inference/diagnostic subsystems

| Output | Current status | Likely difficulty | Why |
| --- | --- | --- | --- |
| `cpo`, `gcpo`, `po` | missing | `High` | predictive-ordinate subsystem not currently present |
| `waic` | missing | `High` | needs broader model-criterion support |
| `dic` | missing | `High` | same |
| `residuals` | missing | `Medium` to `High` | depends on residual-definition policy across families |
| `posterior.sample`-style outputs | missing | `Very high` | no posterior sampling parity yet |

### F. Output surfaces blocked by missing model families

| Output | Current status | Likely difficulty | Why |
| --- | --- | --- | --- |
| `model.spde2.blc`, `summary.spde2.blc`, `marginals.spde2.blc`, `size.spde2.blc` | missing | `Very high` | blocked by missing SPDE2 support |
| `model.spde3.blc`, `summary.spde3.blc`, `marginals.spde3.blc`, `size.spde3.blc` | missing | `Very high` | blocked by missing SPDE3 support |

## 5. Exported helper inventory

### A. Link helpers

R-INLA exposes many link and inverse-link helpers, such as:

- `inla.link.log`
- `inla.link.logit`
- `inla.link.probit`
- inverse forms like `inla.link.invlogit`

Current `rustyINLA` status:

- no public helper family of this kind

Likely difficulty:

- `Low` for basic forward/inverse helpers
- `Medium` if the goal is full parity with all named link variants

### B. Marginal utilities

R-INLA exposes a strong marginal-manipulation toolbox:

- `inla.dmarginal`
- `inla.pmarginal`
- `inla.qmarginal`
- `inla.tmarginal`
- `inla.emarginal`
- and others

Current `rustyINLA` status:

- internal marginal machinery exists
- no comparable public R helper toolbox is exported

Likely difficulty:

- `Medium`

This is a good example of an area where the core already has useful building blocks, but the public helper layer is still thin.

### C. Graph and precision-matrix helpers

Examples:

- `inla.read.graph`
- `inla.write.graph`
- `inla.graph2matrix`
- `inla.matrix2graph`
- `inla.qsolve`
- `inla.qinv`
- `inla.qsample`

Current `rustyINLA` status:

- internal graph and precision logic exists
- no public helper suite is exported

Likely difficulty:

- `Medium` for read/write/convert helpers
- `High` for a full public `Q` utility layer with stable object semantics

### D. Mesh/SPDE helpers

Examples:

- `inla.mesh.*`
- `inla.spde.*`
- `inla.fmesher.smorg`

Current `rustyINLA` status:

- absent

Likely difficulty:

- `Very high`

This is a full workflow ecosystem, not just a missing helper function family.

### E. Generic model and likelihood helpers

Examples:

- `inla.rgeneric.define`
- `inla.cgeneric.define`
- `inla.cloglike.define`
- `inla.rprior.define`
- `inla.scopy.define`

Current `rustyINLA` status:

- absent as public surfaces

Likely difficulty:

- `High` to `Very high`

These are architecture-defining extension hooks.

### F. Stack helpers

Examples:

- `inla.stack`
- `inla.stack.data`
- `inla.stack.A`
- `inla.stack.index`

Current `rustyINLA` status:

- absent

Likely difficulty:

- `High`

This is a broader modeling workflow surface, especially relevant once graph-driven and SPDE workflows exist.

### G. Posterior helpers

Examples:

- `inla.hyperpar`
- `inla.hyperpar.sample`
- `inla.posterior.sample`
- `inla.posterior.sample.eval`

Current `rustyINLA` status:

- no public parity

Likely difficulty:

- `High` to `Very high`

## 6. Best next public-API wins

If the goal is to improve public parity efficiently, the best next wins are probably:

1. a constrained subset of `control.fixed`, `control.compute`, `control.mode`, and `control.predictor`
2. public marginal helper utilities built on the current internal marginal machinery
3. additional metadata outputs such as `names.fixed`, `size.*`, `nhyper`, and timing fields

These are much cheaper than:

- SPDE helper ecosystems
- generic model-definition hooks
- posterior sampling parity
- full `misc` parity

## 7. Honest overall read

At the public API level, `rustyINLA` is still much earlier than the core engine.

That is not a problem if we describe it honestly:

- engine maturity for the supported subset is already meaningful
- public API breadth is still intentionally narrow
- many missing API surfaces are packaging and workflow systems, not core inference failures

## 8. Related documents

- [API_IMPLEMENTATION_QUEUE.md](API_IMPLEMENTATION_QUEUE.md)
- [COVERAGE_EVALUATION_2026-04-19.md](COVERAGE_EVALUATION_2026-04-19.md)
- [RINLA_PARITY_GAP_INVENTORY.md](RINLA_PARITY_GAP_INVENTORY.md)
- [EXTENSION_INTERVENTION_MAP.md](EXTENSION_INTERVENTION_MAP.md)
