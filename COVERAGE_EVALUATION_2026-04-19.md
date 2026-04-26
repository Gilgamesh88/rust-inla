# Coverage Evaluation

This note evaluates current `rustyINLA` coverage in two separate ways:

1. solver and inference-core coverage
2. family and latent-model implementation coverage

Those two dimensions are very different.

- The solver and inference core is already substantial.
- The registry-level family and model surface is still small relative to full R-INLA.

That means the engine is much farther along than the feature-count percentages alone would suggest.

## 1. Executive summary

### Solver and inference core

Best current estimate:

- about `60-65%` of the broad R-INLA solver/inference ecosystem by major subsystem
- about `85-90%` of the solver/inference machinery needed for the currently supported stable benchmark subset

Why the range:

- R-INLA does not expose a clean "solver registry" comparable to `inla.list.models()`
- some solver-like surfaces are clearly implemented in `rustyINLA`
- others are only partially represented
- some are really separate workflow systems rather than a single solver

### Family and latent-model surface

Registry-level coverage against the local `INLA 25.10.19` install:

- likelihood families: `5 / 105 = 4.8%`
- latent models: `3 / 52 = 5.8%`
- combined headline model registry: `8 / 157 = 5.1%`

Important nuance:

- registry coverage is still low
- benchmark-relevant coverage for the current actuarial MVP subset is much higher
- the active stable suite is currently `5/5` pass with Tweedie intentionally excluded from MVP

## 2. Solver and inference-core coverage

This section scores major solver and inference subsystems using:

- `implemented`
- `partial`
- `missing`

The percentages here are judgment-based, but grounded in the current codebase and validation work.

| Major subsystem | Current status | Notes |
| --- | --- | --- |
| sparse graph representation | implemented | `graph/` supports diagonal, chain, disjoint union, and generic neighbor graphs |
| sparse precision evaluation | implemented | current `QFunc` path is solid for supported models |
| sparse factorization and log-determinant path | implemented | core solve path exists and is benchmarked on supported subset |
| selected inverse / covariance extraction | implemented | used in current uncertainty path |
| latent mode solving with fixed effects | implemented | central to current benchmarked subset |
| constraint-aware intrinsic-field handling | implemented | recent `rw1` covariance fix closed a real parity bug |
| outer hyperparameter optimization | implemented | stable on current supported benchmark subset |
| CCD hyperparameter integration | implemented | current outer integration path exists and is validated for the supported subset |
| posterior summaries on current subset | partial | means, SDs, quantiles, and some marginals exist, but not the full R-INLA output surface |
| fitted-value and linear-predictor summaries | partial | solid for supported cases, but not full observation-level marginal parity |
| posterior sampling / joint posterior utilities | missing | no parity with `inla.posterior.sample` or similar surfaces |
| generic latent/likelihood plugin machinery | missing | no public `rgeneric`, `cgeneric`, or `cloglike` parity |
| group / copy / joint wrapper subsystems | missing | no `group`, `copy`, `scopy`, or `joint` parity |
| mesh / SPDE workflow stack | missing | no mesh or SPDE workflow parity yet |
| advanced solver-backend ecosystem | missing | no broad parity for `pardiso`, `taucs`, `numa`, reordering-control surfaces |

### Coverage reading

If you score:

- `implemented = 1`
- `partial = 0.5`
- `missing = 0`

then this table gives a rough solver/inference score of about:

- `9 / 15 = 60%`

That is the best single-number estimate for broad solver/inference ecosystem coverage.

### Why the current MVP feels farther along than `60%`

Because the missing `40%` is concentrated in very large feature surfaces:

- SPDE and mesh workflows
- generic user-defined models
- posterior sampling and joint utilities
- copy/group/joint wrapper surfaces
- advanced backend-control parity

Those matter a lot for full R-INLA parity, but they are not required for the current supported actuarial benchmark subset.

For the supported benchmark subset, the relevant solver stack is much closer to complete:

- sparse graph and precision logic
- latent mode solving
- fixed effects
- hyperparameter optimization
- CCD integration
- covariance extraction
- fitted summaries

That is why the practical subset coverage is closer to the `85-90%` range.

## 3. Family and latent-model coverage

This section is exact rather than approximate because the local R-INLA install exposes a model registry.

Snapshot source:

- local package: `INLA 25.10.19`
- query method: `capture.output(INLA::inla.list.models())`

### Likelihood families

Current `rustyINLA` implementations:

- `gaussian`
- `poisson`
- `gamma`
- `zeroinflatedpoisson1`
- `tweedie`

Coverage:

- `5 / 105 = 4.8%`

### Latent models

Current `rustyINLA` implementations:

- `iid`
- `rw1`
- `ar1`

Coverage:

- `3 / 52 = 5.8%`

### Combined registry headline

If we combine the two main registry counts:

- `8 / 157 = 5.1%`

This is a useful headline, but it understates real progress because all models are not equally central to the target use case.

## 4. What the percentages miss

Raw registry percentages do not capture:

- current benchmark pass rate on the active MVP subset
- how much of the hard solver work is already done
- the fact that one strong engine can support many later families and latent models

So the honest interpretation is:

- solver core: already strong for the supported subset
- registry breadth: still early relative to full R-INLA

## 5. Practical interpretation for project status

If the question is:

"How much of full R-INLA do we cover by count?"

Then the answer is:

- only a small fraction of the family and latent-model registry

If the question is:

"How much of the hard numerical core do we cover for the current actuarial MVP subset?"

Then the answer is:

- most of it

That is why the current state can honestly be described as:

- low registry breadth
- high progress on the intended MVP engine

## 6. Recommended next reading

For the detailed missing-model inventory, see [RINLA_PARITY_GAP_INVENTORY.md](RINLA_PARITY_GAP_INVENTORY.md).

For the public API-surface gap inventory, including `control.*`, output objects, and exported helpers with difficulty ratings, see [RINLA_API_SURFACE_INVENTORY.md](RINLA_API_SURFACE_INVENTORY.md).
