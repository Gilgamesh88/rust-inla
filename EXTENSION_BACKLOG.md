# Extension Backlog

This backlog is the recommended development order for extending `rustyINLA` beyond the current stable subset.

The ordering is based on:

- implementation efficiency
- fit with the current architecture
- benchmark and scientific value
- how much API expansion is required

For the subsystem inventory behind these priorities, see [IMPLEMENTATION_INVENTORY_AND_EXTENSION_GUIDE.md](IMPLEMENTATION_INVENTORY_AND_EXTENSION_GUIDE.md). For a quick directory-level touch map, see [EXTENSION_INTERVENTION_MAP.md](EXTENSION_INTERVENTION_MAP.md).

## Priority 0: keep the current subset boringly reliable

These items come before any major feature expansion.

- keep the active benchmark suite stable and reproducible
- keep the external reference suite stable and reproducible
- save and maintain golden reference outputs for the supported subset
- keep workspace checks and Windows R build setup documented and working
- continue narrowing remaining performance gaps on supported models

Why first:

- a stable base makes every later extension cheaper to validate
- parity regressions are easier to catch when the supported subset is already frozen

`rw2` and `ar2` are now implemented. The next highest-value pre-release work is the first fixed-effects productization slice from Phase 7.

## Priority 1: low-friction, high-fit extensions

These are the best next additions because they fit the current abstractions with minimal front-end redesign.

### 1. Phase 7A fixed-effects productization

Why now:

- arbitrary-width fixed effects already work in the core
- the R-side design-matrix path already exists through `model.matrix()`
- making that support explicit is more valuable to beta users than one more latent topology

Expected work:

- validate rank-deficient or aliased fixed designs explicitly
- add benchmark/reference coverage with multiple fixed columns
- document the supported fixed-effect formula subset
- decide whether any fixed-prior controls should be exposed before release
- fix the current fixed-effect SD underestimation before treating Phase 7A as complete
- at the end of Phase 7A, adapt and run the external comprehensive validation bundle
  (`inla_test_suite_part1.R`, `inla_test_suite_part2_fremtpl2.R`,
  `inla_test_suite_part3_stress.R`, `run_all_benchmarks.R`,
  `inla_complete_test_suite.R`) against the supported subset instead of
  running it mid-phase

Current Phase 7A gate:

- run `tools/run-phase7a-validation.ps1` before treating a fixed-effects
  change as a merge candidate
- keep the next branch-local validation target focused on supported-subset
  multi-latent Poisson coverage; the first case is
  `stress_multi_re_three_iid`, a deterministic proxy for the uploaded stress
  `MultiRE_3Effects` surface

### 2. one additional GLM-like likelihood family

Candidate examples:

- negative binomial, if count overdispersion is the most valuable next actuarial case
- binomial, if generalized-link validation breadth is more important

Why now:

- the likelihood trait is already a clean extension point
- this usually requires less API redesign than a new spatial GMRF family

Expected work:

- implement the likelihood struct
- register its string name and defaults in the bridge
- add unit tests, end-to-end tests, and reference comparisons

### If we add exactly one thing before the first public release

Recommended choice:

- Phase 7A fixed-effects productization

Why:

- it is the lowest-risk extension that still proves the architecture can grow
- it reuses the existing chain-graph and latent-model machinery
- it expands the smoothing subset naturally
- it is easier to validate cleanly than a new family with richer observation semantics
- it gives us a stronger "the engine generalizes beyond the initial three latent models" story

Best second choice after that:

- one additional GLM-like family, with `nbinomial` the strongest candidate if count overdispersion is the most valuable next actuarial case

Practical interpretation:

- if we choose one thing for beta usability, choose Phase 7A
- if we choose one thing for immediate actuarial model breadth after that, choose a GLM-like family

## Priority 2: modest API expansion with strong payoff

These features likely need some widening of the backend spec, but not a core-engine rewrite.

### 3. generic graph input for latent models

Why now:

- the core graph layer already supports generic edge lists
- the current blocker is mainly the public API surface

Expected work:

- extend `f(...)` and the backend spec to accept adjacency or edge lists
- validate indexing, symmetry, and constraints in the bridge
- keep the new input contract simple and explicit

### 4. `besag`-style intrinsic spatial model

Why after generic graph input:

- it becomes much cleaner once graph input exists
- it is a high-value spatial extension without jumping straight to `SPDE`

Expected work:

- add the core `QFunc`
- expose it in the R frontend
- add graph-based reference cases and constraints checks

## Priority 3: richer likelihood surface

These are useful, but they likely require widening the current single-predictor likelihood contract.

### 5. `zeroinflatedpoisson2` or other multi-part count likelihoods

Why later:

- they are not just another scalar-family extension
- they need clearer support for family-specific covariates or multiple linear predictors

Expected work:

- widen the backend spec
- decide how multiple predictors are represented across the binding and R layers
- add careful parity validation

### 6. additional family-specific observation inputs

Examples:

- family-specific weights, trials, or auxiliary inputs that are not currently part of the backend spec

Why later:

- they affect the public interface and validation contract more than the core optimizer

## Priority 4: architecture milestones

These are important, but they should be treated as explicit milestones rather than normal incremental additions.

### 7. `bym2`

Why later:

- coupled latent structure
- scaling conventions
- more complicated validation story

### 8. `SPDE` and mesh workflows

Why later:

- much larger surface area
- mesh-generation and spatial-workflow implications
- far beyond a local `QFunc` addition

### 9. additional language bindings

Examples:

- Python via a thin `PyO3` crate
- Julia via a similarly thin adapter over `inla_core`

Why later:

- the core should stay reusable, but feature expansion in the core is higher value right now than multiplying bindings too early

## Recommended order of execution

If development capacity is limited, the most efficient sequence is:

1. keep the current stable subset benchmark-clean
2. finish Phase 7A fixed-effects productization
3. add one additional GLM-like family
4. widen the backend spec for generic graph input
5. add `besag`
6. revisit richer multi-part families
7. treat `bym2`, `SPDE`, and new language bindings as separate milestones

## Acceptance rule for any backlog item

A backlog item is not done when it merely compiles. It is done when it has:

- core tests
- at least one regression or end-to-end test
- an explicit reference comparison
- updated docs
- a clear statement of whether it is inside the stable benchmark subset
