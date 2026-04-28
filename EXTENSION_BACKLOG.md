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

`rw2` and `ar2` are now implemented. Phase 7A fixed-effects
productization is complete for the MVP supported subset; the next pre-release
phase is prior/control metadata design, so custom prior support is not treated
as unfinished Phase 7A work.

## Priority 1: low-friction, high-fit extensions

These are the best next additions because they fit the current abstractions with minimal front-end redesign.

### 1. Phase 7A fixed-effects productization

Status: complete for the MVP supported subset.

Why now:

- arbitrary-width fixed effects already work in the core
- the R-side design-matrix path already exists through `model.matrix()`
- making that support explicit is more valuable to beta users than one more latent topology

Completed work:

- validate rank-deficient or aliased fixed designs explicitly
- add benchmark/reference coverage with multiple fixed columns
- document the supported fixed-effect formula subset
- support fixed-effect-only GLMs through the zero-latent backend path
- add public `rusty_inla()` validation-error tests
- add GLM/MAP comparators for fixed-only GLMs
- add fixed-effect SD gates for the fixed-only and curated supported subsets

Current Phase 7A gate:

- run `tools/run-phase7a-validation.ps1` before treating a fixed-effects
  change as a merge candidate
- fixed-effect SD underestimation is no longer an open generic blocker: the
  Rust core has exact dense-reference covariance tests for Gaussian fixed
  effects with and without latent terms, `tests/fixed-only-parity.R` checks
  fixed-only SDs against R-INLA and the direct Poisson MAP Hessian, and
  `tools/run_supported_subset_validation.R` tracks fixed SD drift across the
  curated supported subset
- keep the next branch-local validation target focused on supported-subset
  multi-latent Poisson coverage; the first case is
  `stress_multi_re_three_iid`, a deterministic proxy for the uploaded stress
  `MultiRE_3Effects` surface

Deferred out of Phase 7A:

- do not expose fixed-prior controls as part of Phase 7A
- do not add general custom priors as part of Phase 7A
- adapt the external comprehensive validation bundle
  (`inla_test_suite_part1.R`, `inla_test_suite_part2_fremtpl2.R`,
  `inla_test_suite_part3_stress.R`, `run_all_benchmarks.R`,
  `inla_complete_test_suite.R`) as a later supported-subset harness expansion
  rather than a Phase 7A completion condition

### 2. Phase 8 prior/control metadata reuse

Status: Phase 8A active and Phase 8B started. See
[PHASE8_ROADMAP.md](PHASE8_ROADMAP.md).

Why next:

- default priors are already embedded in model and likelihood implementations
- custom-prior support should reuse those model definitions instead of adding
  one-off R-side override paths
- constrained `control.fixed` belongs with prior/control metadata, not with
  Phase 7A fixed-design validation

Expected work:

- Phase 8A: stabilize the one-`iid` fixed-plus-`iid` cross evidence state and
  keep benchmark/accuracy gates attached to every commit candidate
- Phase 8B: formalize that reusable update states store old-data evidence, not
  posterior objects blindly reused as priors
- Phase 8C: extract theta-mixture old-data evidence over CCD/theta support;
  first slice stores compact CCD-support evidence in a `theta_evidence`
  container and leaves solver integration for Phase 8D
- Phase 8D: add a theta-dependent old-evidence objective with log constants
- Phase 8E: validate against joint refits on fixed effects, random effects,
  fitted values, theta marginals, tails, time, and memory
- Phase 8F: defer broader latent structures (`rw1`, `rw2`, `ar1`, `ar2`) until
  the one-`iid` path is benchmark-clean
- inventory the existing model, likelihood, and fixed-effect prior metadata
- design a shared prior specification representation for Rust and R bridge
  code
- keep the first posterior-state implementation restricted to one `iid` latent
  block, using a Gaussian approximation to the previous internal
  log-precision posterior as the next fit's prior
- keep the first fixed-plus-`iid` update-state implementation restricted to
  dense fixed Gaussian evidence plus diagonal `iid` Gaussian evidence and the
  fixed-`iid` cross block, with new levels expanded as zero old evidence and
  posterior SD ratios checked against joint refits
- keep the diagonal fixed-plus-`iid` evidence mode as a diagnostic comparator
  only; the cross-block mode is the mergeable Phase 8 target
- define how model defaults are surfaced, overridden, serialized, and validated
- decide the first constrained public surface, likely fixed-effect prior
  controls before arbitrary expression/table priors
- add reference tests that prove default-prior behavior is unchanged when no
  overrides are supplied
- keep expression priors, `rprior`, table priors, and broad R-INLA prior
  registry parity out of scope until the metadata layer is stable

### 3. one additional GLM-like likelihood family

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

### If we add exactly one more thing before the first public release

Recommended next choice:

- Phase 8 prior/control metadata reuse

Why:

- Phase 7A fixed-effects productization is already complete for the MVP
  supported subset
- it keeps custom-prior and prior-control semantics honest before exposing
  public overrides
- it protects current default-prior behavior with explicit metadata and
  reference tests
- it gives the next API expansion a reusable foundation instead of a one-off
  wrapper path

Best second choice after that:

- one additional GLM-like family, with `nbinomial` the strongest candidate if count overdispersion is the most valuable next actuarial case

Practical interpretation:

- if we choose one thing for beta usability, choose Phase 7A
- if we choose one thing for release honesty after Phase 7A, choose Phase 8
  prior/control metadata design
- if we choose one thing for immediate actuarial model breadth after Phase 8,
  choose a GLM-like family

## Priority 2: modest API expansion with strong payoff

These features likely need some widening of the backend spec, but not a core-engine rewrite.

### 4. generic graph input for latent models

Why now:

- the core graph layer already supports generic edge lists
- the current blocker is mainly the public API surface

Expected work:

- extend `f(...)` and the backend spec to accept adjacency or edge lists
- validate indexing, symmetry, and constraints in the bridge
- keep the new input contract simple and explicit

### 5. `besag`-style intrinsic spatial model

Why after generic graph input:

- it becomes much cleaner once graph input exists
- it is a high-value spatial extension without jumping straight to `SPDE`

Expected work:

- add the core `QFunc`
- expose it in the R frontend
- add graph-based reference cases and constraints checks

## Priority 3: richer likelihood surface

These are useful, but they likely require widening the current single-predictor likelihood contract.

### 6. `zeroinflatedpoisson2` or other multi-part count likelihoods

Why later:

- they are not just another scalar-family extension
- they need clearer support for family-specific covariates or multiple linear predictors

Expected work:

- widen the backend spec
- decide how multiple predictors are represented across the binding and R layers
- add careful parity validation

### 7. additional family-specific observation inputs

Examples:

- family-specific weights, trials, or auxiliary inputs that are not currently part of the backend spec

Why later:

- they affect the public interface and validation contract more than the core optimizer

## Priority 4: architecture milestones

These are important, but they should be treated as explicit milestones rather than normal incremental additions.

### 8. `bym2`

Why later:

- coupled latent structure
- scaling conventions
- more complicated validation story

### 9. `SPDE` and mesh workflows

Why later:

- much larger surface area
- mesh-generation and spatial-workflow implications
- far beyond a local `QFunc` addition

### 10. additional language bindings

Examples:

- Python via a thin `PyO3` crate
- Julia via a similarly thin adapter over `inla_core`

Why later:

- the core should stay reusable, but feature expansion in the core is higher value right now than multiplying bindings too early

## Recommended order of execution

If development capacity is limited, the most efficient sequence is:

1. keep the current stable subset benchmark-clean
2. keep Phase 7A fixed-effects productization closed and gated
3. complete Phase 8 prior/control metadata design
4. add one additional GLM-like family
5. widen the backend spec for generic graph input
6. add `besag`
7. revisit richer multi-part families
8. treat `bym2`, `SPDE`, and new language bindings as separate milestones

## Acceptance rule for any backlog item

A backlog item is not done when it merely compiles. It is done when it has:

- core tests
- at least one regression or end-to-end test
- an explicit reference comparison
- updated docs
- a clear statement of whether it is inside the stable benchmark subset
