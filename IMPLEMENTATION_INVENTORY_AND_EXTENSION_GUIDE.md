# Implementation Inventory and Extension Guide

This document inventories the currently implemented `rustyINLA` subset and turns that inventory into a practical path for future development.

It is not a line-by-line map of every INLA C symbol. It is a subsystem-level inventory of what has already been reverse engineered into Rust, what is still missing, and how expensive common extension paths are likely to be.

For a directory-level diagram showing where to intervene for future additions, see [EXTENSION_INTERVENTION_MAP.md](EXTENSION_INTERVENTION_MAP.md). For a detailed snapshot of what is still missing relative to R-INLA, see [RINLA_PARITY_GAP_INVENTORY.md](RINLA_PARITY_GAP_INVENTORY.md).
For a worked latent-model extension template using `ar2` as the example, see
[AR2_EXTENSION_EXAMPLE.md](AR2_EXTENSION_EXAMPLE.md).

## 1. Why this inventory exists

This repo has reached the point where future development should be guided by architecture, not by memory.

This inventory is meant to answer:

- what parts of the INLA-style engine already exist in Rust
- what parts are currently exposed through the R package
- what kinds of additions fit the current abstractions cleanly
- what kinds of additions require API or architecture expansion

## 2. Current implemented subset

### Likelihood families

Implemented in [src/rust/inla_core/src/likelihood/mod.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/likelihood/mod.rs):

| Family string | Rust type | Hyperparameters | Status |
| --- | --- | --- | --- |
| `gaussian` | `GaussianLikelihood` | `1` | implemented |
| `poisson` | `PoissonLikelihood` | `0` | implemented |
| `gamma` | `GammaLikelihood` | `1` | implemented |
| `zeroinflatedpoisson1` | `ZipLikelihood` | `1` | implemented |
| `tweedie` | `TweedieLikelihood` | `2` | implemented but currently outside the stable MVP benchmark suite |

### Latent GMRF models

Implemented in [src/rust/inla_core/src/models/mod.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/models/mod.rs):

| Model string | Rust type | Hyperparameters | Graph pattern | Status |
| --- | --- | --- | --- | --- |
| `iid` | `IidModel` | `1` | diagonal | implemented |
| `rw1` | `Rw1Model` | `1` | linear chain | implemented |
| `rw2` | `Rw2Model` | `1` | second-order chain | implemented |
| `ar1` | `Ar1Model` | `2` | linear chain | implemented |
| `ar2` | `Ar2Model` | `3` | second-order chain | implemented |
| compound block-diagonal combinations | `CompoundQFunc` | sum of block parameters | disjoint union of block graphs | implemented |

For `rw2`, the current bridge now passes numeric latent level values into the Rust core when they are available. That means the precision and intrinsic linear-trend constraint follow the actual coordinate spacing for irregular grids, which is the INLA-style behavior needed by examples such as `SemiPar::lidar`.
For `ar2`, the current implementation uses INLA-style PACF hyperparameters internally and is benchmarked against `R-INLA` `model = "ar", order = 2` in the external reference harness.

### Core engine subsystems

| Subsystem | Main location | Current status |
| --- | --- | --- |
| sparse graph representation | [graph/mod.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/graph/mod.rs) | implemented |
| sparse solver and selected inverse | [solver/](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/solver) | implemented |
| latent mode and Schur fixed-effects solves | [problem/mod.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/problem/mod.rs) | implemented |
| Laplace objective and outer optimization | [optimizer/mod.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/optimizer/mod.rs) | implemented |
| CCD grid construction and weighting | [optimizer/ccd.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/optimizer/ccd.rs) | implemented |
| posterior summaries and fitted marginals | [inference/mod.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/inference/mod.rs) | implemented for current subset |
| marginal utilities | [marginal/](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/marginal) | implemented |
| diagnostics and profiling hooks | [diagnostics/](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/diagnostics) | implemented |

### Interface layers

| Layer | Main location | Current role |
| --- | --- | --- |
| reusable statistical core | [src/rust/inla_core/](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core) | canonical inference engine |
| thin R binding crate | [src/rust/src/lib.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/src/lib.rs) | name-based registration, input validation, result export |
| R helper for latent terms | [R/f.R](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/R/f.R) | supported `f(..., model=...)` interface |
| R formula and backend spec builder | [R/interface.R](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/R/interface.R) | formula parsing, matrix building, output shaping |

## 3. Reverse-engineered subset by responsibility

### Graph and sparsity

The current graph layer already supports:

- diagonal graphs for `iid`
- linear-chain graphs for `rw1` and `ar1`
- second-order chain graphs for `rw2` and `ar2`
- generic edge-list graphs through [`Graph::from_neighbors`](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/graph/mod.rs:119)
- block-diagonal unions through [`Graph::disjoint_union`](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/graph/mod.rs:146)
- extra fill induced by `A^T A` structure through [`Graph::build_a_t_a_edges`](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/graph/mod.rs:164)

This is an important milestone: the core is no longer limited to hard-coded chain models. The current limitation is mostly the front-end input contract, not the graph representation itself.

### Likelihood layer

The likelihood trait already captures the main pieces needed for many new families:

- pointwise log-likelihood evaluation
- link function
- number of family-specific hyperparameters
- analytic gradient and curvature with respect to the linear predictor
- family-specific hyperprior on the internal theta scale

That makes likelihood extension relatively local when the family fits the existing `y`, `eta`, `theta` contract.

### Latent model layer

The `QFunc` trait already supports:

- sparse graph exposure
- elementwise precision evaluation
- optional analytic derivatives with respect to theta
- proper versus improper priors
- model-specific hyperpriors

In the current subset, `rw2` is the first model that uses extra structural metadata beyond block length alone: when the R side provides numeric latent values, [Rw2Model](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/models/mod.rs) builds the second-order precision on those coordinates instead of assuming unit spacing.
The current proper-chain subset now also includes [Ar2Model](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/models/mod.rs), which reuses the same second-order sparsity pattern but swaps in the stationary AR2 precision induced by unconstrained PACF hyperparameters.

That is enough for many additional latent models, provided they can be expressed as a sparse precision matrix over the current latent-vector representation.

### Inference layer

The inference engine already supports:

- latent mode finding with fixed effects
- constraint-aware covariance summaries for intrinsic models
- outer theta optimization
- CCD-based theta mixing
- fitted summaries on the response scale
- export of theta support and weights for downstream diagnostics

This means many future additions can reuse the same end-to-end inference path instead of needing new optimizers or new result objects.

## 4. What is still structurally missing

The main missing pieces are not in the sparse solver or the optimizer. They are in the feature surface.

### Missing or limited family surface

- no current support for likelihoods that need extra per-observation inputs beyond the current backend spec
- no current support for multi-part likelihood structures such as `zeroinflatedpoisson2`
- no current support for family-specific covariates outside the single linear predictor path

### Missing or limited latent-model surface

- no current user-facing way to pass a generic adjacency graph into `f(...)`
- no current support for graph-based spatial models such as `besag`
- no current support for coupled or scaled latent constructions such as `bym2`
- no `SPDE` or mesh workflow

### Missing or limited binding surface

The R-side latent-model helper in [R/f.R](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/R/f.R) currently accepts only:

- `iid`
- `rw1`
- `rw2`
- `ar1`
- `ar2`

The binding registration in [src/rust/src/lib.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/src/lib.rs:155) now mirrors that same five-model subset.

So the next latent-model bottleneck is no longer `rw2` or `ar2`; it is the lack of a public graph-input contract for models such as `besag`.

## 5. Extension difficulty classes

### Class A: new family that fits the current input contract

Examples:

- another GLM-like family with one linear predictor
- a family with analytic gradient and curvature in `eta`

Typical work:

- add a new likelihood struct in `likelihood/mod.rs`
- register its string name in `src/rust/src/lib.rs`
- add default `theta_init`
- add core tests and at least one reference comparison

Expected difficulty:

- easy to medium

This is usually the cheapest meaningful expansion path.

### Class B: new latent model using the current latent-block shape

Examples:

- `rw2`
- `ar2`
- another chain-based or block-diagonal GMRF with no extra front-end inputs

Typical work:

- add a new `QFunc` in `models/mod.rs`
- use an existing graph constructor or add a simple new one in `graph/mod.rs`
- register the model string in `R/f.R` and `src/rust/src/lib.rs`
- add default `theta_init`
- add regression and reference tests

Expected difficulty:

- easy to medium

This is the cheapest latent-model extension path.

### Class C: new graph-driven GMRF using the current core but a wider API

Examples:

- `besag`
- weighted or custom-neighbor intrinsic models

Typical work:

- implement the `QFunc` in the core
- expose graph input in the R-side backend spec
- validate graph indexing, symmetry, and constraints in the bridge
- add tests and spatial reference cases

Expected difficulty:

- medium to hard

The core can likely support this already, but the public API does not.

### Class D: coupled, scaled, or multi-component latent models

Examples:

- `bym2`
- separable Kronecker models
- `SPDE`

Typical work:

- new core model logic
- likely new scaling conventions
- likely new front-end data structures
- more demanding validation and benchmark design

Expected difficulty:

- hard

These are architecture-expanding features, not just local additions.

## 6. How hard is it to add another family?

Usually easier than adding another latent model.

If the family can be expressed as `p(y_i | eta_i, theta)` with one shared linear predictor and analytic `gradient_and_curvature`, the work is mostly local to:

- [src/rust/inla_core/src/likelihood/mod.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/likelihood/mod.rs)
- [src/rust/src/lib.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/src/lib.rs)
- [src/rust/inla_core/tests/test_basic.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/tests/test_basic.rs)

What makes a family harder:

- more than one predictor
- additional per-observation inputs not in the current backend spec
- non-concavity that needs stabilization
- special functions or approximation regimes that are numerically delicate

Rule of thumb:

- local GLM-like family: low to medium effort
- zero-inflated or hurdle family: medium effort
- multi-part or structurally richer family: medium to high effort

## 7. How hard is it to add another GMRF latent model?

This depends more on the interface than on the math.

If the model only needs:

- `n_levels`
- a standard sparse graph pattern
- a `Q(i,j,theta)` implementation

then the current architecture is already a good fit.

If the model needs:

- a user-supplied graph
- scaling metadata
- multiple coupled blocks
- extra constraints or normalization

then the current public API becomes the main bottleneck.

Rule of thumb:

- chain or diagonal model: low to medium effort
- graph-based spatial model with adjacency input: medium to high effort
- coupled or scaled model family: high effort

## 8. Concrete extension checklist

### To add a new family

1. Add the likelihood struct and trait implementation in [likelihood/mod.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/likelihood/mod.rs).
2. Register the family name in [src/rust/src/lib.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/src/lib.rs:436).
3. Add default likelihood theta initialization in [src/rust/src/lib.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/src/lib.rs:199).
4. Add unit tests for log-likelihood, gradient, curvature, and prior behavior.
5. Add at least one end-to-end or benchmark-style comparison against `R-INLA` or a trusted reference, including timing, memory, and parameter-accuracy metrics for the affected outputs.
6. Update supported-scope documentation.

### To add a new latent model

1. Add the `QFunc` implementation in [models/mod.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/models/mod.rs).
2. Add or reuse the graph constructor in [graph/mod.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/graph/mod.rs).
3. Register the model name in [R/f.R](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/R/f.R) and [src/rust/src/lib.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/src/lib.rs:155).
4. Add default model theta initialization in [src/rust/src/lib.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/src/lib.rs:180).
5. If the model needs new front-end inputs, widen [R/interface.R](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/R/interface.R) and the bridge parser.
6. Add model-level tests, end-to-end tests, and benchmark/reference cases with timing, memory, and parameter-accuracy metrics for the affected outputs.
7. Update supported-scope documentation.

## 9. Recommended path for future development

### Lowest-friction next additions

- one more GLM-like likelihood family

The current architecture still makes another GLM-like family the cheapest next meaningful expansion.

### High-value additions after a modest API expansion

- graph-driven intrinsic spatial models such as `besag`
- families that need slightly richer metadata than the current spec but still fit the single-predictor path

These likely require widening the R/backend spec, but not rewriting the core.

### Deliberately later additions

- `bym2`
- `SPDE`
- mesh workflows
- multi-part likelihood stacks beyond the current subset

These should be treated as separate architecture milestones, not just “next models.”

## 10. Recommended inventory maintenance

This document should be updated whenever one of these changes:

- a new family is added
- a new latent model is added
- the backend spec gains new input kinds
- a previously experimental feature becomes part of the stable benchmark suite
- a new binding path is added beyond the current R interface

If this document is kept current, contributors can quickly answer whether a new idea is:

- a local addition
- a front-end expansion
- or a true architecture milestone

## 11. R-INLA parity inventory

This guide focuses on the implemented subset and the extension path.

For the fuller inventory of what is still missing relative to the local R-INLA reference installation, including live registry counts for likelihoods, latent models, priors, links, and related surfaces, see [RINLA_PARITY_GAP_INVENTORY.md](RINLA_PARITY_GAP_INVENTORY.md).

For the current recommended order of execution, see [EXTENSION_BACKLOG.md](EXTENSION_BACKLOG.md). For the directory-level touch map behind those steps, see [EXTENSION_INTERVENTION_MAP.md](EXTENSION_INTERVENTION_MAP.md).
