# v0.1 Roadmap: Validated Rust-Native Subset

This roadmap defines the first publishable subset for `rustyINLA`.

The goal of `v0.1` is not full INLA parity. The goal is:

- a Rust-native core;
- a validated subset;
- clear documentation of supported and unsupported areas;
- a foundation the community can extend.

For the current subsystem inventory, the detailed R-INLA parity-gap inventory, the directory-level extension map, and the recommended extension path for new likelihoods and latent models, see [IMPLEMENTATION_INVENTORY_AND_EXTENSION_GUIDE.md](IMPLEMENTATION_INVENTORY_AND_EXTENSION_GUIDE.md), [RINLA_PARITY_GAP_INVENTORY.md](RINLA_PARITY_GAP_INVENTORY.md), [EXTENSION_INTERVENTION_MAP.md](EXTENSION_INTERVENTION_MAP.md), and [EXTENSION_BACKLOG.md](EXTENSION_BACKLOG.md).

## 1. v0.1 release goal

Ship a small, honest, reproducible subset with:

- useful actuarial and latent-Gaussian use cases;
- benchmark-based validation against INLA;
- a clean provenance story;
- a stable architecture for future contributors.

## 2. What `v0.1` should include

### Latent models

- `iid`
- `rw1`
- `ar1`
- compound block-diagonal combinations of those models

### Likelihoods

- `gaussian`
- `poisson`
- `gamma`
- `zeroinflatedpoisson1`
- `tweedie`

### Core engine features

- sparse graph construction
- sparse precision evaluation
- sparse Cholesky factorization
- log-determinant computation
- selected inverse for current subset
- latent mode finding
- outer hyperparameter optimization
- CCD-based hyperparameter integration
- offsets and exposure handling for count models
- fixed effects
- fitted values and marginals for supported cases
- diagnostics and profiling hooks

### Interfaces

- R frontend through the current Rust bridge
- reproducible benchmark scripts
- saved golden reference outputs for supported cases

## 3. What `v0.1` should explicitly not include

- `SPDE`
- mesh workflows requiring `fmesher`
- `rw2` unless fully validated first
- multivariate likelihood stacks beyond the current subset
- `zeroinflatedpoisson2`
- expression-prior parsing via `muparser`
- native INI/config parser compatibility
- `PARDISO` / `MKL` backend support
- METIS-dependent solver strategy

## 4. Status by subsystem

| Subsystem | Current state | v0.1 target |
| --- | --- | --- |
| Sparse solver core | Implemented in Rust | Stabilize and benchmark |
| Selected inverse | Implemented in Rust | Keep and validate for supported models |
| Hyperparameter optimizer | Implemented in Rust | Improve parity and runtime |
| CCD integration | Implemented in Rust | Validate weights and grid behavior |
| `iid`, `rw1`, `ar1` | Implemented | Validate thoroughly |
| Compound latent blocks | Implemented | Validate with multi-block benchmarks |
| Gaussian / Poisson / Gamma / ZIP1 / Tweedie | Implemented | Tighten parity and regression tests |
| Fixed effects | Implemented | Expand validation coverage |
| Exposure / offset handling | Implemented | Keep benchmark-validated behavior |
| SPDE / mesh | Not implemented | Defer |
| Expression priors | Not implemented | Defer |
| METIS / PARDISO / MKL | Not implemented | Defer |

## 5. Validation requirements for `v0.1`

Before release, each supported benchmark family should have:

- a frozen input dataset or synthetic case;
- saved INLA reference outputs;
- saved control settings and initial values;
- accepted tolerances for fixed effects;
- accepted tolerances for hyperparameters;
- accepted tolerances for fitted values;
- runtime and memory snapshots;
- a note of any known deviation.

## 6. Benchmark families to prioritize

The first public validation set should include at least:

1. Gaussian + `rw1`
2. Poisson + `iid`
3. Poisson + `iid + iid`
4. Poisson + `ar1`
5. Gamma + `rw1`
6. ZIP1 + `iid`
7. Tweedie + `rw1`

## 7. Engineering milestones

### Milestone A: Project hygiene

- choose and apply the project license
- replace placeholder package metadata
- add provenance and third-party docs
- add a validation matrix

### Milestone B: Trustworthy parity

- keep narrowing the mismatch in Poisson multi-block cases
- improve AR1 hyperparameter calibration
- improve ZIP and Tweedie robustness
- document every known non-parity case

### Milestone C: Performance credibility

- keep diagnostics enabled for internal profiling
- reduce repeated latent-mode solves
- improve warm-start behavior
- revisit ordering heuristics
- publish reproducible memory/runtime comparisons

### Milestone D: Community readiness

- write a contributor-oriented architecture overview
- define how new likelihoods and latent models get added
- require golden outputs for new functionality
- make unsupported areas explicit

## 8. Definition of done for `v0.1`

`v0.1` is done when:

- the supported subset is clearly documented;
- benchmark cases run reproducibly;
- parity gaps are known and bounded;
- memory/runtime claims are backed by scripts;
- contributors can see what is safe to extend;
- deferred systems are explicitly out of scope rather than silently missing.

## 9. Post-v0.1 candidates

Good post-`v0.1` candidates:

- better ordering/reordering layer
- additional latent models with strong demand
- more robust priors and hyperparameter controls
- Python or CLI frontend
- optional advanced approximation modes

Bad post-`v0.1` candidates for the immediate next step:

- full-package INLA parity
- immediate `SPDE` port
- direct migration of mixed-license native subsystems
- hard dependency on `PARDISO` or `METIS`
