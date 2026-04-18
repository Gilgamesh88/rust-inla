# Rusty-INLA

**Rusty-INLA** is a high-performance, clean-room port of the `R-INLA` statistical backend engineered in Rust. Purpose-built for massive actuarial datasets, it bypasses the traditional memory-heavy C pathways of native INLA by parsing R inputs via `ALTREP` direct memory binding, and executing exact Laplace approximations and Central Composite Design (CCD) topologies using optimized sparse algebra.

## Current Benchmark Snapshot

The latest local parity sweep was run on April 18, 2026 against the `freMTPL2` data from `CASdatasets` using [benchmark.R](benchmark.R). The active benchmark suite currently excludes Tweedie because that path remains unstable and is not yet a reliable parity target.

Headline result on the active stable suite:

- `rustyINLA` passes `5/5` benchmark cases.
- `rustyINLA` uses less memory than `R-INLA` in every benchmarked case.
- Count-model parity is now strong for:
  - Poisson + `iid`
  - Poisson + `iid + iid`
  - Poisson + `ar1`
  - Zero-inflated Poisson + `iid`
- Gamma + `rw1` now passes after the intrinsic-field covariance fix.

| Likelihood Model | Latent Component | Rusty-INLA | R-INLA | Status |
| --- | --- | --- | --- | --- |
| **Poisson** (Freq) | `IID` (VehBrand) | `58.61 sec`, `359.3 MB` | `22.81 sec`, `1112.0 MB` | PASS |
| **Poisson** (Freq) | `IID` (VehBrand) + `IID` (Region) | `52.22 sec`, `845.6 MB` | `39.17 sec`, `1339.2 MB` | PASS |
| **Gamma** (Severity) | `RW1` (AgeGroup) | `3.95 sec`, `825.4 MB` | `2.42 sec`, `970.5 MB` | PASS |
| **Poisson** (Freq) | `AR1` (AgeIndex) | `55.83 sec`, `1012.1 MB` | `36.50 sec`, `1590.9 MB` | PASS |
| **Zero-Infl. Poisson** | `IID` (VehBrand) | `102.97 sec`, `1195.6 MB` | `98.02 sec`, `1849.7 MB` | PASS |

The detailed comparison note is tracked in [scratch/BENCHMARK_SUMMARY_2026-04-18.md](scratch/BENCHMARK_SUMMARY_2026-04-18.md).

## Implementation Roadmap (75% Complete)

Our goal is to port the subset of INLA specifically relied upon by the actuarial industry, enhancing velocity without sacrificing gradient accuracy.

- [x] **Phase 1:** Core Optimization (L-BFGS, Laplace Newton-Raphson solvers).
- [x] **Phase 2:** Base Latent Topologies (IID, Random Walk 1, Auto-Regressive 1).
- [x] **Phase 3:** Hyperparameter Uncertainty via Central Composite Design (CCD).
- [x] **Phase 4:** Core Likelihoods (Gaussian).
- [x] **Phase 5:** Actuarial Likelihoods (ZIP Type-1, initial Tweedie prototype).
- [x] **Phase 6:** Native R Formula Parsing Interface (`y ~ 1 + f(...)`).
- [ ] **Phase 7:** Generalized Fixed Effects Matrix. (Expanding the parser to handle `$X\beta$` dense covariate matrices).
- [ ] **Phase 8:** Multi-variate Likelihoods. (Allowing structural covariates to predict zero-inflation probabilities via joint frameworks such as `ZIP Type-2`).
- [ ] **Phase 9:** Dynamic Arbitrary Priors. (Exposing prior modification arrays to the R frontend).

Tweedie support remains experimental and is currently excluded from the active parity benchmark sweep until the instability path is better understood.

---
*Built via `extendr` inside `rust-inla`.*
