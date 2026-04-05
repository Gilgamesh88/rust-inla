# Rusty-INLA

**Rusty-INLA** is a high-performance, clean-room port of the `R-INLA` statistical backend engineered in Rust. Purpose-built for massive actuarial datasets, it bypasses the traditional memory-heavy C pathways of native INLA by parsing R inputs via `ALTREP` direct memory binding, and executing exact Laplace Approximations and Central Composite Design (CCD) topologies using optimized sparse algebra.

## Performance & Accuracy Benchmark

Evaluated against the ~678,000 policy rows of the `freMTPL2` dataset from `CASdatasets`. Results demonstrate exact mathematical parity with native `R-INLA` while drastically reducing execution time and computational memory overhead.

| Likelihood Model | Latent Component | Rusty-INLA (Time) | R-INLA (Time) | Rusty-INLA (Mem) | R-INLA (Mem) | Parameter Match (Mean) |
| --- | --- | --- | --- | --- | --- | --- |
| **Poisson** (Freq) | `IID` (VehBrand) | **10.91 sec** | 15.41 sec | **2.95 GB** | 4.79 GB | ± 0.0006 |
| **Gamma** (Severity) | `RW1` (AgeGroup) | **1.03 sec** | 2.32 sec | **3.13 GB** | 3.28 GB | ± 0.0030 |
| **Zero-Infl. Poisson** | `IID` (VehBrand) | **36.83 sec** | 92.14 sec | **3.14 GB** | 4.75 GB | ± 0.0027 |
| **Tweedie** (Premium)| `RW1` (AgeGroup) | **28.33 sec** | 165.43 sec | **3.30 GB** | 4.72 GB | ± 0.0017 |

*Note: Rusty-INLA's Tweedie implementation operates natively via a mathematically bounded Saddlepoint Approximation, completely side-stepping the numerical instability crashes associated with R-INLA's infinite series evaluators.*

## Implementation Roadmap (75% Complete)

Our goal is to port the subset of INLA specifically relied upon by the actuarial industry, enhancing velocity without sacrificing gradient accuracy. 

- [x] **Phase 1:** Core Optimization (L-BFGS, Laplace Newton-Raphson solvers).
- [x] **Phase 2:** Base Latent Topologies (IID, Random Walk 1, Auto-Regressive 1).
- [x] **Phase 3:** Hyperparameter Uncertainty via Central Composite Design (CCD).
- [x] **Phase 4:** Core Likelihoods (Gaussian).
- [x] **Phase 5:** Actuarial Likelihoods (ZIP Type-1, Extended Tweedie Saddlepoint).
- [x] **Phase 6:** Native R Formula Parsing Interface (`y ~ 1 + f(...)`).
- [ ] **Phase 7:** Generalized Fixed Effects Matrix. (Expanding the parser to handle `$X\beta$` dense covariate matrices).
- [ ] **Phase 8:** Multi-variate Likelihoods. (Allowing structural covariates to predict Zero-Inflation probabilities via joint frameworks: `ZIP Type-2`).
- [ ] **Phase 9:** Dynamic Arbitrary Priors. (Exposing prior modification arrays to the R frontend).

---
*Built via `extendr` inside `rust-inla`.*
