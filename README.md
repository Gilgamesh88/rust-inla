# Rusty-INLA

**Rusty-INLA** is a high-performance, clean-room port of the `R-INLA` statistical backend engineered in Rust. Purpose-built for massive actuarial datasets, it bypasses the traditional memory-heavy C pathways of native INLA by parsing R inputs via `ALTREP` direct memory binding, and executing exact Laplace approximations and Central Composite Design (CCD) topologies using optimized sparse algebra.

## Current Benchmark Snapshot

The latest local parity sweep was refreshed on April 21, 2026 against the `freMTPL2` data from `CASdatasets` using [benchmark.R](benchmark.R). The active benchmark suite currently excludes Tweedie because that path remains unstable and is not yet a reliable parity target.

Headline result on the active stable suite:

- `rustyINLA` passes `5/5` benchmark cases.
- `rustyINLA` uses less memory than `R-INLA` in every benchmarked case.
- Implemented latent models now include `iid`, `rw1`, `rw2`, `ar1`, and `ar2`.
- Count-model parity is now strong for:
  - Poisson + `iid`
  - Poisson + `iid + iid`
  - Poisson + `ar1`
  - Zero-inflated Poisson + `iid`
- Gamma + `rw1` now passes after the intrinsic-field covariance fix.
- External/reference coverage now also includes:
  - Gaussian + `rw2`, with the LIDAR smoothing example numerically close to `R-INLA`
  - synthetic Gaussian + `ar2`, compared against `R-INLA` `model = "ar", order = 2`
  - synthetic Gaussian + multiple fixed effects + `iid`, covering the first Phase 7A fixed-effects slice

| Likelihood Model | Latent Component | Rusty-INLA | R-INLA | Status |
| --- | --- | --- | --- | --- |
| **Poisson** (Freq) | `IID` (VehBrand) | `33.51 sec`, `393.4 MB` | `22.11 sec`, `1112.2 MB` | PASS |
| **Poisson** (Freq) | `IID` (VehBrand) + `IID` (Region) | `32.41 sec`, `871.7 MB` | `39.03 sec`, `1339.5 MB` | PASS |
| **Gamma** (Severity) | `RW1` (AgeGroup) | `2.52 sec`, `826.7 MB` | `2.30 sec`, `970.8 MB` | PASS |
| **Poisson** (Freq) | `AR1` (AgeIndex) | `41.67 sec`, `1038.3 MB` | `37.41 sec`, `1591.1 MB` | PASS |
| **Zero-Infl. Poisson** | `IID` (VehBrand) | `73.95 sec`, `1221.9 MB` | `98.45 sec`, `1850.1 MB` | PASS |

The detailed comparison note is tracked in [scratch/BENCHMARK_SUMMARY_2026-04-19.md](scratch/BENCHMARK_SUMMARY_2026-04-19.md).

For deeper parity inspection of returned summaries, set `RUSTYINLA_OUTPUT_PROFILE=benchmark` before running the local harnesses. That extended mode compares additional fit-object surfaces such as fixed-effect standard deviations, hyperparameter summaries, and linear-predictor summaries.

For the current implemented subset, the coverage evaluation, the detailed R-INLA parity gap inventory, the public API-surface inventory, the API implementation queue, the posterior-state update RFC, the external-example benchmarking guide, the directory-level intervention map, and the recommended path for adding new families or latent models, see [IMPLEMENTATION_INVENTORY_AND_EXTENSION_GUIDE.md](IMPLEMENTATION_INVENTORY_AND_EXTENSION_GUIDE.md), [COVERAGE_EVALUATION_2026-04-19.md](COVERAGE_EVALUATION_2026-04-19.md), [RINLA_PARITY_GAP_INVENTORY.md](RINLA_PARITY_GAP_INVENTORY.md), [RINLA_API_SURFACE_INVENTORY.md](RINLA_API_SURFACE_INVENTORY.md), [API_IMPLEMENTATION_QUEUE.md](API_IMPLEMENTATION_QUEUE.md), [POSTERIOR_STATE_UPDATE_RFC.md](POSTERIOR_STATE_UPDATE_RFC.md), [EXTERNAL_EXAMPLE_BENCHMARKING_GUIDE.md](EXTERNAL_EXAMPLE_BENCHMARKING_GUIDE.md), [EXTENSION_INTERVENTION_MAP.md](EXTENSION_INTERVENTION_MAP.md), and [EXTENSION_BACKLOG.md](EXTENSION_BACKLOG.md).

## Implementation Roadmap (75% Complete)

Our goal is to port the subset of INLA specifically relied upon by the actuarial industry, enhancing velocity without sacrificing gradient accuracy.

- [x] **Phase 1:** Core Optimization (L-BFGS, Laplace Newton-Raphson solvers).
- [x] **Phase 2:** Base Latent Topologies (IID, Random Walk 1, Random Walk 2, Auto-Regressive 1, Auto-Regressive 2).
- [x] **Phase 3:** Hyperparameter Uncertainty via Central Composite Design (CCD).
- [x] **Phase 4:** Core Likelihoods (Gaussian).
- [x] **Phase 5:** Actuarial Likelihoods (ZIP Type-1, initial Tweedie prototype).
- [x] **Phase 6:** Native R Formula Parsing Interface (`y ~ 1 + f(...)`).
- [ ] **Phase 7:** Generalized Fixed Effects Matrix. (Expanding the parser to handle `$X\beta$` dense covariate matrices).
- [ ] **Phase 8:** Multi-variate Likelihoods. (Allowing structural covariates to predict zero-inflation probabilities via joint frameworks such as `ZIP Type-2`).
- [ ] **Phase 9:** Dynamic Arbitrary Priors. (Exposing prior modification arrays to the R frontend).

Tweedie support remains experimental and is currently excluded from the active parity benchmark sweep until the instability path is better understood.

Phase 7 has now started in a narrow productization slice: multiple fixed-effect columns are validated through the current `model.matrix()` path, rank-deficient fixed designs fail fast with a clear error, and the external reference harness includes a multi-fixed-effect Gaussian + `iid` comparison against `R-INLA`.

## Installing Rust

If you want to build `rustyINLA` from source, or install it from GitHub with `remotes::install_github()` / `devtools::install_github()`, install Rust first with the official `rustup` tool.

On Windows:

1. Download and run the official `rustup-init.exe` installer from [rust-lang.org](https://www.rust-lang.org/tools/install).
2. Open a new PowerShell window after the installer finishes.
3. Add the GNU target used by this package build:

```powershell
rustup target add x86_64-pc-windows-gnu
```

4. Verify the toolchain is available:

```powershell
rustc --version
cargo --version
rustup target list --installed
```

On macOS, Linux, or WSL, install Rust with `rustup` using the official command from [rust-lang.org](https://www.rust-lang.org/tools/install), then verify with `rustc --version` and `cargo --version`.

For Windows source installs, you will also need a working R toolchain such as `Rtools`, because this package is built with the GNU target on Windows.

## Windows Validation

For the current Windows GNU toolchain flow, use:

```powershell
.\tools\check-rust-workspace-win.ps1
```

That wrapper bootstraps the R and `extendr` build environment and then runs the workspace Rust checks on the same target configuration used by the package build.

If `R CMD INSTALL` is flaky in the current Windows shell, the local benchmark harnesses can still load the package directly from the worktree through [tools/load_worktree_package.R](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/tools/load_worktree_package.R).

---
*Built via `extendr` inside `rust-inla`.*
