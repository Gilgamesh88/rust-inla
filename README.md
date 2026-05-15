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
  - synthetic Gaussian + multiple fixed effects + `iid` + offset, covering the completed Phase 7A fixed-effects slice
  - fixed-effect-only GLMs through the zero-latent backend path

| Likelihood Model | Latent Component | Rusty-INLA | R-INLA | Status |
| --- | --- | --- | --- | --- |
| **Poisson** (Freq) | `IID` (VehBrand) | `33.51 sec`, `393.4 MB` | `22.11 sec`, `1112.2 MB` | PASS |
| **Poisson** (Freq) | `IID` (VehBrand) + `IID` (Region) | `32.41 sec`, `871.7 MB` | `39.03 sec`, `1339.5 MB` | PASS |
| **Gamma** (Severity) | `RW1` (AgeGroup) | `2.52 sec`, `826.7 MB` | `2.30 sec`, `970.8 MB` | PASS |
| **Poisson** (Freq) | `AR1` (AgeIndex) | `41.67 sec`, `1038.3 MB` | `37.41 sec`, `1591.1 MB` | PASS |
| **Zero-Infl. Poisson** | `IID` (VehBrand) | `73.95 sec`, `1221.9 MB` | `98.45 sec`, `1850.1 MB` | PASS |

The detailed comparison note is tracked in [scratch/BENCHMARK_SUMMARY_2026-04-19.md](scratch/BENCHMARK_SUMMARY_2026-04-19.md).

For deeper parity inspection of returned summaries, set `RUSTYINLA_OUTPUT_PROFILE=benchmark` before running the local harnesses. That extended mode compares additional fit-object surfaces such as fixed-effect standard deviations, hyperparameter summaries, and linear-predictor summaries.

For the current implemented subset, the core architecture and file map, the Phase 7A fixed-effects formula scope, the uploaded-suite supported-subset manifest, the coverage evaluation, the detailed R-INLA parity gap inventory, the public API-surface inventory, the API implementation queue, the Phase 8 sequential-update roadmap, the clean-room actuarial distribution roadmap, the posterior-state update RFC, the external-example benchmarking guide, the directory-level intervention map, and the recommended path for adding new families or latent models, see [CORE_ARCHITECTURE_MAP.md](CORE_ARCHITECTURE_MAP.md), [IMPLEMENTATION_INVENTORY_AND_EXTENSION_GUIDE.md](IMPLEMENTATION_INVENTORY_AND_EXTENSION_GUIDE.md), [FIXED_EFFECTS_FORMULA_SUBSET.md](FIXED_EFFECTS_FORMULA_SUBSET.md), [SUPPORTED_SUBSET_VALIDATION_MANIFEST.md](SUPPORTED_SUBSET_VALIDATION_MANIFEST.md), [COVERAGE_EVALUATION_2026-04-19.md](COVERAGE_EVALUATION_2026-04-19.md), [RINLA_PARITY_GAP_INVENTORY.md](RINLA_PARITY_GAP_INVENTORY.md), [RINLA_API_SURFACE_INVENTORY.md](RINLA_API_SURFACE_INVENTORY.md), [API_IMPLEMENTATION_QUEUE.md](API_IMPLEMENTATION_QUEUE.md), [PHASE8_ROADMAP.md](PHASE8_ROADMAP.md), [CLEAN_ROOM_DISTRIBUTION_ROADMAP.md](CLEAN_ROOM_DISTRIBUTION_ROADMAP.md), [POSTERIOR_STATE_UPDATE_RFC.md](POSTERIOR_STATE_UPDATE_RFC.md), [EXTERNAL_EXAMPLE_BENCHMARKING_GUIDE.md](EXTERNAL_EXAMPLE_BENCHMARKING_GUIDE.md), [EXTENSION_INTERVENTION_MAP.md](EXTENSION_INTERVENTION_MAP.md), and [EXTENSION_BACKLOG.md](EXTENSION_BACKLOG.md).

## Implementation Roadmap (75% Complete)

Our goal is to port the subset of INLA specifically relied upon by the actuarial industry, enhancing velocity without sacrificing gradient accuracy.

- [x] **Phase 1:** Core Optimization (L-BFGS, Laplace Newton-Raphson solvers).
- [x] **Phase 2:** Base Latent Topologies (IID, Random Walk 1, Random Walk 2, Auto-Regressive 1, Auto-Regressive 2).
- [x] **Phase 3:** Hyperparameter Uncertainty via Central Composite Design (CCD).
- [x] **Phase 4:** Core Likelihoods (Gaussian).
- [x] **Phase 5:** Actuarial Likelihoods (ZIP Type-1, initial Tweedie prototype).
- [x] **Phase 6:** Native R Formula Parsing Interface (`y ~ 1 + f(...)`).
- [x] **Phase 7A:** Generalized Fixed Effects Matrix. (Productizing the supported `$X\beta$` dense covariate subset).
- [ ] **Phase 8:** Prior/control metadata reuse and sequential update states. (Phase 8A is the active one-`iid` fixed-cross evidence path; later 8C-8E steps build toward theta-mixture, full-Bayes-like updating.)
- [ ] **Phase 9:** Clean-room actuarial distribution expansion. (Native `nbinomial2`, constant zero-inflated count families, Tweedie stabilization, and then structural covariates for zero-inflation through a second-predictor surface such as `ZIP Type-2`).
- [ ] **Phase 10:** Dynamic Arbitrary Priors. (Exposing richer prior modification arrays to the R frontend after the Phase 8 metadata layer exists).

Tweedie support remains experimental and is currently excluded from the active parity benchmark sweep until the instability path is better understood.

Phase 9 is tracked in [CLEAN_ROOM_DISTRIBUTION_ROADMAP.md](CLEAN_ROOM_DISTRIBUTION_ROADMAP.md). It is explicitly clean-room: external packages such as `glmmTMB` may be used as optional black-box reference oracles, but not as runtime dependencies or source-code references.

Phase 7A is complete for the MVP supported subset: multiple fixed-effect columns are validated through the current `model.matrix()` path, fixed-effect-only GLMs are supported through the zero-latent backend path, rank-deficient fixed designs fail fast with a clear error, unsupported formula surfaces are rejected before Rust is called, and the external reference harness includes a multi-fixed-effect Gaussian + `iid` + offset comparison against `R-INLA`.

The current fixed-effects subset intentionally supports bare numeric/logical/factor columns, simple interactions among those columns, formula offsets, fixed-effect-only GLMs, and standalone `f(...)` latent terms. Transformed fixed terms such as `log(x)`, `I(x^2)`, `poly(...)`, and `factor(...)` should be materialized in `data` first. See [FIXED_EFFECTS_FORMULA_SUBSET.md](FIXED_EFFECTS_FORMULA_SUBSET.md) for the exact contract.

Custom priors, fixed-prior controls such as `control.fixed`, and broader prior override surfaces are now tracked as Phase 8 prior/control metadata reuse rather than unfinished Phase 7A work.

The first Phase 8 experiment is intentionally narrow: `rusty_posterior_state()`
can extract an `iid` hyperparameter posterior state, and `control.update` can
reuse that state as a Gaussian prior for a later one-`iid` fit. This is an
experimental posterior-as-prior path for testing sequential Bayesian updating,
not full latent-state reuse.

The second Phase 8 experiment adds `rusty_update_state()` for a stricter
fixed-plus-`iid` workflow: it extracts dense fixed-effect Gaussian evidence and
diagonal `iid` level evidence plus the fixed-`iid` cross block from an old fit,
then reuses that object in
`control.update = list(state = state, mode = "fixed_iid_cross_gaussian_evidence")`.
New `iid` levels are allowed and receive zero old evidence, so they start from
the ordinary zero-mean `iid` prior and update from new data. This path remains
experimental because the old evidence is still a local Gaussian approximation
and hyperparameter uncertainty is simplified, but the cross block directly
addresses the SD narrowing seen in the diagonal-only diagnostic mode.

The active Phase 8 breakdown is tracked in [PHASE8_ROADMAP.md](PHASE8_ROADMAP.md):
8A stabilizes fixed-plus-`iid` cross evidence, 8B formalizes old-data evidence
semantics, 8C and 8D add theta-mixture evidence and a theta-dependent objective,
8E defines joint-refit validation gates, and 8H starts the recursive evidence
graph for multiple `iid` effects.

The current 8B contract is explicit: `rusty_update_state()` stores old-data
likelihood evidence, not a posterior-as-prior object, and update fits keep the
original model priors active while adding those old-data evidence factors.

The 8C slice uses the actual old CCD/theta support when available:
`rusty_update_state()` carries a `theta_evidence` container with support-point
theta values, normalized weights, log marginal likelihoods, local Gaussian log
constants, and the fixed/`iid` evidence blocks (`H_beta_beta`, `h_beta`,
`H_u_u_diag`, `h_u`, and `H_u_beta`). The 8D opt-in mode
`fixed_iid_cross_theta_evidence` consumes that container with one-dimensional
linear interpolation across theta support points; the older
`fixed_iid_cross_gaussian_evidence` mode remains the frozen source-mode
comparator. For rolling updates, `rusty_compose_update_state(previous_state,
fit)` carries compressed evidence forward by adding the previous state to the
current period's extracted likelihood evidence; this is the supported
experimental path for year-by-year diagnostics. Factor `iid` levels that are
present in the update state but absent from the current period are carried as
dormant latent parameters, so actuarial tables can keep old brand parameters
through zero-exposure periods and update them again when exposure returns.

The first multi-`iid` implementation carries dense fixed evidence, diagonal iid
evidence for every block, fixed-iid cross blocks, and sparse iid-iid cross
edges. The theta-dependent path can consume the same graph at old theta support
points. Multiple theta dimensions now use a guarded local interpolation rule:
inside the old CCD support box it blends nearby support evidence, while outside
that box it falls back to the nearest support point to avoid unsafe
extrapolation. For multi-`iid` states, `rusty_update_state()` also adds
expanded axial guard support by fixed-theta replay on the old data; in the
focused synthetic outside-cloud test this reduced theta drift from `6.099695`
with the original CCD support to `1.999375`. Multi-`iid` theta drift is still
diagnostic-only until this path is validated on real rolling data.

For the curated uploaded-suite subset, run:

```powershell
& 'C:\Program Files\R\R-4.5.3\bin\Rscript.exe' tools\run_supported_subset_validation.R
```

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
