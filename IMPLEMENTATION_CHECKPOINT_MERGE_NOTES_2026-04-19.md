# Implementation Checkpoint Merge Notes

This document annotates the large implementation checkpoint committed after the planning and RFC pass.

Its purpose is to make the checkpoint mergeable later without forcing future work to rediscover the meaning of a large mixed diff.

## 1. Why this checkpoint exists

The worktree accumulated several rounds of related implementation work before it was split into clean commits.

This checkpoint intentionally captures:

- engine and numerical-core changes
- R-interface and benchmark-harness changes
- benchmark and diagnostic scripts used to validate those changes
- Windows workspace helper/tooling needed to reproduce checks locally

It does **not** try to be the final polished history shape.

The expected later workflow is:

- keep this checkpoint as a safe backup
- later split or cherry-pick the parts into cleaner topic commits if needed

## 2. Main implementation themes captured here

### A. Core correctness and stability work

The checkpoint includes work related to:

- Schur stabilization and numerical robustness
- ZIP stability and fixed-theta replay alignment
- constraint-aware covariance summaries for intrinsic models such as `rw1`
- AR1 precision/correlation corrections
- tighter summary evaluation behavior at CCD points

Main code areas:

- `src/rust/inla_core/src/problem/`
- `src/rust/inla_core/src/optimizer/`
- `src/rust/inla_core/src/likelihood/`
- `src/rust/inla_core/src/models/`
- `src/rust/inla_core/src/inference/`

### B. Public bridge and R-layer work

The checkpoint includes work related to:

- warm-start support through the Rust bridge
- output-profile support (`thin` vs `benchmark`)
- additional exported summaries and marginals
- benchmark-facing R object shaping

Main code areas:

- `src/rust/src/lib.rs`
- `R/interface.R`
- `R/f.R`
- `R/extendr-wrappers.R`

### C. Performance work

The checkpoint includes low-risk performance work, especially around repeated inner computations and CCD evaluation reuse.

Main code areas:

- `src/rust/inla_core/src/problem/`
- `src/rust/inla_core/src/optimizer/`
- `src/rust/inla_core/src/solver/`

### D. Benchmark and external-reference work

The checkpoint includes:

- internal benchmark harness updates
- external reference benchmark harnesses
- parity diagnostics and replay scripts
- benchmark summary notes

Main areas:

- `benchmark.R`
- `scratch/*.R`
- selected `scratch/*.md`

### E. Tooling and workspace reproducibility

The checkpoint includes:

- Windows R build-environment helper
- wrapper/workspace cleanup support
- helper changes in package build config

Main areas:

- `tools/with-r-build-env.ps1`
- `tools/check-rust-workspace-win.ps1`
- `tools/config.R`
- `src/Makevars.in`
- `src/Makevars.win.in`

## 3. What is intentionally excluded from the checkpoint

These local artifacts should remain outside the commit history unless there is a specific reason to version them:

- local R library installs under `scratch/rlib/`
- generated import-library outputs under `scratch/build_support/`
- downloaded or cached external reference data under `scratch/external_reference_data/`
- generated benchmark CSVs
- generated plots
- `.Rhistory`
- transient `last_failed_matrix` snapshots unless explicitly needed

If those are needed later, they should be added deliberately in their own commit.

## 4. Recommended later split

If this checkpoint ever needs to be split into cleaner history, the recommended order is:

1. `fix(engine): restore numerical parity for current supported subset`
   Likely paths:
   - `src/rust/inla_core/src/problem/`
   - `src/rust/inla_core/src/optimizer/`
   - `src/rust/inla_core/src/likelihood/`
   - `src/rust/inla_core/src/models/`
   - `src/rust/inla_core/src/inference/`

2. `feat(api): expose warm starts and benchmark-oriented output shaping`
   Likely paths:
   - `src/rust/src/lib.rs`
   - `R/interface.R`
   - `R/f.R`
   - `R/extendr-wrappers.R`

3. `perf(core): reduce repeated inner-work overhead`
   Likely paths:
   - `src/rust/inla_core/src/problem/`
   - `src/rust/inla_core/src/solver/`
   - possibly `src/rust/inla_core/src/optimizer/`

4. `chore(tooling): add Windows R workspace helper and cleanup`
   Likely paths:
   - `tools/with-r-build-env.ps1`
   - `tools/config.R`
   - `src/Makevars.in`
   - `src/Makevars.win.in`
   - deletion of `src/rust/tests/try_faer.rs`

5. `chore(research): preserve benchmark and diagnostic scripts`
   Likely paths:
   - `benchmark.R`
   - selected `scratch/*.R`
   - selected `scratch/*.md`

## 5. Verification context known at checkpoint time

Known validation context from the work that led to this checkpoint:

- active internal benchmark suite reached `5/5` pass with Tweedie intentionally outside MVP
- external/reference suite also reached `5/5` pass in the validated rerun path
- benchmark memory remained below `R-INLA` across the active suite
- several parity bugs were diagnosed through exact-theta and common-support scripts preserved under `scratch/`

This checkpoint should therefore be read as:

- not a random WIP snapshot
- but a backed-up implementation state with meaningful validation behind it

## 6. Merge guidance for future work

When merging or building on this checkpoint:

- trust the high-level benchmark state, but rerun the relevant suite before final merge
- prefer splitting source changes from generated or local-only artifacts
- treat `scratch/` as diagnostic provenance, not production API
- keep the new planning docs and RFCs as the map for what to formalize next

## 7. Related planning docs

- `POSTERIOR_STATE_UPDATE_RFC.md`
- `API_IMPLEMENTATION_QUEUE.md`
- `RINLA_API_SURFACE_INVENTORY.md`
- `RINLA_PARITY_GAP_INVENTORY.md`
- `EXTENSION_INTERVENTION_MAP.md`
