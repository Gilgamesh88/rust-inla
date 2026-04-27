# Phase 8 Roadmap

Phase 8 is the prior/control metadata and sequential-update phase. The guiding
rule is that reusable update states must represent old-data evidence, not a
posterior object blindly reused as a prior. That keeps original model priors
from being counted twice and gives each update mode a validation target: compare
against a full joint refit on old plus new data.

## Phase 8A: fixed+iid cross evidence state

Status: active; first checkpoint committed.

Scope:

- exactly one `iid` latent block
- same family
- same fixed-effect column names
- same `iid` covariate name
- allow new `iid` levels with zero old evidence
- carry dense fixed old-data evidence
- carry diagonal `iid` old-data evidence
- carry the fixed-`iid` cross evidence block
- keep `fixed_iid_gaussian_evidence` as a diagnostic diagonal-only comparator
- prefer `fixed_iid_cross_gaussian_evidence` for real validation

Acceptance gates before treating 8A as stable:

- synthetic old/new/joint refit with born `iid` levels
- actuarial VehBrand old/new/joint refit
- fixed means and SDs versus joint refit
- random means and SDs versus joint refit
- fitted values versus joint refit
- theta mode versus joint refit
- time and memory reported before any commit
- `cargo test -p inla_core`
- `tools/check-rust-workspace-win.ps1 -SkipPackageInstallFallback`
- `tools/run-phase8-validation.ps1`
- `tools/run_phase8_vehbrand_update.R`
- `tools/run-phase7a-validation.ps1`

Current 8A checkpoint:

- commit `9182252` adds the cross-block update-state mode
- `tests/posterior-state-fixed-iid-contract.R` is the focused contract gate
  for state metadata, born-level handling, diagonal-mode diagnostics, and
  rejection of changed family, fixed design, iid covariate, or latent model
- synthetic fixed+iid SD ratios improved from diagonal-only `0.182454` fixed
  and `0.244942` random to cross-block `0.993448` fixed and `0.993159` random
- VehBrand cross-block evidence reached theta drift `0.010236`, fitted new-row
  max relative drift `0.000186`, fixed SD ratio `0.992582`, and random SD ratio
  `0.974102` versus the joint refit

## Phase 8B: formal old-data evidence semantics

Status: active; semantics fields and contract checks started.

Goal: make the public contract precise.

Work:

- document that `rusty_update_state()` stores old-data evidence terms
- document that original model priors remain active in the new fit
- define state versioning and compatibility checks
- keep old-data evidence separate from warm starts and posterior summaries
- improve validation messages for changed family, changed fixed design, changed
  iid covariate name, and unsupported latent structures
- define what metadata must be present for a fit to be a valid update source

Current 8B checkpoint:

- `rusty_update_state()` declares `semantics$kind =
  "old_data_likelihood_evidence"`
- update metadata records the evidence semantics, prior policy, posterior-reuse
  policy, theta policy, state version, and source observation count
- signature mismatch errors now name the specific mismatch: family,
  fixed-effect columns, iid covariate, latent-block count, or latent model
- `tests/posterior-state-fixed-iid-contract.R` checks these semantics and
  mismatch errors

Expected remaining size: small, mostly documentation, API naming, and
validation cleanup.

## Phase 8C: theta-mixture evidence extraction

Goal: move from one local old-data evidence block at `theta_opt` to a small
mixture of old-data evidence blocks over the old CCD/theta support.

State shape:

```text
theta_s
weight_s
H_beta_beta(theta_s)
H_u_u(theta_s)
H_u_beta(theta_s)
h_beta(theta_s)
h_u(theta_s)
log_constant(theta_s)
```

Work:

- extract evidence blocks at supported theta points
- store weights and log constants
- keep memory bounded for large iid blocks
- preserve born-level expansion rules
- validate that the single-point mixture reproduces Phase 8A behavior

Expected size: medium.

## Phase 8D: theta-dependent objective

Goal: let the solver evaluate old-data evidence as a function of theta rather
than as one frozen block.

Work:

- add backend support for mixture evidence states
- include old evidence constants in the hyperparameter objective
- decide whether to select nearest support, interpolate, or log-sum-exp mixture
  contributions
- keep the one-iid restriction until the objective is benchmark-clean
- add diagnostics that show which theta support points are active

Expected size: large; this is the main solver extension toward a full
Bayes-like sequential approximation.

## Phase 8E: joint-refit validation gates

Goal: make the theta-mixture path credible.

Validation must compare update fits against joint refits for:

- fixed means and SDs
- random means and SDs
- fitted means and intervals
- theta mode and theta marginals
- tail intervals, not only means
- log marginal likelihood diagnostics where meaningful
- runtime and memory

Required cases:

- synthetic iid with old/new/joint split
- born-level iid split
- VehBrand actuarial split
- fixed-only GLM control case where state reuse should be rejected or skipped
- at least one stress case with multiple fixed columns

## Phase 8F: broader latent structures

Deferred until iid is stable.

Candidate extensions:

- `rw1`
- `rw2`
- `ar1`
- `ar2`

Open problems:

- changed graph structures
- intrinsic constraints and null spaces
- non-factor index expansion
- graph-compatible old evidence storage
- whether sequential updates should reject or approximate changed structure

This phase should not begin until Phase 8A through 8E are benchmark-clean for
the one-`iid` workflow.
