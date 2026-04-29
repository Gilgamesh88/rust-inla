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

Status: checkpointed in `4f1b42e`; compact CCD-support extraction implemented.

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

- current slice stores compact old-data evidence blocks in an explicit
  `theta_evidence` container over the old CCD/theta support
- each support point records normalized weight, log marginal likelihood,
  log weight, and the local Gaussian log constant
- the 8C slice stores the CCD-support container; the 8D opt-in mode can now
  consume it while the source-mode path remains available as a comparator
- extract evidence blocks at supported theta points
- store weights and log constants
- keep memory bounded for large iid blocks
- preserve born-level expansion rules
- validate that the single-point mixture reproduces Phase 8A behavior

Current 8C checkpoint candidate:

- `rusty_update_state()` emits a version-3 state with
  `semantics$theta_evidence_policy = "ccd_support_modes_not_integrated"` when
  CCD support is available
- `state$theta_evidence` stores:
  - `theta`
  - `weights`
  - `log_weights`
  - `log_unnormalized_weights`
  - `log_mlik`
  - `log_constants`
  - `H_beta_beta`
  - `h_beta`
  - `H_u_u_diag`
  - `h_u`
  - `H_u_beta`
- `state$theta_evidence$solver_status = "not_integrated"` describes the stored
  object itself; update-fit metadata records `"linear_1d_integrated"` when the
  8D solver path consumes it
- `tests/posterior-state-theta-evidence-shape.R` checks the container shape and
  verifies that removing the container leaves the active source-mode update
  numerically unchanged

Expected size: medium.

## Phase 8D: theta-dependent objective

Status: first opt-in implementation active.

Goal: let the solver evaluate old-data evidence as a function of theta rather
than as one frozen block.

Work:

- add backend support for mixture evidence states
- include old evidence constants in the hyperparameter objective
- first implementation uses one-dimensional linear interpolation across the
  old CCD/theta support
- keep the one-iid restriction until the objective is benchmark-clean
- add diagnostics that show which theta support points are active

Current 8D candidate:

- new opt-in mode: `fixed_iid_cross_theta_evidence`
- restricted to exactly one theta dimension, currently the one-`iid` Poisson
  split-update path
- existing `fixed_iid_cross_gaussian_evidence` remains the source-mode frozen
  block comparator
- `tests/posterior-state-theta-objective.R` validates the integrated path
- VehBrand pseudo-period benchmark improved theta drift versus joint from
  `0.010236` with frozen source-cross evidence to `0.003678` with
  theta-dependent evidence; fitted new-row max relative drift improved from
  `0.000186` to `0.000112`
- `tools/run_phase8_fremtpl_three_part_update.R` adds a three-part freMTPL
  pseudo-period benchmark. Refit-based sequential updating stays close to
  joint refits; naive rolling re-extraction drifts because it drops previously
  compressed evidence, while the first explicit composed-state operator brings
  the rolling path back near the refit-based update.

Expected size: large; this is the main solver extension toward a full
Bayes-like sequential approximation.

## Phase 8E: joint-refit validation gates

Status: started with explicit state composition diagnostics.

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
- freMTPL ordered three-part split with both refit-based and rolling
  composition diagnostics
- fixed-only GLM control case where state reuse should be rejected or skipped
- at least one stress case with multiple fixed columns

Current composition candidate:

- `rusty_compose_update_state(previous_state, fit)` extracts current-period
  likelihood evidence from `fit` and adds it to `previous_state`
- composition is support-by-support for the compact theta-evidence container,
  with one-dimensional linear interpolation of the previous state onto the new
  state's theta support
- synthetic rolling validation improved theta drift from `0.070368` with naive
  re-extraction to `0.009584` with composition, and fitted drift from
  `0.109865` to `0.006090`
- freMTPL three-part diagnostic improved part3 rolling theta drift from
  `0.260389` with naive re-extraction to `0.011807` with composition; fitted
  drift improved from `0.008549` to `0.000312`
- `tools/run_phase8_fremtpl_born_brand_update.R` isolates the lowest-exposure
  VehBrand (`B14`) so it is absent in period 1 and born in period 2. The update
  metadata records `B14` as a born level, assigns zero old evidence at entry,
  and the composed state adds positive period-2 evidence for that level.
- dormant factor `iid` levels are now carried from the update state into the
  next period's latent table even when they have zero current exposure; metadata
  records active, dormant, and factor-levels-carried sets
- `tools/run_phase8_fremtpl_four_part_dormant_brand_update.R` splits the same
  low-exposure VehBrand into periods 2 and 4, leaves it dormant in period 3,
  and compares born, dormant, and re-entry updates against joint refits.

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

## Phase 8G: recursive evidence graph for multiple iid blocks

Deferred until the one-`iid` MVP has been validated on real multi-period data.

Goal: generalize the Phase 8 update state from a fixed-plus-one-`iid` pair to
a recursive block-sparse evidence graph that can represent several simultaneous
factor `iid` random effects.

Theoretical frame:

```text
S_t =
  ComposeEvidenceGraphs(
    S_{t-1},
    ExtractEvidence(
      Fit(D_t | S_{t-1})
    )
  )
```

Here `S_t` is a compressed representation of cumulative likelihood evidence,
not a posterior object blindly reused as a prior. Each new batch contributes an
evidence increment, and composition is graph addition after level alignment.

Minimum extracted evidence from each batch:

- compatibility signature:
  - family
  - fixed-effect column names
  - latent block list
  - `iid` covariate names and model types
  - theta parameter names and order
- theta support:
  - `theta_s`
  - support weights
  - log constants needed for theta posterior comparison
- fixed node evidence:
  - `Delta H_beta_beta(theta_s)`
  - `Delta h_beta(theta_s)`
- each `iid` node evidence:
  - levels
  - `Delta H_kk_diag(theta_s)`
  - `Delta h_k(theta_s)`
- fixed-random edge evidence for each `iid` block:
  - `Delta H_k_beta(theta_s)`
- random-random edge evidence for each pair of `iid` blocks:
  - sparse `Delta H_ij(theta_s)` triplets
- level metadata:
  - active
  - born
  - dormant
  - re-entered
  - source rows and exposure when available

Evidence graph structure:

```text
nodes:
  beta
  u_1
  ...
  u_K

node terms:
  H_ii(theta), h_i(theta)

edges:
  beta -- u_k
  u_i  -- u_j

edge terms:
  H_ij(theta)
```

The current MVP is the smallest nontrivial graph:

```text
nodes:
  beta
  u_1

edge:
  beta -- u_1
```

For multiple iid blocks the same recursion should apply:

```text
G_t(theta) =
  Align(G_{t-1}(theta), Delta G_t(theta))
  +
  Delta G_t(theta)
```

Composition rules:

- align fixed columns exactly; reject changed fixed-effect design
- align each factor `iid` block by covariate name and level labels
- keep dormant levels and their old evidence
- add born levels with zero old evidence
- expand node terms for born/dormant levels
- expand fixed-random edges for born/dormant levels
- expand random-random sparse edges for all affected level pairs
- add node evidence, edge evidence, and log constants support-by-support

Implementation shape:

- R should keep the public state inspectable and explicit:
  - `signature`: family, fixed columns, latent block descriptors, theta names
  - `theta_support`: support points, weights, and constants
  - `blocks`: one fixed block plus one entry per factor `iid` block
  - `edges`: fixed-random and random-random evidence links
  - `level_metadata`: active, born, dormant, and re-entered levels
- Rust should own the validated numerical representation:
  - `EvidenceGraph`
  - `EvidenceNode`
  - `EvidenceEdge`
  - `ThetaEvidenceSlice`
  - sparse triplet or compressed sparse storage for random-random edges
- the R object should be easy to print, audit, serialize, and compare in tests
- the Rust object should be strict about dimensions, level alignment, and theta
  support compatibility

Suggested R state layout:

```text
state <- list(
  version = "phase8_graph_v1",
  signature = list(...),
  theta_support = list(points = ..., weights = ..., constants = ...),
  blocks = list(
    beta = list(kind = "fixed", names = ..., H = ..., h = ...),
    VehBrand = list(kind = "iid", levels = ..., H_diag = ..., h = ...),
    Region = list(kind = "iid", levels = ..., H_diag = ..., h = ...)
  ),
  edges = list(
    list(from = "beta", to = "VehBrand", H = ...),
    list(from = "beta", to = "Region", H = ...),
    list(from = "VehBrand", to = "Region", H_sparse = ...)
  ),
  level_metadata = list(...)
)
```

Suggested Rust structure:

```text
struct EvidenceGraph {
    signature: EvidenceSignature,
    theta_slices: Vec<ThetaEvidenceSlice>,
    blocks: Vec<EvidenceBlock>,
    edges: Vec<EvidenceEdge>,
}

struct ThetaEvidenceSlice {
    theta: Vec<f64>,
    weight: f64,
    log_constant: f64,
}

enum EvidenceBlock {
    Fixed { names: Vec<String>, h: Vec<f64>, hessian: DenseMatrix },
    Iid { covariate: String, levels: Vec<String>, h: Vec<f64>, h_diag: Vec<f64> },
}

enum EvidenceEdge {
    FixedIid { fixed: BlockId, iid: BlockId, hessian: DenseMatrix },
    IidIid { left: BlockId, right: BlockId, hessian: SparseMatrix },
}
```

The recursion should be implemented as one composition operator rather than as
model-specific branches:

```text
compose(previous_state, fitted_batch):
  increment = extract_evidence(fitted_batch)
  aligned_previous, aligned_increment =
    align_graphs(previous_state.graph, increment.graph)
  return add_graphs(aligned_previous, aligned_increment)
```

This makes "two random effects" the first validation target, not a permanent
special case. If the graph abstraction is right, a third `iid` block should be
more bookkeeping and validation, not a new mathematical design.

Recommended implementation path:

1. define the R state shape as human-readable `blocks` and `edges`
2. define Rust typed structs for a block-sparse `EvidenceGraph`
3. prove the graph abstraction reproduces the current one-`iid` MVP
4. enable exactly two `iid` blocks as the first validation target
5. validate two-block updates against joint refits before allowing arbitrary K

Key warning:

- two `iid` blocks require the random-random cross edge `H_12`
- three or more `iid` blocks are not mathematically different, but they reveal
  the general graph structure and require careful sparse level alignment
- ignoring cross edges can make posterior uncertainty lie; this is the same
  lesson learned from the fixed-random cross block in Phase 8A
