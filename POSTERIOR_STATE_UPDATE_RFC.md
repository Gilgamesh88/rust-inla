# Posterior State Update RFC

This note proposes a `rustyINLA`-native design for approximate posterior-state updating.

The staged implementation roadmap is tracked in
[PHASE8_ROADMAP.md](PHASE8_ROADMAP.md). In short: Phase 8A stabilizes the
one-`iid` fixed-cross evidence state, Phase 8B formalizes old-data evidence
semantics, Phase 8C and 8D build theta-mixture evidence and a theta-dependent
objective, Phase 8E validates against joint refits, and Phase 8F defers broader
latent structures.

The core idea is:

- do not try to copy `R-INLA` as-is here
- do not pretend that warm starts are the same thing as Bayesian updating
- do not promise exact posterior reuse when the approximation state does not support that claim

Instead, define a clean, explicit state object that can be reused safely for a restricted class of updates.

## 1. Why this RFC exists

`R-INLA` already supports:

- custom hyperpriors
- warm starts and mode reuse through `control.mode(...)`
- reruns from previous fits through `inla.rerun(...)`

But that is not the same as a general "take posterior A and use it as prior B" workflow.

The main reason is structural:

- the posterior approximation is not just one independent prior per hyperparameter
- there are dependencies between hyperparameters
- there are dependencies between latent state and hyperparameters
- intrinsic-model constraints matter
- the quality of the approximation depends on the local Laplace / CCD construction used for the full model fit

So a general posterior-to-prior update interface needs to be explicit about what is preserved and what is not.

## 2. What problem we are trying to solve

We want a useful workflow for:

- repeated model refits on the same structural model
- new data arriving over time
- updating a previous fit without starting from scratch
- carrying forward useful information from previous fits

We do **not** want to claim:

- exact sequential Bayes for arbitrary model changes
- exact posterior reuse for arbitrary latent-field changes
- full equivalence to rerunning the complete model from the union of old and new data

## 3. Design principles

### Principle 1: be honest about approximation

The feature should be named and documented as:

- state reuse
- approximate update
- posterior-state initialization

and not as exact posterior continuation unless we can prove that.

### Principle 2: separate warm starts from prior updates

These are different things:

- warm start: use previous mode as initialization
- prior update: use previous approximation to change the next prior

The API should expose them separately.

### Principle 3: start with hyperparameters, not full latent fields

The safest first version is a hyperparameter-state update, because:

- it is lower-dimensional
- it is closer to the CCD/Laplace approximation we already export
- it is easier to validate

The first Phase 8 experiment narrows this further to one `iid` latent block:
reuse the previous fit's internal log-precision posterior as a Gaussian prior
for the next fit's `iid` precision. This is intentionally not full latent-state
reuse.

### Principle 4: require same model structure first

Version 1 should only support updates where these are unchanged:

- family
- latent model topology
- fixed-effect design meaning
- hyperparameter meaning and ordering

That restriction makes the feature much safer.

## 4. Proposed state object

Introduce a new explicit object, for example:

- `posterior_state`

Initial contents for version 1:

- `family`
- `latent_model_signature`
- `fixed_effect_signature`
- `theta_mode`
- `theta_cov`
- `theta_support`
- `theta_log_mlik`
- normalized CCD weights
- prior metadata used in the original fit
- engine version / package version
- internal-scale conventions

Optional later fields:

- latent mode summary
- selected latent covariance summaries
- fitted-summary metadata

Fields intentionally out of scope for version 1:

- full observation-level marginals
- full latent posterior marginals
- arbitrary posterior samples
- arbitrary `misc` dump

## 5. Proposed user-facing API shape

### A. Warm start only

Example shape:

```r
rusty_inla(
  ...,
  control.mode = list(
    result = previous_fit,
    restart = TRUE
  )
)
```

Purpose:

- reuse `theta` and optionally latent mode as initial values
- no claim of posterior-to-prior updating

### B. Extract a reusable state

Example shape:

```r
state <- rusty_posterior_state(fit)
```

Purpose:

- create an explicit portable approximation object

### C. Use a previous state as an approximate prior/state initializer

Example shape:

```r
state <- rusty_posterior_state(fit1, scope = "iid_hyper")

fit2 <- rusty_inla(
  ...,
  control.update = list(
    posterior_state = state,
    mode = "iid_hyper_gaussian"
  )
)
```

Possible update modes:

- `"warm_start"`: initialization only
- `"iid_hyper_gaussian"`: Phase 8 experimental mode; replace the default
  prior for the first `iid` log-precision hyperparameter with a Gaussian
  approximation to the previous posterior on the internal theta scale
- `"fixed_iid_gaussian_evidence"`: Phase 8 experimental mode; reuse a
  `rusty_update_state()` object containing dense fixed-effect old-data
  evidence and diagonal `iid` old-data evidence; retained as a diagnostic
  comparator
- `"fixed_iid_cross_gaussian_evidence"`: Phase 8 experimental mode; reuse a
  `rusty_update_state()` object containing dense fixed-effect evidence,
  diagonal `iid` evidence, and the fixed-`iid` cross block
- later maybe `"hyper"` for broader hyperparameter-state reuse
- later maybe `"hyper_and_latent"` for restricted same-structure models

## 6. Version 1 scope

The hyperparameter-only version supports only:

- one `iid` latent block
- same family
- same fixed-effect column names and interpretation
- same `iid` covariate name
- new observed levels of that `iid` covariate may appear, because the current
  state reuses only the shared precision hyperparameter, not per-level latent
  means
- approximate reuse of the `iid` log-precision posterior only
- validation against a full refit on old plus new data

The fixed-plus-`iid` evidence version supports only:

- one `iid` latent block
- same family
- same fixed-effect column names and interpretation
- same `iid` covariate name
- dense fixed-effect Gaussian old-data evidence
- diagonal `iid` level Gaussian old-data evidence
- fixed-`iid` cross Gaussian old-data evidence in the cross-block mode
- new observed `iid` levels, expanded with zero old evidence
- original model priors in the new fit, plus old-data evidence factors
- explicit state semantics:
  `old_data_likelihood_evidence`, `not_posterior_as_prior`, and
  `original_model_priors_remain_active_in_update`

These versions should not support:

- changing from one family to another
- changing latent topology
- changing graph structure
- changing the meaning of fixed-effect columns
- arbitrary posterior-as-prior updates for the full latent field
- joint hyperparameter covariance beyond the one-dimensional `iid` precision
  approximation

## 7. Internal implementation path

### Phase 1: public warm-start parity

Add a constrained public `control.mode`-style interface:

- `result`
- `theta`
- maybe `x`
- `restart`
- `fixed`

This gives us a clean public base.

### Phase 2: extract hyperparameter state

Add a helper that serializes the existing hyperparameter approximation:

- theta mode
- CCD support
- CCD weights
- internal-scale metadata

### Phase 3: re-use hyperparameter state

Allow the next run to:

- initialize from the previous state
- optionally build a prior approximation from the previous hyperparameter state

This should be described as an approximation layer, not exact Bayes.

Current experimental implementation:

- `rusty_posterior_state(fit, scope = "iid_hyper")` extracts a one-dimensional
  Gaussian approximation to the previous `iid` log-precision posterior from the
  internal CCD grid
- `rusty_inla(..., control.update = list(posterior_state = state,
  mode = "iid_hyper_gaussian"))` applies that Gaussian approximation as the
  next model's `iid` log-precision prior
- the implementation rejects non-`iid` latent structures, family changes,
  fixed-effect signature changes, and `iid` covariate-name changes
- because no latent means are reused, a newly observed `iid` level enters with
  the usual prior mean `0` and the carried-forward precision prior; the new
  level can move away from `0` when the new data provide enough information
- `rusty_update_state(fit, scope = "fixed_iid_gaussian")` extracts a reusable
  Gaussian evidence object with dense fixed-effect precision/linear terms,
  diagonal `iid` precision/linear terms, and fixed-`iid` cross precision terms
- the state object records versioned old-data evidence semantics so update
  fits can distinguish evidence reuse from posterior-as-prior reuse
- `rusty_inla(..., control.update = list(state = state,
  mode = "fixed_iid_cross_gaussian_evidence"))` adds those old-data evidence
  factors to the new fit while keeping the original model priors
- `mode = "fixed_iid_gaussian_evidence"` remains available as a diagonal-only
  diagnostic comparator, and should not be treated as the preferred update path
- for levels absent from the old state but present in the new data, the R
  bridge expands the `iid` evidence with precision `0` and linear term `0`
- signature validation reports the concrete mismatch, such as changed family,
  changed fixed-effect columns, changed iid covariate name, or non-`iid` latent
  model
- the first Phase 8C extraction slice adds `state$theta_evidence`, a compact
  container over the old CCD/theta support with normalized weights, log
  marginal likelihoods, local Gaussian log constants, and the fixed/`iid`
  evidence blocks; it is marked `solver_status = "not_integrated"` because
  update fits still use the Phase 8A source-mode fields until the
  theta-dependent objective lands

### Phase 4: restricted latent-state reuse

Only after the above is validated should we consider:

- same-dimension latent-state reuse
- constrained intrinsic-model state reuse
- same-design linear predictor reuse

This phase is much riskier.

## 8. Validation strategy

The feature is only credible if we validate it against full refits.

For version 1, validate on:

- same model, different random seed / same data
- same model, small additional data batch
- same model, same structure, repeated fit from prior state

For each case compare:

- theta mode
- theta marginals
- fixed effects
- fitted values
- log marginal likelihood

Acceptance rule:

- approximate update must be close to the corresponding full refit
- if not, the feature should fall back to warm-start-only behavior

The first focused regression is
`tests/posterior-state-iid-experimental.R`, wrapped by
`tools/run-phase8-validation.ps1`. It compares an old fit, a new-data default
fit, a new-data posterior-state update, and a full joint refit. The initial
deterministic case shows the posterior-state update pulling the `iid`
log-precision mode toward the previous posterior and much closer to the full
joint fit than the new-data-only fit.

The focused regression also includes a born-level diagnostic: the old fit sees
only groups `A` and `B`, while the update fit sees a new group `C`. Since the
current state carries only the `iid` precision posterior, group `C` is not
fixed to zero. It starts with a zero-mean iid prior and updates from its own
data; in the deterministic diagnostic its posterior mean moved to about
`1.10` on the log scale. If the previous precision posterior is very tight,
new levels will shrink strongly toward zero unless the new data are strong
enough to overcome that prior.

The fixed-plus-`iid` evidence regression is
`tests/posterior-state-fixed-iid-evidence.R`, also wrapped by
`tools/run-phase8-validation.ps1`. It fits old data with groups `1` through
`4`, updates on new data that include a born group `5`, and compares against a
full joint refit. The current deterministic run keeps the diagonal-only mode
as a comparator and treats the cross-block mode as the real update target. In
that run, fixed-effect maximum absolute drift moved from `0.182334` for
new-data-only to `0.140508` for diagonal evidence and `0.002104` for
cross-block evidence. Random-effect maximum absolute drift moved from
`0.463243` to `0.181983` to `0.004527`, and fitted new-row maximum relative
drift moved from `0.308039` to `0.048803` to `0.003383`. Minimum posterior SD
ratios versus the joint refit moved from `0.182454` to `0.993448` for fixed
effects and from `0.244942` to `0.993159` for random effects. Those SD ratios
are part of the gate because diagonal evidence can make the approximation too
narrow or internally inconsistent.

The first actuarial-scale diagnostic is
`tools/run_phase8_vehbrand_update.R`. It uses `freMTPL2freq`, splits the data
into an old first two-thirds and a new final one-third by `IDpol`, fits
`ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid")`, then
compares:

- new data only with default priors
- new data with the old `iid` posterior state as prior
- new data with diagonal fixed-plus-`iid` Gaussian evidence
- new data with cross-block fixed-plus-`iid` Gaussian evidence
- diagnostic point-offset fits that reuse old fixed and/or VehBrand posterior
  means as offsets
- a full joint refit on old plus new data

This is a pseudo-period split because `freMTPL2freq` does not expose an
explicit time variable in the current benchmark setup. It is useful for
actuarial scale and stable rating-factor semantics, but it is not a true
calendar-year update. The current split has no newly born `VehBrand` level.

The current run improved the internal `iid` log-precision distance to the
joint refit from `1.452517` for new-data-only to `0.345385` for the
hyperparameter-only update and `0.010236` for the cross-block fixed-plus-`iid`
evidence update. Fitted new-row relative drift improved from `0.026080` to
`0.022881` with the hyperparameter update and `0.000186` with cross-block
evidence. Minimum SD ratios versus the joint refit were `0.992582` for fixed
effects and `0.974102` for random effects in the cross-block row. The
diagonal-only row is now explicitly diagnostic: in this VehBrand split it drove
the log-precision far from the joint fit and narrowed SDs, which is exactly the
failure mode the cross block fixes. Point-offset rows remain diagnostics only:
they can improve fitted values but treat old posterior means as known offsets
and do not carry uncertainty.

## 9. Why this is easier in `rustyINLA` than in current R-INLA practice

Because we still control the public surface.

We can design:

- an explicit state object
- explicit guarantees
- explicit restrictions
- explicit fallback behavior

instead of overloading:

- priors
- warm starts
- reruns

into something they were not originally designed to be.

## 10. Recommendation on timing

This is the right moment to design the feature, but not the right moment to implement the full version.

Recommended order:

1. finish the current MVP stabilization
2. add constrained public `control.mode`
3. add metadata and internal hyperparameter outputs needed for state extraction
4. then implement `posterior_state` version 1 for hyperparameters only

That order keeps us from designing in the dark while also avoiding premature overengineering.

## 11. Recommendation on package scope before public release

We do not need full posterior-state updating before a first public package release.

What we do need before public release is:

- honest supported-scope documentation
- stable benchmark-backed subset
- a clear statement that sequential state reuse is planned, not finished

On families:

- we do not need many more families just to make the package public
- we do need at least one more obvious next extension in the roadmap, probably `rw2` or one additional GLM-like family

So the recommended release posture is:

- release the package once the current supported subset and API are honestly documented
- keep posterior-state updating as a planned differentiator
- avoid delaying release just to inflate family count

## 12. Best next engineering steps

1. Phase 8A: stabilize the committed one-`iid` fixed-cross evidence state and
   keep diagonal evidence as a diagnostic comparator only.
2. Phase 8B: formalize old-data evidence semantics so state extraction,
   validation, and docs all say the same thing.
3. Phase 8C: extract theta-mixture evidence blocks over the old CCD/theta
   support.
4. Phase 8D: add a theta-dependent objective with old-evidence log constants.
5. Phase 8E: validate theta-mixture updates against joint refits, including
   means, SDs, theta marginals, tails, time, and memory.
6. Phase 8F: defer broader latent structures until the one-`iid` path is
   benchmark-clean.

## 13. Related documents

- [PHASE8_ROADMAP.md](PHASE8_ROADMAP.md)
- [API_IMPLEMENTATION_QUEUE.md](API_IMPLEMENTATION_QUEUE.md)
- [RINLA_API_SURFACE_INVENTORY.md](RINLA_API_SURFACE_INVENTORY.md)
- [COVERAGE_EVALUATION_2026-04-19.md](COVERAGE_EVALUATION_2026-04-19.md)
- [RINLA_PARITY_GAP_INVENTORY.md](RINLA_PARITY_GAP_INVENTORY.md)
