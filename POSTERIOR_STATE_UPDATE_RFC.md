# Posterior State Update RFC

This note proposes a `rustyINLA`-native design for approximate posterior-state updating.

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
fit2 <- rusty_inla(
  ...,
  control.update = list(
    posterior_state = state,
    mode = "hyper"
  )
)
```

Possible update modes:

- `"warm_start"`: initialization only
- `"hyper"`: reuse hyperparameter approximation
- later maybe `"hyper_and_latent"` for restricted same-structure models

## 6. Version 1 scope

Version 1 should support only:

- same family
- same hyperparameter meaning
- same latent topology
- same fixed-effect interpretation
- approximate hyperparameter-state reuse
- optional warm-start reuse for latent mode

Version 1 should not support:

- changing from one family to another
- changing latent topology
- changing graph structure
- changing the meaning of fixed-effect columns
- arbitrary posterior-as-prior updates for the full latent field

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

1. expose constrained `control.mode`
2. expose internal-scale hyperparameter outputs cleanly
3. define the `posterior_state` S3 object
4. implement hyperparameter-only approximate state reuse
5. validate against full refits

## 13. Related documents

- [API_IMPLEMENTATION_QUEUE.md](API_IMPLEMENTATION_QUEUE.md)
- [RINLA_API_SURFACE_INVENTORY.md](RINLA_API_SURFACE_INVENTORY.md)
- [COVERAGE_EVALUATION_2026-04-19.md](COVERAGE_EVALUATION_2026-04-19.md)
- [RINLA_PARITY_GAP_INVENTORY.md](RINLA_PARITY_GAP_INVENTORY.md)
