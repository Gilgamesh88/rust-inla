# Priority 1 Termwise Count Comparison on April 15, 2026

## Scope

This pass compared current Rust versus local `R-INLA-devel` term by term for:

1. `poisson + iid + iid + offset`
2. `poisson + ar1 + offset`
3. `zeroinflatedpoisson1 + iid + offset`

The comparison used:

- `scratch/compare_rust_vs_inla_termwise_count_models.R`
- `control.inla(strategy = "auto", int.strategy = "auto")`
- `control.compute = list(config = TRUE, dic = FALSE, waic = FALSE, cpo = FALSE)`
- `control.predictor = list(compute = TRUE)`
- `num.threads = 1`

The focus here was latent calibration and hyperparameter behavior, not exposure handling.

## Headline results

| Case | Rust elapsed | INLA elapsed | Rust mlik | INLA mlik | Rust intercept | INLA intercept |
|---|---:|---:|---:|---:|---:|---:|
| `poisson + iid + iid` | `119.63s` | `41.87s` | `-112094.63` | `-110900.32` | `-2.90263` | `-2.57347` |
| `poisson + ar1` | `253.70s` | `37.68s` | `-111982.97` | `-110767.73` | `-0.51819` | `-2.51714` |
| `zip1 + iid` | `189.24s` | `103.20s` | `-111866.86` | `-110686.97` | `-1.97747` | `-1.62203` |

## Case 1: `poisson + iid + iid + offset`

### Hyperparameters

- `prec(VehBrand)`
  - Rust internal `3.88733`
  - INLA internal `4.22391`
  - Rust external `48.78`
  - INLA external `68.30`
- `prec(Region)`
  - Rust internal `4.13718`
  - INLA internal `4.01135`
  - Rust external `62.63`
  - INLA external `55.22`

### Latent comparison

- `VehBrand` latent mean RMSE: `0.04847`
- `Region` latent mean RMSE: `0.05888`
- `VehBrand` latent SD average:
  - Rust `0.03406`
  - INLA `0.05023`
- `Region` latent SD average:
  - Rust `0.05328`
  - INLA `0.05651`

### Predictor comparison

- linear predictor RMSE: `0.34409`
- response-mean RMSE: `0.01260`
- plug-in log-likelihood
  - Rust `-112041.69`
  - INLA `-110828.06`

### Interpretation

- The two `iid` precision priors match INLA exactly here:
  - both are `loggamma(1, 5e-05)`
  - Rust's current generic prior formula is the same shape on the internal scale
- That means this case is **not** mainly a hyperprior-parity problem.
- The remaining gap is a count-path calibration gap:
  - intercept is too low by about `0.329`
  - the linear predictor is shifted downward overall
  - latent SDs are somewhat too tight

## Case 2: `poisson + ar1 + offset`

### Hyperparameters

- precision
  - Rust internal `2.59348`
  - INLA internal `2.32823`
  - Rust external `13.38`
  - INLA external `10.26`
- correlation parameter
  - Rust internal `6.87808`
  - INLA internal `0.67738`
  - If interpreted with the **INLA** transform, Rust would imply `rho = 0.99794`
  - Current Rust core actually uses `rho = tanh(theta)`, which gives `rho = 0.99999788`
  - INLA uses `rho = 2 * exp(theta) / (1 + exp(theta)) - 1 = tanh(theta / 2)`, which gives `rho = 0.32631`

### AR1 transform sanity check

- At the copied initial value `theta2 = 2`
  - INLA means `rho = 0.76159`
  - Rust currently means `rho = 0.96403`
- So even before optimization, Rust is **not** starting from the same AR1 correlation state as INLA.

### Prior comparison

- Rust currently gives the AR1 correlation theta the same generic prior shape as a precision term:
  - at Rust theta: `+6.82954`
- INLA uses `normal(0, 0.15)` on the internal AR1 correlation scale:
  - at Rust theta: `-3.54810`
  - at INLA theta: `-0.03441`

### Latent comparison

- latent mean average
  - Rust `-2.32416`
  - INLA `-0.03316`
- latent mean RMSE: `2.29335`
- latent mean max absolute error: `2.44121`
- latent SD average
  - Rust `0.87012`
  - INLA `0.26377`

### Predictor comparison

- linear predictor RMSE: `0.31841`
- response-mean RMSE: `0.01343`
- plug-in log-likelihood
  - Rust `-111970.58`
  - INLA `-110741.43`

### Interpretation

- `poisson + ar1` is a **separate structural failure mode**, not just the generic Poisson count gap.
- The main causes are already identifiable:
  1. Rust uses the wrong internal-to-`rho` transform.
  2. Rust uses the wrong prior family for the AR1 correlation hyperparameter.
- The huge intercept gap (`+1.999`) is compensating for an AR1 latent field that is badly miscalibrated.

## Case 3: `zeroinflatedpoisson1 + iid + offset`

### Hyperparameters

- latent precision
  - Rust internal `4.62420`
  - INLA internal `4.46609`
  - Rust external `101.92`
  - INLA external `87.02`
- zero-inflation probability
  - Rust internal `0.37462`
  - INLA internal `0.43088`
  - Rust external `0.59257`
  - INLA external `0.60608`

### Prior comparison

- The ZIP likelihood hyper prior does **not** match INLA:
  - Rust still uses the generic `theta - 5e-05 * exp(theta)` shape
  - INLA uses `gaussian(-1, 0.2)` on the internal logit-probability scale
- Numerically, though, the fitted zero-probability parameter is still fairly close:
  - Rust `0.5926`
  - INLA `0.6061`
- So the ZIP prior mismatch is real, but it does **not** look like the dominant source of the current ZIP parity gap.

### Latent comparison

- `VehBrand` latent mean RMSE: `0.05999`
- `VehBrand` latent SD average
  - Rust `0.03399`
  - INLA `0.04727`

### Predictor comparison

- linear predictor RMSE: `0.38361`
- response-mean RMSE: `0.01298`
- plug-in log-likelihood
  - Rust `-111860.03`
  - INLA `-110653.71`

### Interpretation

- ZIP now looks much more like the multi-`iid` Poisson case than like AR1.
- The extra zero-probability parameter is not wildly wrong.
- The main signal is again:
  - intercept too low by about `0.355`
  - latent block close in shape but somewhat too tight
  - overall predictor shifted downward

## Cross-case takeaway

### 1. `poisson + iid + iid` and `zip1 + iid` appear to share the same base count calibration gap

Common pattern:

- intercept too low by roughly `0.33` to `0.36`
- latent block mean RMSE only around `0.05` to `0.06`
- plug-in log-likelihood still worse by about `1200`
- linear predictor RMSE around `0.34` to `0.38`

Interpretation:

- the main remaining issue in these two cases is **not** exposure handling
- and it is **not** primarily the extra ZIP hyperparameter either
- it looks more like a shared Poisson-family latent/Laplace calibration problem

### 2. `poisson + ar1` is structurally broken for a more direct reason

The evidence now strongly points to:

1. wrong AR1 correlation transform
2. wrong AR1 correlation prior

This should be fixed before drawing broader conclusions from the AR1 benchmark.

### 3. Hyperprior mismatch is not a universal explanation

- It is **not** the explanation for `poisson + iid + iid`, because the relevant priors already match.
- It is a likely contributor for:
  - AR1 correlation
  - ZIP zero-probability
- But only the AR1 case currently shows a truly catastrophic hyperparameter mismatch.

## Best next coding actions

1. Fix Rust AR1 to use the same internal correlation transform as INLA:
   - `rho = tanh(theta / 2)`
   - not `rho = tanh(theta)`
2. Stop using the generic prior formula for all thetas.
   - keep it only for precision-like `loggamma(1, 5e-05)` terms
   - add model-specific priors for:
     - AR1 correlation: `normal(0, 0.15)` on the internal scale
     - ZIP zero-probability: `gaussian(-1, 0.2)` on the internal scale
3. After AR1 is corrected, return to the shared `poisson multi-iid` / `zip1 + iid` gap.
4. For that shared gap, the next useful Rust instrumentation is to expose the final Laplace decomposition terms directly:
   - sum log-likelihood
   - log prior
   - log-determinant pieces
   - latent quadratic form

That should make the remaining non-AR1 count mismatch easier to localize precisely.

## Update after AR1 transform and prior patch

After implementing:

1. AR1 correlation transform parity with INLA
   - `rho = tanh(theta / 2)`
2. model-specific priors in Rust for:
   - AR1 correlation: `normal(0, 0.15)`
   - ZIP zero-probability: `gaussian(-1, 0.2)`
   - keeping explicit loggamma precision priors for precision-like terms

the comparison script was rerun.

### What changed

#### `poisson + ar1`

- runtime dropped from about `253.70s` to `103.81s`
- intercept gap improved from about `+1.999` to about `-0.254`
- AR1 internal correlation theta moved from `6.878` to `1.262`
- AR1 correlation moved from essentially `rho ~= 0.998` toward `rho ~= 0.559`
- latent mean RMSE dropped from about `2.293` to about `0.112`
- latent SD RMSE dropped from about `0.606` to about `0.194`

Interpretation:

- the AR1-specific structural mismatch was real
- fixing the transform and prior removed the catastrophic AR1 failure mode
- `poisson + ar1` is still not at parity, but it is now much closer and no longer obviously broken for the same reason as before

#### `poisson + iid + iid`

- effectively unchanged

Interpretation:

- this confirms the shared multi-`iid` count gap is separate from the AR1-specific issue

#### `zip1 + iid`

- zero-probability parameter stayed in the same neighborhood
- plug-in log-likelihood improved slightly
- marginal log-likelihood did not materially improve

Interpretation:

- adding the correct ZIP prior was the right parity move
- but it did not unlock the main remaining ZIP gap by itself

### Refined next target

With AR1 no longer dominating the count debugging story, the next best move is now clearer:

1. expose the Rust Laplace decomposition terms directly
   - sum log-likelihood
   - log prior
   - `log|Q|`
   - `log|Q + W|`
   - latent quadratic form
2. compare those terms first on the user-priority count cases:
   - `zeroinflatedpoisson1 + iid + offset`
   - `poisson + iid + offset`
3. then extend the same decomposition comparison to:
   - `poisson + iid + iid + offset`
4. keep `poisson + ar1 + offset` as a lower-priority follow-up check after the shared `iid` count path improves

That should isolate the remaining shared count mismatch more directly than continuing to compare only final summaries.

## Update after ZIP/IID Laplace decomposition and start audit

A new focused script now exists for the current user-priority cases:

- `scratch/compare_rust_vs_inla_zip_iid_laplace_terms.R`

It does two things in one pass:

1. audits the relevant installed `INLA` hyper defaults directly from `inla.models()`
2. runs only:
   - `zeroinflatedpoisson1 + iid + offset`
   - `poisson + iid + offset`
   while collecting the new Rust Laplace decomposition terms

### Starting-value audit result

The installed `INLA` metadata confirms:

- `iid` precision initial = `4`
- `ar1` precision initial = `4`
- `ar1` correlation initial = `2`
- `zeroinflatedpoisson1` internal zero-probability initial = `-1`
- Poisson likelihood hyperparameter count = `0`

For the focused Rust runs:

- `zip + iid` used starts `(4, -1)` exactly
- `poisson + iid` used start `(4)` exactly

Interpretation:

- the current `zip + iid` and `poisson + iid` parity gap is **not** another start-value mismatch

### Focused case results

#### `zeroinflatedpoisson1 + iid + offset`

- intercept gap remains about `-0.359`
- internal ZIP theta is fairly close:
  - Rust `0.3644`
  - INLA `0.4309`
- external zero-probability is very close:
  - Rust `0.5901`
  - INLA `0.6061`
- latent `iid` block mean RMSE is about `0.0600`
- plug-in log-likelihood gap is still about `-1189.62`

Interpretation:

- the ZIP inflation parameter itself is no longer the obvious problem
- the remaining ZIP gap still looks like the same base count-calibration issue seen in the Poisson `iid` path

#### `poisson + iid + offset`

- intercept gap remains about `-0.302`
- Rust precision is still lower than INLA on the internal scale:
  - Rust `4.1558`
  - INLA `4.4651`
- latent `iid` block mean RMSE is about `0.0617`
- plug-in log-likelihood gap is still about `-1151.75`

Interpretation:

- this is the same shared count mismatch seen before, now isolated in the simpler single-`iid` Poisson case

### What the new Laplace decomposition says

For both focused cases, Rust now exposes:

- `sum_loglik`
- model and likelihood prior pieces
- determinant contribution
- latent/fixed quadratic penalties

The important pattern is:

- `zip + iid`
  - Rust `log_mlik - plugin_loglik ~= -24.15`
  - INLA `log_mlik - plugin_loglik ~= -33.26`
- `poisson + iid`
  - Rust `log_mlik - plugin_loglik ~= -25.69`
  - INLA `log_mlik - plugin_loglik ~= -28.71`

Interpretation:

- the **bulk** of the Rust-vs-INLA gap is still already present in the fitted mean / plug-in likelihood term
- only a small remainder is explained by the outer Laplace correction pieces
- so the next debugging target should move inward toward the latent mode / linear predictor calibration path, not back outward to starts or high-level prior plumbing

### Refined next target

The next best move is now:

1. expose the Rust final mode objects directly for these cases:
   - latent mode `x_hat`
   - fixed-effect mode `beta_hat`
   - mode-scale linear predictor `eta_hat`
2. use `INLA` config/mode internals to extract the corresponding mode-level quantities on the R side
3. compare those mode objects first for:
   - `zeroinflatedpoisson1 + iid + offset`
   - `poisson + iid + offset`

That should tell us whether the remaining parity loss is already in the inner mode solve / likelihood curvature path, which now looks more likely than another outer-hyper or start-value issue.

## Update after exact mode comparison for ZIP/IID

Rust now exposes the final optimizer-mode objects directly:

- latent mode `x_hat`
- fixed-effect mode `beta_hat`
- predictor mode `eta_hat`

A new focused script compares those exact Rust mode objects against the exact `INLA` mode vector, sliced via:

- `fit[['mode']][['x']]`
- `fit[['misc']][['configs']][['contents']]`

for:

- `zeroinflatedpoisson1 + iid + offset`
- `poisson + iid + offset`

### Exact mode results

#### `zeroinflatedpoisson1 + iid + offset`

- starting values still match exactly: `(4, -1)`
- mode theta remains close to the previous posterior summary comparison:
  - precision internal: Rust `4.6271`, INLA `4.4661`
  - ZIP theta internal: Rust `0.3644`, INLA `0.4309`
- fixed intercept mode gap is about `-0.3591`
- latent mode RMSE is about `0.0600`
- predictor-mode RMSE is about `0.3868`
- mode plug-in log-likelihood gap is still about `-1187.67`

#### `poisson + iid + offset`

- starting values still match exactly: `(4)`
- precision internal mode gap remains:
  - Rust `4.1558`
  - INLA `4.4651`
- fixed intercept mode gap is about `-0.3033`
- latent mode RMSE is about `0.0617`
- predictor-mode RMSE is about `0.3344`
- mode plug-in log-likelihood gap is still about `-1151.89`

### Interpretation

This is the strongest localization so far:

- the remaining `zip + iid` and `poisson + iid` parity loss is already present at the **exact inner mode**
- it is **not** primarily being created by:
  - hyperparameter initials
  - CCD mixing
  - outer Laplace correction terms

So the main suspect has now narrowed further to the inner mode solve path itself, especially one of:

1. count-likelihood gradient / curvature assembly
2. fixed-effect / latent coupling inside the Newton system
3. predictor-side construction used inside the mode solve

### Refined next target

The next best move is now:

1. expose Rust per-observation mode quantities used by the Newton solve:
   - predictor `eta`
   - gradient contribution
   - curvature contribution
2. extract whatever matching mode-scale information we can from `INLA` config objects
3. compare first for:
   - `zeroinflatedpoisson1 + iid + offset`
   - `poisson + iid + offset`

At this point, more work on starts, priors, or outer Laplace bookkeeping is unlikely to explain the bulk of the remaining gap.

## Update after per-observation mode-input audit and fixed-effect offset RHS fix

The per-observation mode-input comparison exposed two key facts for both

- `zeroinflatedpoisson1 + iid + offset`
- `poisson + iid + offset`

1. Rust likelihood derivatives were internally self-consistent at the returned mode.
2. The returned Rust mode still had enormous score residuals, while INLA's mode residuals were near numerical zero.

That pushed the main suspicion away from likelihood formulas and toward the fixed-effect / latent IRLS solve.

### New diagnosis

The fixed-effect Schur-complement solver was building the pseudo-response RHS with

- `W z`

instead of

- `W (z - offset)`

when an offset was present.

Because the offset is a known additive term in

- `eta = A x + X beta + offset`

the IRLS normal equations must solve for the unknown linear predictor part only:

- `A x + X beta`

Omitting the offset subtraction biased the fixed/random solve downward and explains the earlier

- intercept shift around `-0.30` to `-0.36`
- predictor RMSE around `0.33` to `0.39`
- huge fixed-score residuals around `6.7e3`

### Result after the fix

With the offset subtraction added inside the fixed-effect IRLS RHS, the two priority cases now line up very closely with `R-INLA-devel`.

#### `zeroinflatedpoisson1 + iid + offset`

- runtime dropped to about `79.1s` versus previous `244s` to `310s`
- log marginal likelihood:
  - Rust `-110677.1782`
  - INLA `-110686.9708`
- predictor-mode RMSE fell to about `1.21e-4`
- gradient RMSE versus INLA mode inputs fell to about `4.93e-6`
- fixed-score residual is now essentially zero: about `-6.14e-8`
- latent score residual RMSE is now about `9.89`, which is basically the same scale as INLA's own `9.79`
- inner mode solves almost stopped hitting the iteration cap:
  - `181` solves
  - only `3` max-iter hits
  - average about `4.48` iterations

#### `poisson + iid + offset`

- runtime dropped to about `33.8s`
- log marginal likelihood:
  - Rust `-111204.5528`
  - INLA `-111209.5031`
- predictor-mode RMSE fell to about `3.39e-5`
- gradient RMSE versus INLA mode inputs fell to about `1.61e-6`
- fixed-score residual is now essentially zero: about `2.26e-8`
- latent score residual RMSE is now about `9.87`, essentially the same scale as INLA's `9.91`
- inner mode solves now converge cleanly:
  - `112` solves
  - `0` max-iter hits
  - average about `4.01` iterations

### Interpretation

This looks like the main parity bug for the user-priority count cases.

- `zip + iid + offset` is now effectively calibrated to INLA at the mode level
- `poisson + iid + offset` is now effectively calibrated to INLA at the mode level
- the old large count gap was mostly a fixed-effect IRLS offset-handling bug inside the mode solve, not a start-value problem and not a raw likelihood-derivative problem

### Best next move

Carry the same comparison back to the shared lower-priority count cases next:

1. `poisson + iid + iid + offset`
2. recheck `zeroinflatedpoisson1 + iid + offset` with the broader termwise harness
3. only then revisit any remaining AR1 or no-fixed-effect issues

## Update after rerunning the broader termwise harness

The broader script `scratch/compare_rust_vs_inla_termwise_count_models.R` was rerun after the fixed-effect offset RHS fix, covering:

- `poisson + iid + iid + offset`
- `poisson + ar1 + offset`
- `zeroinflatedpoisson1 + iid + offset`

### Main confirmation

The same core mode-calibration fix also closes the previously shared multi-IID count issue.

#### `poisson + iid + iid + offset`

- intercept difference is now only about `+0.00173`
- hyperparameter internal differences are tiny:
  - `prec(VehBrand)` about `-8.88e-4`
  - `prec(Region)` about `-4.67e-3`
- latent posterior mean RMSE is now very small:
  - `VehBrand` about `7.25e-4`
  - `Region` about `9.31e-4`
- predictor RMSE is about `7.84e-4`
- plug-in log-likelihood now matches almost exactly:
  - Rust `-110828.0849`
  - INLA `-110828.0554`

So the old large multi-IID count mismatch was part of the same fixed-effect offset bug.

#### `zeroinflatedpoisson1 + iid + offset`

The broader harness confirms the focused ZIP result:

- intercept difference about `+8.79e-4`
- ZIP zero-probability internal theta difference about `-1.52e-4`
- predictor RMSE about `4.29e-4`
- plug-in log-likelihood still matches essentially exactly:
  - Rust `-110653.7491`
  - INLA `-110653.7117`

#### `poisson + ar1 + offset`

AR1 is no longer a catastrophic predictor-calibration failure:

- intercept difference about `+0.0488`
- predictor RMSE about `2.08e-4`
- plug-in log-likelihood is effectively identical:
  - Rust `-110741.4243`
  - INLA `-110741.4254`

But the AR1 correlation hyperparameter is still not aligned:

- Rust internal rho theta about `1.4657`
- INLA internal rho theta about `0.6773`

and the marginal likelihood still differs by about `6.68`, so AR1 now looks like a lower-priority hyper / prior / Laplace-correction issue rather than a mode-calibration issue.

### What still remains

For the user-priority ZIP / Poisson count cases, the main remaining differences are no longer in:

- exposure handling
- inner predictor calibration
- fixed intercept calibration
- plug-in likelihood at the fitted predictor

The remaining gap has moved to the uncertainty side:

- marginal likelihood (`mlik`) differences are now in the single digits to teens
- latent posterior SDs are still notably lower than INLA in several cases

For example in `poisson + iid + iid + offset`:

- mode / predictor alignment is excellent
- but latent SD averages are still lower than INLA for both IID blocks
- and `mlik` still differs by about `15.9`

So the next likely target is no longer the mode equations. It is the Gaussian/Laplace uncertainty correction path:

- latent SD calibration
- log-determinant terms
- CCD / marginal approximation effects

### Updated conclusion

The user-priority count calibration problem is now essentially solved at the mode / predictor level for:

1. `zeroinflatedpoisson1 + iid + offset`
2. `poisson + iid + offset`
3. `poisson + iid + iid + offset`

The next phase should focus on closing the remaining uncertainty / marginal-likelihood gap rather than continuing to debug exposure or inner mode construction.
