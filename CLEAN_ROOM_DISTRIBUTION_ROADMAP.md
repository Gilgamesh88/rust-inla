# Phase 9: Clean-Room Actuarial Distribution Roadmap

This roadmap complements the Phase 8 rolling evidence work introduced in
commit `230b9c8` (`feat(phase8): add rolling iid evidence update MVP`) and the
recursive evidence graph roadmap introduced in commit `ecaf119`
(`docs(phase8): outline recursive evidence graph roadmap`).

Phase 8 improves how old-data likelihood evidence can be reused across
sequential fits. The recursive evidence graph then generalizes that idea from
one fixed-plus-`iid` block to block-sparse evidence over multiple `iid` blocks.
This distribution track improves the likelihood families that can use that
infrastructure, especially for actuarial frequency and severity models.

The goal is to absorb the statistical lessons from mature mixed-model tooling
without taking source-code dependencies or implementation expression from
external projects.

## Phase placement

This document is the working plan for Phase 9.

Phase 9 now owns the clean-room actuarial distribution expansion:

- Phase 9A: native `nbinomial2`
- Phase 9B: constant zero-inflated count families, especially
  `zeroinflatednbinomial2`
- Phase 9C: Tweedie density stabilization
- Phase 9D: zero-inflation predictor surface, including the existing
  `ZIP Type-2` style goal
- Phase 9E: Gamma severity validation and later hurdle severity design

Phase 10 remains the dynamic arbitrary-prior phase. The Phase 9 likelihood work
should define stable theta names, transforms, and compatibility metadata so
Phase 10 prior controls have a clean target.

## Clean-room boundary

Allowed inputs:

- public statistical formulas and papers
- package manuals, vignettes, and user-facing documentation
- independently derived derivatives and numerical contracts
- black-box output comparisons from optional reference packages
- hand-calculated unit tests and deterministic simulations

Disallowed inputs for implementation work:

- copying source code from AGPL/GPL projects
- translating source code line-by-line into Rust
- borrowing implementation-specific helper structure, names, or control flow
- making `glmmTMB`, `TMB`, or related compiled stacks runtime dependencies of
  `rustyINLA`

Practical validation rule:

- optional reference scripts may use `requireNamespace("glmmTMB", quietly = TRUE)`
  in `tools/` or `scratch/`
- production code must remain native Rust/R code in this repo
- reference packages are oracles for behavior, not implementation sources

## Phase 9A: negative binomial 2

Status: recommended first native likelihood addition after the Phase 8 MVP
gates are stable.

Why first:

- high actuarial value for overdispersed claim counts
- fits the current one-response, one-linear-predictor likelihood contract
- requires only one family hyperparameter
- has clean analytic gradient and curvature with respect to `eta`

Parameterization:

```text
Y ~ NB2(mu, phi)
mu = exp(eta)
phi > 0

E[Y] = mu
Var[Y] = mu + mu^2 / phi
```

Probability mass function:

```text
P(Y = y) =
  Gamma(y + phi) / (Gamma(phi) Gamma(y + 1))
  * (phi / (phi + mu))^phi
  * (mu / (phi + mu))^y
```

Log-likelihood:

```text
ell =
  lgamma(y + phi) - lgamma(phi) - lgamma(y + 1)
  + phi * log(phi / (phi + mu))
  + y * log(mu / (phi + mu))
```

Derivatives with respect to `eta = log(mu)`:

```text
d ell / d eta =
  y - (y + phi) * mu / (phi + mu)

-d2 ell / d eta2 =
  (y + phi) * mu * phi / (phi + mu)^2
```

Initial implementation scope:

- family string: `nbinomial2`
- internal likelihood hyperparameter: `theta_lik[0] = log(phi)`
- fixed-effect-only GLM path
- fixed plus one `iid` path
- optional black-box comparison against `glmmTMB(..., family = nbinom2)`
- no dispersion covariates in the first slice

Acceptance gates:

- hand-calculated PMF/log-PMF tests
- derivative checks against finite differences
- fixed-only MAP/Hessian comparison
- one-`iid` end-to-end comparison against a stable reference fit
- benchmark profile for time, memory, fixed effects, random effects, fitted
  values, and hyperparameter summary

## Phase 9B: constant zero-inflated count families

Status: extend current `zeroinflatedpoisson1` semantics before widening the
public API.

General mixture:

```text
pi = logistic(theta_zi)
f(y | mu, theta) = base count density

P(Y = 0) = pi + (1 - pi) * f(0 | mu, theta)
P(Y = y > 0) = (1 - pi) * f(y | mu, theta)
```

Log-likelihood:

```text
if y = 0:
  ell = log(pi + (1 - pi) * f0)

if y > 0:
  ell = log(1 - pi) + log f(y)
```

For ZIP:

```text
f0 = exp(-mu)
```

For ZINB2:

```text
f0 = (phi / (phi + mu))^phi
```

Recommended order:

1. keep `zeroinflatedpoisson1` as the constant-`pi` baseline
2. add `zeroinflatednbinomial2` after native `nbinomial2`
3. share numerical helpers for stable `log(pi + (1 - pi) * f0)`
4. add black-box comparisons for fixed-only and one-`iid` cases

Acceptance gates:

- exact zero and positive-count log-likelihood tests
- finite-difference checks for `eta`
- finite-difference checks for the zero-inflation hyperparameter
- regression tests proving ZIP behavior is unchanged
- ZINB2 comparison against the NB2 baseline when `pi` is near zero

## Phase 9C: Tweedie density stabilization

Status: replace the current experimental Tweedie path with a more explicit
compound Poisson-Gamma density strategy before including Tweedie in the stable
benchmark subset.

Useful insurance range:

```text
1 < power < 2
mu = exp(eta)
phi > 0

E[Y] = mu
Var[Y] = phi * mu^power
```

Compound Poisson-Gamma representation:

```text
Y = sum_{j=1}^N X_j
N ~ Poisson(lambda)
X_j ~ Gamma(alpha, scale = beta)

lambda = mu^(2 - power) / (phi * (2 - power))
alpha = (2 - power) / (power - 1)
beta = phi * (power - 1) * mu^(power - 1)
```

Zero mass:

```text
P(Y = 0) = exp(-lambda)
log P(Y = 0) = -lambda
```

Positive density:

```text
f(y) =
  sum_{n=1}^infinity
    P(N = n) * GammaDensity(y; shape = n * alpha, scale = beta)
```

Stable log-space form:

```text
log f(y) =
  logsumexp_n [
    -lambda + n * log(lambda) - lgamma(n + 1)
    + (n * alpha - 1) * log(y)
    - y / beta
    - n * alpha * log(beta)
    - lgamma(n * alpha)
  ]
```

Implementation requirements:

- use `theta_lik[0] = log(phi)`
- use `theta_lik[1]` mapped to `power = 1 + logistic(theta_lik[1])`
- evaluate the zero atom exactly
- evaluate positive densities with bounded log-sum-exp series
- report diagnostics for series truncation bounds and dropped tail mass
- keep saddlepoint approximations as diagnostics only until validated

Acceptance gates:

- exact zero-probability tests
- positive-density tests against independently generated high-precision values
- finite-difference checks for `eta`
- stress tests near `power -> 1` and `power -> 2`
- fixed-only and one-`iid` black-box comparisons
- explicit decision before re-adding Tweedie to the stable benchmark suite

## Phase 9D: zero-inflation predictor surface

Status: defer until constant zero-inflated families are stable.

Goal: introduce a second linear predictor for structural-zero probability:

```text
eta_count_i = x_i beta + A_i u + offset_i
eta_zero_i = z_i gamma
pi_i = logistic(eta_zero_i)
```

Possible public shape:

```text
rusty_inla(
  y ~ x + f(group, model = "iid"),
  data = data,
  family = "zeroinflatedpoisson2",
  control.zeroinflation = list(formula = ~ z1 + z2)
)
```

First slice:

- fixed effects only in the zero-inflation predictor
- no random effects in the zero-inflation predictor
- no offset in the zero-inflation predictor
- explicit rejection of unsupported surfaces

Open design questions:

- whether zero-inflation fixed effects should share fixed-effect prior controls
- how to name and return `summary.zeroinflation`
- how to store second-predictor design matrices in `internal.design`
- how Phase 8 update states should treat second-predictor evidence

## Phase 9E: Gamma severity and hurdle severity

Status: keep current Gamma severity as the positive-severity baseline.

Clean mathematical boundary:

- ordinary Gamma severity assumes positive responses
- zero claim outcomes belong to a frequency or hurdle component
- a hurdle Gamma model should be represented as two parts, not as ordinary
  Gamma with zeros silently accepted

Recommended order:

1. keep `gamma` stable for positive severities
2. improve validation on fixed-only and one-`iid` Gamma severity cases
3. after second-predictor infrastructure exists, define a two-part hurdle
   severity model explicitly

## Suggested execution order

1. Write optional black-box reference harnesses for current ZIP, Gamma, and
   experimental Tweedie behavior.
2. Implement native `nbinomial2`.
3. Add constant `zeroinflatednbinomial2`.
4. Replace Tweedie with the compound Poisson-Gamma log-sum-exp density path.
5. Design and implement a fixed-effect-only zero-inflation predictor surface.
6. Revisit dispersion predictors after the multi-predictor shape is stable.

## Relationship to Phase 8

The Phase 8 update-state work should remain family-compatible and conservative.

Rules:

- update states must record the family and family hyperparameter order
- changed families continue to be rejected
- new families must provide enough metadata for theta names and transformed
  hyperparameter summaries
- rolling evidence validation should first use families with stable local
  curvature: Poisson, Gaussian, Gamma, and then NB2
- Tweedie should not be used as a Phase 8 validation family until its density
  path is stable and benchmark-clean

## Relationship to recursive evidence graphs

The recursive evidence graph roadmap in commit `ecaf119` means future
likelihoods should be designed with evidence extraction in mind, not only
single-fit optimization.

For each new family, the implementation should define:

- a stable family signature and theta ordering
- transformed hyperparameter names for summaries and compatibility checks
- pointwise log-likelihood, gradient, and positive curvature with respect to
  the count or severity predictor
- enough local quadratic information to extract fixed, `iid`, fixed-`iid`, and
  eventually `iid`-`iid` evidence blocks
- rejection rules for families whose curvature or auxiliary inputs are not yet
  safe for recursive composition

Implications by Phase 9 slice:

- NB2 is the first candidate for recursive evidence validation after Poisson
  because it has a stable scalar predictor contract and analytic curvature.
- Constant zero-inflated families can join after their zero-mass mixture
  curvature is stable across sparse-zero and all-zero edge cases.
- Tweedie must stay out of recursive evidence validation until the positive
  density series and curvature behavior are stable over the actuarial range.
- Second-predictor zero-inflation surfaces require a future graph extension:
  the evidence graph needs separate nodes for the conditional predictor and
  the zero-inflation predictor, plus any cross-evidence that is retained.
