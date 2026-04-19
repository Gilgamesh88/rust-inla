# External Examples for the Active Benchmark Models

Date: 2026-04-18

This note collects public INLA examples and book references that are close to the five active benchmark model combinations in `rustyINLA`.

The goal is not exact dataset replication. It is to anchor each benchmark case to external references for:

- likelihood and link semantics
- offset or exposure handling
- latent effect interpretation
- hyperparameter interpretation
- qualitative expectations for fitted values and random effects

Current active benchmark status in `rustyINLA`:

| Benchmark case | Current Rusty-INLA status |
| --- | --- |
| `poisson + iid + offset` | PASS |
| `poisson + iid + iid + offset` | PASS |
| `gamma + rw1` | PASS |
| `poisson + ar1 + offset` | PASS |
| `zeroinflatedpoisson1 + iid + offset` | PASS |

## 1. Poisson + IID + Offset

Closest public references:

- Official Poisson likelihood doc:
  - <https://inla.r-inla-download.org/r-inla.org/doc/likelihood/poisson.pdf>
  - The doc states that the Poisson mean is `lambda(eta) = E * exp(eta)` and gives a simulated example with `E`.
- `Bayesian inference with INLA`, Chapter 2:
  - <https://becarioprecario.bitbucket.io/inla-gitbook/ch-INLA.html>
  - North Carolina SIDS example with exposure `E = EXP74` and then an overdispersion term `f(CNTY.ID, model = "iid")`.
- `Bayesian inference with INLA`, Chapter 4:
  - <https://becarioprecario.bitbucket.io/inla-gitbook/ch-multilevel.html>
  - NYC stop-and-frisk count example with `offset = log((15/12) * past.arrests)` and `f(precinct, model = "iid")`.

Why this matches our case:

- same Poisson log-link
- same exposure or offset logic
- same unstructured Gaussian random effect

What to compare with our results:

- one latent precision hyperparameter for the `iid` block
- exposure can be represented either as `E` or `offset(log(E))`
- fitted means should be interpreted as exposure-adjusted counts
- the `iid` term should absorb extra-Poisson heterogeneity, not serial or structured dependence

## 2. Poisson + IID + IID + Offset

Closest public references:

- `Bayesian inference with INLA`, Chapter 4:
  - <https://becarioprecario.bitbucket.io/inla-gitbook/ch-multilevel.html>
  - The count-data section fits
    `stops ~ eth + f(precinct, model = "iid") + f(ID, model = "iid")`
    with an offset on the log scale.
- `Bayesian inference with INLA`, Chapter 3:
  - <https://becarioprecario.bitbucket.io/inla-gitbook/ch-mixed.html>
  - General mixed-effects discussion for multiple latent terms.

Why this matches our case:

- same Poisson plus offset structure
- two independent latent Gaussian random effects
- same interpretation of separate precision hyperparameters for separate unstructured components

What to compare with our results:

- two latent precision hyperparameters, one per `iid` block
- random effects should represent distinct grouping factors rather than one merged effect
- fitted means should remain stable relative to the one-`iid` model while uncertainty and residual structure improve

## 3. Gamma + RW1

Closest public references:

- Official Gamma likelihood doc:
  - <https://inla.r-inla-download.org/r-inla.org/doc/likelihood/gamma.pdf>
  - Documents the default log-link, the Gamma precision hyperparameter, and a regression example.
- `Bayesian inference with INLA`, Chapter 9:
  - <https://becarioprecario.bitbucket.io/inla-gitbook/ch-smoothing.html>
  - Shows how `rw1` is used as a smooth latent effect.
- `Bayesian inference with INLA`, Chapter 5:
  - <https://becarioprecario.bitbucket.io/inla-gitbook/ch-priors.html>
  - Discusses `rw1` prior scaling and precision interpretation.
- Highstat two-volume INLA guide:
  - <https://www.highstat.com/index.php/books2?catid=18&id=25&view=article>
  - Volume I explicitly includes Gamma GLMs and later time-series INLA chapters.

Important note:

- I did not find an openly accessible book or vignette example with the exact `gamma + rw1` pairing on the first pass.
- The comparison here is therefore an inference from two primary sources:
  - Gamma likelihood semantics from the official likelihood doc
  - RW1 smooth-term semantics from the INLA book chapters

Why this is still useful:

- it validates the expected model anatomy: one Gamma likelihood precision plus one RW1 precision
- it validates that the latent effect is a smooth, ordered effect on the log-mean scale

What to compare with our results:

- two hyperparameters: Gamma precision and RW1 precision
- positive fitted means under the log-link
- latent means should vary smoothly across the ordered index
- uncertainty should be finite and locally smooth, not explode across the whole field

## 4. Poisson + AR1 + Offset

Closest public references:

- Official AR1 latent-model doc:
  - <https://inla.r-inla-download.org/r-inla.org/doc/latent/ar1.pdf>
  - Includes a simulated Poisson-count example with exposure `E`.
- `Bayesian inference with INLA`, Chapter 8:
  - <https://becarioprecario.bitbucket.io/inla-gitbook/ch-temporal.html>
  - Earthquake counts example:
    `number ~ 1 + f(year, model = "ar1")`, `family = "poisson"`.
- Highstat guide, Volume I:
  - <https://www.highstat.com/index.php/books2?catid=18&id=25&view=article>
  - Chapter 14 is dedicated to time-series analysis in R-INLA.

Why this matches our case:

- same Poisson count likelihood
- same AR1 latent structure
- official AR1 doc also includes exposure `E`, which makes it especially close to our benchmark

What to compare with our results:

- two latent hyperparameters: AR1 precision and `rho`
- `rho` should usually be materially above zero when temporal persistence is present
- fitted trajectories should smooth local count noise while preserving serial dependence

## 5. Zero-Inflated Poisson Type 1 + IID + Offset

Closest public references:

- Official zero-inflated likelihood doc:
  - <https://inla.r-inla-download.org/r-inla.org/doc/likelihood/zeroinflated.pdf>
  - Defines Type 1 as
    `p * 1[y=0] + (1-p) * Poisson(y)`,
    which is exactly the family we benchmark as `zeroinflatedpoisson1`.
- Highstat guide, Volume II:
  - <https://www.highstat.com/index.php/books2?catid=18&id=25&view=article>
  - Chapter 18 is specifically "Zero-inflated models for count data in R-INLA".
- inlabru ZIP vignette:
  - <https://inlabru-org.github.io/inlabru/articles/zip_zap_models.html>
  - Uses `family = "zeroinflatedpoisson1"` with exposure `E = area`.

Important note:

- The inlabru vignette uses a spatial latent field instead of an `iid` random effect, so it is not the same latent structure.
- It is still useful as a public example of `zeroinflatedpoisson1` together with exposure handling.

What to compare with our results:

- one zero-inflation hyperparameter on the internal logit scale
- one latent precision hyperparameter for the `iid` effect
- exposure should still scale the Poisson mean component
- the model should explain extra zeros through the zero-inflation term, not by forcing the `iid` precision to pathological values

## Practical Comparison Checklist

For each active benchmark model, the most useful comparison questions are:

1. Does our family and latent-effect parameterization match the official INLA docs?
2. Does our offset or `E` handling match the documented INLA semantics?
3. Do we recover the same number and interpretation of hyperparameters as the external examples?
4. Are our fitted values and random-effect summaries qualitatively consistent with what those examples imply?
5. Are any remaining gaps only dataset-specific or approximation-specific, rather than structural mismatches?

## Source List

- R-INLA Poisson likelihood doc:
  - <https://inla.r-inla-download.org/r-inla.org/doc/likelihood/poisson.pdf>
- R-INLA Gamma likelihood doc:
  - <https://inla.r-inla-download.org/r-inla.org/doc/likelihood/gamma.pdf>
- R-INLA AR1 latent-model doc:
  - <https://inla.r-inla-download.org/r-inla.org/doc/latent/ar1.pdf>
- R-INLA zero-inflated likelihood doc:
  - <https://inla.r-inla-download.org/r-inla.org/doc/likelihood/zeroinflated.pdf>
- `Bayesian inference with INLA` online book:
  - <https://becarioprecario.bitbucket.io/inla-gitbook/>
- `Bayesian inference with INLA`, Chapter 2:
  - <https://becarioprecario.bitbucket.io/inla-gitbook/ch-INLA.html>
- `Bayesian inference with INLA`, Chapter 4:
  - <https://becarioprecario.bitbucket.io/inla-gitbook/ch-multilevel.html>
- `Bayesian inference with INLA`, Chapter 8:
  - <https://becarioprecario.bitbucket.io/inla-gitbook/ch-temporal.html>
- `Bayesian inference with INLA`, Chapter 9:
  - <https://becarioprecario.bitbucket.io/inla-gitbook/ch-smoothing.html>
- Highstat two-volume INLA guide:
  - <https://www.highstat.com/index.php/books2?catid=18&id=25&view=article>
- inlabru ZIP/ZAP vignette:
  - <https://inlabru-org.github.io/inlabru/articles/zip_zap_models.html>
