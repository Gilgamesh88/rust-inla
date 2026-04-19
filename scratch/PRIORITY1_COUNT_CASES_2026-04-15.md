# Priority 1 Extension: Remaining Count Cases on April 15, 2026

## Scope

This note extends the earlier `poisson + iid + offset` investigation to the remaining benchmark-matched count cases:

- `poisson + iid + iid + offset`
- `poisson + ar1 + offset`
- `zeroinflatedpoisson1 + iid + offset`

## Source inspection

- In `r-inla-devel/rinla/R/create.data.file.R`, the same response-side `E` path used by Poisson also includes:
  - `zeroinflatedpoisson0`
  - `zeroinflatedpoisson1`
  - `zeroinflatedpoisson2`
- Those families are handled in the block that builds `response <- cbind(ind, E, y.orig)`.
- This means the `E=` handling is not a special case for plain Poisson only; it also applies to the ZIP family used in the benchmark.
- The latent model choice (`iid` versus `ar1`, or one `iid` block versus two) does not change that response serialization path. It changes the latent Gaussian model and hyperparameter block, but not how exposure is passed to the likelihood.

## Empirical check on benchmark-matched cases

Using installed `INLA_25.10.19` with:

- `control.inla(strategy = "auto", int.strategy = "auto")`
- `control.compute = list(config = FALSE, dic = FALSE, waic = FALSE, cpo = FALSE)`
- `control.predictor = list(compute = TRUE)`
- `num.threads = 1`

Results:

| Case | Path | Elapsed | Intercept | Hyper 1 | Hyper 2 | mlik |
|---|---|---:|---:|---:|---:|---:|
| `poisson + iid + iid` | `offset(log(E))` | `39.21s` | `-2.573466772` | `72.21040210` | `56.70157413` | `-110900.318361` |
| `poisson + iid + iid` | `E=` | `32.57s` | `-2.573466773` | `72.21041022` | `56.70157230` | `-110900.318361` |
| `poisson + ar1` | `offset(log(E))` | `35.06s` | `-2.517136074` | `10.73161868` | `0.363386275` | `-110767.731974` |
| `poisson + ar1` | `E=` | `30.09s` | `-2.517136292` | `10.73156667` | `0.363391694` | `-110767.731956` |
| `zeroinflatedpoisson1 + iid` | `offset(log(E))` | `97.03s` | `-1.622027628` | `0.605816770` | `93.73748974` | `-110686.970837` |
| `zeroinflatedpoisson1 + iid` | `E=` | `91.83s` | `-1.622027925` | `0.605816792` | `93.73741500` | `-110686.970837` |

Observed differences (`offset - E`):

- `poisson + iid + iid`
  - elapsed: `+6.64s`
  - intercept: `+1.55e-09`
  - hyper 1: `-8.12e-06`
  - hyper 2: `+1.83e-06`
  - marginal log-likelihood: `-2.76e-08`
- `poisson + ar1`
  - elapsed: `+4.97s`
  - intercept: `+2.18e-07`
  - hyper 1: `+5.20e-05`
  - hyper 2: `-5.42e-06`
  - marginal log-likelihood: `-1.83e-05`
- `zeroinflatedpoisson1 + iid`
  - elapsed: `+5.20s`
  - intercept: `+2.97e-07`
  - hyper 1: `-2.18e-08`
  - hyper 2: `+7.47e-05`
  - marginal log-likelihood: `-8.64e-09`

## Interpretation

- Across all three remaining count benchmarks, `E=` and `offset(log(E))` are still numerically equivalent to practical precision.
- The `E=` path remains faster in all three cases, not just for `poisson + iid`.
- This strongly suggests that the offset-versus-exposure distinction is not the main reason Rust is missing parity on:
  - multi-block Poisson
  - Poisson AR1
  - ZIP
- The ZIP case is especially useful because it confirms that the response-side `E` shortcut extends to `zeroinflatedpoisson1`, not only to plain Poisson.

## Best immediate Priority 1 follow-up

The next count-parity comparison should focus on decomposition differences between Rust and `R-INLA-devel` for:

1. `poisson + iid + iid + offset`
2. `poisson + ar1 + offset`
3. `zeroinflatedpoisson1 + iid + offset`

In other words:

- the `E` route has now been checked across the benchmark-relevant count families
- the next likely sources of mismatch are latent-model calibration, hyperparameter optimization, and likelihood-specific implementation details
