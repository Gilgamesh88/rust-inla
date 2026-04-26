# Benchmark Output Profile Comparison

Date: 2026-04-18

This note compares the active 5-case benchmark suite under two `rustyINLA`
output contracts:

- `thin`: current lightweight default
- `benchmark`: fairer parity-oriented output profile with:
  - `summary.linear.predictor`
  - `model.matrix`
  - `marginals.fixed`
  - `marginals.random`
  - light `.args`

The benchmark harness used:

- [benchmark.R](/C:/Users/Antonio/Documents/rustyINLA/rustyINLA/benchmark.R)

Saved CSV outputs:

- [benchmark_active_5_thin_2026-04-18.csv](/C:/Users/Antonio/Documents/rustyINLA/rustyINLA/scratch/benchmark_active_5_thin_2026-04-18.csv)
- [benchmark_active_5_benchmark_2026-04-18.csv](/C:/Users/Antonio/Documents/rustyINLA/rustyINLA/scratch/benchmark_active_5_benchmark_2026-04-18.csv)

## Headline Result

Both profiles pass the active stable suite:

- `thin`: `5/5` PASS
- `benchmark`: `5/5` PASS

The benchmark profile makes the memory comparison fairer, but the memory win
still remains in every active case.

## Per-Case Comparison

| Case | Thin time | Benchmark time | Delta time | Thin mem | Benchmark mem | Delta mem | R-INLA mem (thin run) | R-INLA mem (benchmark run) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gamma + rw1` | `2.85s` | `2.80s` | `-0.05s` | `826.3 MB` | `911.9 MB` | `+85.6 MB` | `970.5 MB` | `1054.7 MB` |
| `poisson + ar1 + offset` | `34.50s` | `34.44s` | `-0.06s` | `1037.9 MB` | `1184.7 MB` | `+146.8 MB` | `1590.9 MB` | `1692.6 MB` |
| `poisson + iid + offset` | `35.78s` | `35.94s` | `+0.16s` | `393.2 MB` | `427.4 MB` | `+34.2 MB` | `1112.0 MB` | `1176.2 MB` |
| `poisson + iid + iid + offset` | `45.54s` | `45.71s` | `+0.17s` | `871.5 MB` | `976.5 MB` | `+105.0 MB` | `1339.2 MB` | `1389.3 MB` |
| `ZIP + iid + offset` | `76.41s` | `76.32s` | `-0.09s` | `1221.5 MB` | `1409.8 MB` | `+188.3 MB` | `1849.7 MB` | `1973.3 MB` |

## Aggregate Summary

- Average extra Rust memory from `benchmark` vs `thin`: `+111.98 MB`
- Average Rust runtime change from `benchmark` vs `thin`: `+0.03 sec`
- Average Rust/R-INLA memory ratio under `thin`: `0.634`
- Average Rust/R-INLA memory ratio under `benchmark`: `0.669`

So the fairer output contract narrows the gap, but only modestly:

- memory ratio moves from about `63.4%` of R-INLA to about `66.9%`
- runtime is essentially unchanged

## Interpretation

This result is useful because it answers the earlier question directly:

- Yes, part of the original memory win came from returning a thinner object.
- But even after adding a fairer benchmark-oriented payload, `rustyINLA`
  remains materially lighter than `R-INLA` on every active benchmark case.

That suggests the memory advantage is a mix of:

- thinner output contract
- lower R-side overhead
- genuine backend-side efficiency

## What Still Explains the Remaining Gap

The `benchmark` profile still does **not** include the real `R-INLA` memory
walls:

- `misc$configs`-style configuration dumps
- full per-observation fitted marginals
- full per-observation linear predictor marginals
- full hyperparameter marginal objects
- richer log/debug/bookkeeping state

Those remain intentionally out of the default and benchmark contracts because
they scale poorly on actuarial-size datasets.

## Recommended Next Step

The next useful extension on this track is:

1. add `marginals.hyperpar` as an opt-in benchmark/debug feature
2. rerun the public external reference suite under `thin` and `benchmark`
3. keep full fitted-marginal/config dumps opt-in only

That would move us closer to scientific parity without stepping into the GB-class
output wall.
