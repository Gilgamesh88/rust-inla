# External Reference Output Profile Comparison

Date: 2026-04-19

This note compares the external reference benchmark suite under the two
`rustyINLA` output contracts:

- `thin`
- `benchmark`

Scripts and outputs:

- [benchmark_external_reference_cases.R](/C:/Users/Antonio/Documents/rustyINLA/rustyINLA/scratch/benchmark_external_reference_cases.R)
- [external_reference_benchmark_thin_2026-04-19.csv](/C:/Users/Antonio/Documents/rustyINLA/rustyINLA/scratch/external_reference_benchmark_thin_2026-04-19.csv)
- [external_reference_benchmark_benchmark_2026-04-19.csv](/C:/Users/Antonio/Documents/rustyINLA/rustyINLA/scratch/external_reference_benchmark_benchmark_2026-04-19.csv)

## Headline

The external suite stayed stable:

- `thin`: `4/5` runnable cases passed
- `benchmark`: `4/5` runnable cases passed

The same public AR1 earthquake case is still the only failure, so the
benchmark-oriented output profile did not introduce a new accuracy regression.

## Per-Case Comparison

| Case | Thin pass | Benchmark pass | Thin mem | Benchmark mem | Delta mem | Thin time | Benchmark time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gamma + rw1` synthetic reference | PASS | PASS | `157.9 MB` | `160.1 MB` | `+2.2 MB` | `0.05s` | `0.04s` |
| `poisson + ar1` earthquake | FAIL | FAIL | `157.7 MB` | `162.2 MB` | `+4.5 MB` | `0.05s` | `0.03s` |
| `poisson + iid + iid + offset` NYC stops | PASS | PASS | `158.0 MB` | `169.3 MB` | `+11.3 MB` | `0.13s` | `0.13s` |
| `poisson + iid + offset` NC SIDS | PASS | PASS | `130.8 MB` | `135.4 MB` | `+4.6 MB` | `0.03s` | `0.03s` |
| `ZIP + iid + offset` synthetic reference | PASS | PASS | `157.9 MB` | `160.0 MB` | `+2.1 MB` | `0.05s` | `0.05s` |

## Aggregate Summary

- Average extra Rust memory from `benchmark` vs `thin`: `+4.94 MB`
- Average Rust runtime change from `benchmark` vs `thin`: about `-0.01 sec`
- Average Rust/R-INLA memory ratio under `thin`: `0.737`
- Average Rust/R-INLA memory ratio under `benchmark`: `0.759`

## Interpretation

Compared with the large internal actuarial benchmarks, the profile difference is
much smaller on the external reference suite. That is expected:

- these public datasets are much smaller
- the additional benchmark outputs, including `marginals.hyperpar`, do not
  dominate object size here

The practical takeaway is:

- the fairer output profile has negligible runtime cost on these small
  reference cases
- it increases memory only modestly
- `rustyINLA` still remains below `R-INLA` on memory in every external case

## What Changed Scientifically

The new `benchmark` profile now includes:

- `marginals.fixed`
- `marginals.random`
- `marginals.hyperpar`
- `summary.linear.predictor`
- `model.matrix`
- light `.args`

So the output contract is now closer to `R-INLA` on both the internal and
external benchmark tracks without moving into the expensive `misc$configs` /
full per-observation marginal regime.
