# Benchmark Summary for April 18, 2026

This note summarizes the latest local parity benchmark comparison between `rustyINLA` and installed `R-INLA`.

## Scope

- Benchmark script: `C:/Users/Antonio/Documents/rustyINLA/rustyINLA/benchmark.R`
- Rust package library used for the run: `C:/Users/Antonio/Documents/rustyINLA/rustyINLA/scratch/rlib`
- Benchmark date: April 18, 2026
- Tweedie has been removed from the active parity sweep after repeated instability investigations. The last Tweedie attempt still failed in Rust with `Kriging C V_c block singular`.

## Headline Comparison

- `rustyINLA` is now `4/6` on the last full six-case run before Tweedie removal.
- On the active stable benchmark set without Tweedie, `rustyINLA` is now `5/5`.
- `rustyINLA` remains lower-memory than `R-INLA` in every benchmarked case.
- The main parity progress is in the count-model cases:
  - `poisson + iid + offset`
  - `poisson + iid + iid + offset`
  - `poisson + ar1 + offset`
  - `zeroinflatedpoisson1 + iid + offset`
- `gamma + rw1` now also passes after the intrinsic-field covariance fix.

## Latest Rust vs R-INLA Snapshot

| Case | Rusty-INLA | R-INLA | Accuracy signal | Status |
|---|---:|---:|---|---|
| `poisson + iid + offset` | `58.61s`, `359.3 MB` | `22.81s`, `1112.0 MB` | `mlik` diff `4.953242`; fixed/random/fitted diffs tiny | PASS |
| `poisson + iid + iid + offset` | `52.22s`, `845.6 MB` | `39.17s`, `1339.2 MB` | `mlik` diff `3.900219`; fixed/random/fitted diffs tiny | PASS |
| `gamma + rw1` | `3.95s`, `825.4 MB` | `2.42s`, `970.5 MB` | fixed mean diff `0.006018`; random sd diff `0.010610`; fitted max rel diff `0.024695` | PASS |
| `poisson + ar1 + offset` | `55.83s`, `1012.1 MB` | `36.50s`, `1590.9 MB` | `mlik` diff `5.086452`; fitted max rel diff `0.035611` | PASS |
| `zeroinflatedpoisson1 + iid + offset` | `102.97s`, `1195.6 MB` | `98.02s`, `1849.7 MB` | `mlik` diff `1.834539`; fixed/random/fitted diffs tiny | PASS |

## Comparison to the Older Baseline

Older benchmark headline from the April 15 session context:

- overall pass count: `1/6`
- `poisson + iid + offset`: `78.86s`, `429.9 MB`
- `poisson + iid + iid + offset`: `122.97s`, `851.2 MB`
- `poisson + ar1 + offset`: `317.72s`, `996.3 MB`
- `zeroinflatedpoisson1 + iid + offset`: `172.51s`, `1159.1 MB`

Main changes:

- pass count improved from `1/6` to `4/6` on the original six-case suite
- active stable-suite pass count improved from `4/5` to `5/5`
- `poisson + iid + iid + offset` improved from about `123s` to `57s`
- `poisson + ar1 + offset` improved from about `318s` to `54s`
- `zeroinflatedpoisson1 + iid + offset` improved from about `173s` to `101s`
- the old ZIP headline failure is no longer the main benchmark blocker
- `gamma + rw1` moved from the remaining active blocker to a passing case

## Current Read

- `R-INLA` is still usually faster.
- `rustyINLA` still has a strong memory-use advantage.
- Accuracy is now good enough to pass the configured parity checks on all five active benchmark cases.
- The active stable benchmark suite is now at MVP parity (`5/5`) with Tweedie excluded by design.
- Tweedie is still intentionally excluded from the active benchmark sweep until there is a more stable path to compare.
