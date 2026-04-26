# Benchmark Summary for April 19, 2026

This note summarizes the latest local active-suite parity benchmark after the April 19 speed pass on the latent-solve assembly path.

## Scope

- Benchmark script: `C:/Users/Antonio/Documents/rustyINLA/rustyINLA/benchmark.R`
- Rust package library used for the run: `C:/Users/Antonio/Documents/rustyINLA/rustyINLA/scratch/rlib`
- Benchmark date: April 19, 2026
- Active benchmark output: `C:/Users/Antonio/Documents/rustyINLA/rustyINLA/scratch/benchmark_active_5_thin_2026-04-19_speedpass.csv`
- Tweedie remains excluded from the active parity sweep due to instability.

## Headline

- `rustyINLA` passes `5/5` cases on the active stable suite.
- `rustyINLA` remains lower-memory than `R-INLA` in every active benchmark case.
- After the latest speed pass:
  - `poisson + iid + iid + offset` is now faster than `R-INLA`
  - `zeroinflatedpoisson1 + iid + offset` remains faster than `R-INLA`
  - `poisson + iid + offset` and `poisson + ar1 + offset` improved further while keeping parity intact

## Latest Active-Suite Snapshot

| Case | Rusty-INLA | R-INLA | Accuracy signal | Status |
|---|---:|---:|---|---|
| `poisson + iid + offset` | `33.51s`, `393.4 MB` | `22.11s`, `1112.2 MB` | `mlik` diff `4.953235`; fixed/random/fitted diffs tiny | PASS |
| `poisson + iid + iid + offset` | `32.41s`, `871.7 MB` | `39.03s`, `1339.5 MB` | `mlik` diff `3.900219`; fixed/random/fitted diffs tiny | PASS |
| `gamma + rw1` | `2.52s`, `826.7 MB` | `2.30s`, `970.8 MB` | fixed mean diff `0.006018`; random sd diff `0.010610`; fitted max rel diff `0.024695` | PASS |
| `poisson + ar1 + offset` | `41.67s`, `1038.3 MB` | `37.41s`, `1591.1 MB` | `mlik` diff `5.325265`; fitted max rel diff `0.006221` | PASS |
| `zeroinflatedpoisson1 + iid + offset` | `73.95s`, `1221.9 MB` | `98.45s`, `1850.1 MB` | `mlik` diff `1.834541`; fixed/random/fitted diffs tiny | PASS |

## What Changed In The Speed Pass

The latest runtime gains came from targeted Rust-side hot-path cleanup in `problem/mod.rs`:

- split `A'WA` storage into direct diagonal storage plus off-diagonal sparse storage instead of always materializing a diagonal-heavy `HashMap`
- fused repeated observation-level accumulation loops in the fixed-effects latent solve
- removed dead fixed-effects accumulation work that was not used downstream

The timing diagnostics before these changes showed that sparse factorization time was already tiny, while most runtime was spent in repeated likelihood and `A`-assembly loops. The speed pass targeted that assembly-heavy portion directly.

## Current Read

- The active stable benchmark suite is still at MVP parity (`5/5`) with Tweedie excluded by design.
- Memory wins are still real even after the recent apples-to-apples output-profile work.
- The next engineering task is no longer core parity; it is repo cleanup and standards alignment for the now-stable code path.
