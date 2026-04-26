# External Example Benchmarking Guide

This guide turns the external-example notes into a practical benchmarking workflow.

It answers:

- which public examples are closest to the current supported subset
- which ones are exact public matches versus synthetic fallback cases
- how to benchmark `rustyINLA` against those examples
- what to compare for accuracy, speed, and memory

For the underlying reference list, see [scratch/EXTERNAL_EXAMPLES_FOR_ACTIVE_BENCHMARK_MODELS_2026-04-18.md](scratch/EXTERNAL_EXAMPLES_FOR_ACTIVE_BENCHMARK_MODELS_2026-04-18.md). For the runnable external harness, see [scratch/benchmark_external_reference_cases.R](scratch/benchmark_external_reference_cases.R).

## 1. Why use external examples

The internal benchmark suite is the right source of truth for current MVP parity.

The external examples serve a different purpose:

- they anchor our supported subset to public INLA references
- they make benchmarking easier to explain to other contributors
- they help us validate that our parameterization matches published INLA semantics
- they provide a bridge from "our actuarial benchmark" to "recognizable INLA examples"

## 2. Example map

| Model combination | Closest public example | Match quality | Current external harness source |
| --- | --- | --- | --- |
| `poisson + iid + offset` | NC SIDS / Poisson exposure examples | exact public structural match | `spData::nc.sids` |
| `poisson + iid + iid + offset` | NYC stop-and-frisk multilevel count example | exact public structural match | `frisk_with_noise.dat` |
| `gaussian + rw2` | LIDAR smoothing example from the INLA GitBook | exact public structural match | `SemiPar::lidar` |
| `poisson + ar1` | Earthquake temporal count example | exact public structural match | `MixtureInf::earthquake` |
| `fixed-effect-only GLMs` | uploaded-suite GLM surfaces after supported-column filtering | curated supported-subset match | `tools/run_supported_subset_validation.R` |
| `gaussian + ar2` | no clean first-pass public exact dataset match | synthetic exact-family reference | synthetic Gaussian AR2 reference |
| `gaussian + multiple fixed effects + iid + offset` | no clean public exact beta-heavy benchmark was prioritized yet | synthetic exact-family reference | synthetic Gaussian multi-fixed-effect reference |
| `gamma + rw1` | Gamma likelihood doc + RW1 smoothing chapters | public semantic match, not first-pass exact dataset match | synthetic exact-family reference |
| `zeroinflatedpoisson1 + iid + offset` | official ZIP1 docs + inlabru ZIP example | public semantic match, not first-pass exact dataset match | synthetic exact-family reference |

## 3. Exact public matches vs synthetic reference cases

### Exact public structural matches

These are the best examples for public reproducibility:

- `poisson + iid + offset`
- `poisson + iid + iid + offset`
- `gaussian + rw2`
- `poisson + ar1`

Use these when we want to show:

- our family/latent specification matches public INLA examples directly
- our results can be compared on a public dataset with the same model structure

### Synthetic exact-family reference cases

These are still useful, but should be described honestly:

- `gamma + rw1`
- `gaussian + ar2`
- `gaussian + multiple fixed effects + iid + offset`
- `zeroinflatedpoisson1 + iid + offset`

Use these when we want to show:

- exact-family semantics
- current engine correctness on the intended structure
- reproducible parity checks even when a clean public exact dataset was not found

The right wording is:

- "synthetic exact-family benchmark"

not:

- "public example replication"

## 4. How to benchmark against the examples

Use the external harness:

- [scratch/benchmark_external_reference_cases.R](scratch/benchmark_external_reference_cases.R)

Use the uploaded-suite supported-subset harness:

- [tools/run_supported_subset_validation.R](tools/run_supported_subset_validation.R)

What it does:

- prepares the reference datasets
- fits the same model with `rustyINLA` and `R-INLA`
- records accuracy, time, and memory side by side

### Recommended workflow

1. Run the active internal benchmark suite first.
2. Run the external reference harness second.
3. Treat internal parity as the release gate.
4. Treat external examples as public-facing validation and semantic anchoring.
5. Before asking to commit or open a PR for solver, likelihood, covariance,
   optimizer, package-loading, or benchmark-harness changes, present the
   relevant benchmark timings, memory numbers, and parameter-accuracy metrics.

Why this order:

- internal benchmarks are closer to our real MVP subset
- external examples are more useful for communication and broader validation

Timing comparisons must say which build/load path produced the numbers.
Optimized package or release-DLL timings are the merge gate for performance;
debug-DLL timings are useful for diagnostics but should not be used as the
only evidence for a performance decision.

Parameter-accuracy comparisons are part of the same pre-commit gate. For each
affected benchmark or reference case, report the relevant `R-INLA` parity
metrics: fixed-effect means and SDs, random-effect means and SDs, hyperparameter
means and SDs, fitted means, and log marginal likelihood. If a metric is not
applicable to the case, label it as such rather than omitting the accuracy
check.

## 5. What to compare

For each example, compare these first:

- model formula and family semantics
- latent-effect interpretation
- fixed-effect means and SDs
- random-effect means and SDs
- hyperparameter means and SDs
- fitted means
- log marginal likelihood

Then compare these second:

- runtime
- memory
- output-profile sensitivity (`thin` vs `benchmark`)

## 6. Accuracy hierarchy

Not all mismatches are equally important.

### Highest priority

- wrong family semantics
- wrong latent-effect semantics
- wrong hyperparameter interpretation
- unstable or non-finite fitted values

### Medium priority

- moderate `mlik` drift with otherwise good fitted and latent summaries
- small summary mismatches attributable to approximation differences

### Lower priority for public examples

- perfect object-shape parity
- perfect low-level internal config parity

## 7. Recommended benchmark narrative for public communication

When describing the package publicly, use this sequence:

1. State the supported subset.
2. Show the internal benchmark pass rate for that subset.
3. Show the closest public examples for each supported model combination.
4. Explain which external cases are exact public matches and which are synthetic exact-family references.
5. Report accuracy, runtime, and memory together.

This avoids overselling while still making the benchmark story legible.

## 8. Release-facing recommendation

For a first public package release:

- keep the internal benchmark suite as the official release gate
- include the external example suite as a public validation appendix
- clearly label public exact matches versus synthetic exact-family references

That gives us a strong and honest validation story.

## 9. Best next additions for this guide

The guide gets stronger when we add:

- one exact public example for the next GLM-like family, if we add one
- frozen output snapshots for the external suite, not just the scripts

The `rw2` gap is now closed by the public LIDAR smoothing case in
`SemiPar::lidar`, which mirrors the GitBook `rw2` example directly.
The key structural parity detail is that `rw2` must use the actual covariate
values for irregular grids, not just sequential indices. With that fix in
place, the current LIDAR benchmark is now very close to `R-INLA`
(`random_mean_max_abs ~= 4.33e-05`, `random_sd_max_abs ~= 1.31e-05`,
`fitted_mean_max_rel ~= 8.51e-05`).

The current synthetic exact-family coverage also includes:

- Gaussian + `ar2`, benchmarked against `R-INLA` `model = "ar", order = 2`
- Gaussian + multiple fixed effects + `iid` + offset, which exercises the first Phase 7A fixed-effects slice through the current `model.matrix()` path
- Fixed-effect-only Poisson/Gamma-style GLM surfaces through `tools/run_supported_subset_validation.R`

## 10. Related files

- [scratch/EXTERNAL_EXAMPLES_FOR_ACTIVE_BENCHMARK_MODELS_2026-04-18.md](scratch/EXTERNAL_EXAMPLES_FOR_ACTIVE_BENCHMARK_MODELS_2026-04-18.md)
- [scratch/benchmark_external_reference_cases.R](scratch/benchmark_external_reference_cases.R)
- [tools/run_supported_subset_validation.R](tools/run_supported_subset_validation.R)
- [COVERAGE_EVALUATION_2026-04-19.md](COVERAGE_EVALUATION_2026-04-19.md)
- [RINLA_PARITY_GAP_INVENTORY.md](RINLA_PARITY_GAP_INVENTORY.md)
