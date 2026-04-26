# Supported-Subset Validation Manifest

Status: static review of the uploaded INLA suites on 2026-04-25. The full
uploaded suites were not run.

Reviewed files:

- `C:/Users/Antonio/Downloads/inla_test_suite_part1.R`
- `C:/Users/Antonio/Downloads/inla_test_suite_part2_fremtpl2.R`
- `C:/Users/Antonio/Downloads/inla_test_suite_part3_stress.R`
- `C:/Users/Antonio/Downloads/inla_complete_test_suite.R`
- `C:/Users/Antonio/Downloads/run_all_benchmarks.R`

## Current Inclusion Rules

A suite case is inside the current supported-subset manifest when it can be
expressed with:

- family `gaussian`, `poisson`, `gamma`, or `zeroinflatedpoisson1`
- zero or more latent terms using `iid`, `rw1`, `rw2`, `ar1`, or `ar2`
- fixed effects from the Phase 7A subset in
  `FIXED_EFFECTS_FORMULA_SUBSET.md`
- formula offsets through `offset(...)`
- no custom priors, graph inputs, `scale.model`, `E`, `Ntrials`, or other
  R-INLA-only control arguments

The fixed-effects subset is intentionally narrower than R formula syntax.
Supported fixed designs must be generated from bare numeric, integer, logical,
or factor columns plus simple interactions among those columns. Rank-deficient
designs, transformed fixed terms, character columns, unused factor levels that
create aliased columns, non-finite design values, and non-finite offsets are
rejected before Rust is called.

Some R-INLA cases are structurally useful but need a small adapter before they
are fair rustyINLA validations:

- convert `E = exposure` to `offset(log(exposure))`
- remove `hyper = ...` and `scale.model = ...` when validating default-prior
  semantics only
- convert R-INLA `f(time, model = "ar", order = 2)` to rustyINLA
  `f(time, model = "ar2")`

## Part 1 Candidates

| Case | Status | Notes |
| --- | --- | --- |
| `FixedOnly_Poisson_Offset` | Candidate | Fixed-effect-only Poisson GLM with formula offset. |
| `FixedOnly_Gamma` | Candidate | Fixed-effect-only Gamma GLM with supported fixed columns. |
| `Germany_Poisson_IID` | Adaptable | Supported Poisson + `iid` with fixed `x`; convert `E` to formula offset. |
| `Epil_Poisson_IID` | Candidate | Multi-column fixed effects plus `iid`. |
| `Epil_Poisson_IID2` | Candidate | Exact uploaded two-`iid` case; covered by the curated harness after adding fixed/latent mode-step damping for the observation-level `iid` component. |
| `Simulated_Gaussian_AR1` | Candidate | Gaussian + `ar1`; uses supported `constr = FALSE`. |
| `Simulated_Poisson_AR1` | Candidate | Poisson + `ar1`; uses supported `constr = FALSE`. |
| `Simulated_Gaussian_AR2` | Adaptable | Convert R-INLA `model = "ar", order = 2` to rustyINLA `model = "ar2"`. |
| `Simulated_Poisson_RW1` | Adaptable | Drop `scale.model` and custom `hyper` for current default-prior validation. |
| `Simulated_Poisson_RW2` | Adaptable | Drop `scale.model` and custom `hyper`; keep numeric `time`. |
| `Simulated_Gamma_IID` | Candidate | Gamma + fixed columns + `iid`. |

Excluded part 1 groups: binomial cases, Besag/BYM graph cases, ZIP type 0
cases, and Tweedie cases until that path is promoted back into the stable
benchmark subset.

## freMTPL2 Candidates

| Case | Status | Notes |
| --- | --- | --- |
| `F01_Poisson_FixedOnly` | Candidate | Fixed-effect-only frequency GLM after converting `E` to offset. |
| `S01_Gamma_FixedOnly` | Candidate | Fixed-effect-only severity GLM. |
| `F03_Poisson_IID_Area` | Adaptable | Convert `E` to offset; drop custom `hyper`. |
| `F04_Poisson_IID_Region` | Adaptable | Convert `E` to offset; drop custom `hyper`. |
| `F05_Poisson_IID_Overdispersion` | Adaptable | Convert `E` to offset; drop custom `hyper`. |
| `F06_Poisson_RW1_DrivAge` | Adaptable | Convert `E` to offset; drop `scale.model` and custom `hyper`. |
| `F07_Poisson_RW2_DrivAge` | Adaptable | Convert `E` to offset; drop `scale.model` and custom `hyper`. |
| `F08_Poisson_AR1_VehAge` | Adaptable | Convert `E` to offset; drop custom `hyper`. |
| `F09_Poisson_RW1_BonusMalus` | Adaptable | Convert `E` to offset; drop `scale.model` and custom `hyper`. |
| `F10_Poisson_RW1_IID_Combined` | Adaptable | Convert `E` to offset; drop `scale.model` and custom `hyper`. |
| `S02_Gamma_IID_Area` | Candidate | Gamma + fixed columns + `iid`. |
| `S03_Gamma_IID_Overdispersion` | Adaptable | Drop custom `hyper`. |
| `S04_Gamma_RW1_DrivAge` | Adaptable | Drop `scale.model`. |
| `S05_Gamma_RW2_DrivAge` | Adaptable | Drop `scale.model`. |

Excluded freMTPL2 groups: `E`-only offset-only cases with no fixed columns,
ZIP type 0, binomial, and Tweedie cases.

## Stress-Suite Candidates

| Case Pattern | Status | Notes |
| --- | --- | --- |
| `FixedOnly_*` | Candidate | Supported when the case uses implemented families and fixed columns only. |
| `Scale_*_Poisson_IID` | Candidate | Poisson + fixed columns + `iid`. |
| `Scale_*_Poisson_RW1` | Adaptable | Drop `scale.model`. |
| `Scale_*_Gamma_IID` | Candidate | Gamma + fixed column + `iid`. |
| `TimeSeries_*_AR1` | Candidate | Gaussian + `ar1`. |
| `TimeSeries_*_RW1` | Adaptable | Drop `scale.model`. |
| `TimeSeries_*_RW2` | Adaptable | Drop `scale.model`. |
| `Groups_*_IID` | Candidate | Poisson + fixed column + `iid`; useful scale coverage. |
| `MultiRE_2IID` | Candidate | Two `iid` latent terms. |
| `MultiRE_IID_RW1` | Adaptable | Drop `scale.model`. |
| `MultiRE_3Effects` | Candidate | Three `iid` latent terms. |
| `Prediction_AR1_200ahead` | Candidate | Gaussian + fixed column + `ar1` with NA response rows; keep as a separate prediction validation. |
| `Prior_Default` | Candidate | Default-prior Poisson + fixed column + `iid`. |

Excluded stress groups: ZIP type 0, binomial, custom PC-prior variants, and
cases whose purpose is unsupported prior/link/control-surface validation.

## Complete Suite

`inla_complete_test_suite.R` largely combines the part 1 and freMTPL2 surfaces.
Use the same inclusion and adapter rules above rather than running it wholesale.
The complete suite is useful as a future source for generated reference outputs
after `E`, custom priors, binomial, and graph-based latent models have
dedicated support.

## Next Harness Shape

The next mergeable validation harness should not source the uploaded scripts
directly. It should define a curated case table from the candidate rows above,
apply the explicit adapters, and run only the supported subset against both
`rustyINLA` and R-INLA.

`tools/run_supported_subset_validation.R` is the first narrow harness in that
shape. It includes real adapted Germany/Epil cases from part 1, deterministic
synthetic cases mapped to parts 2 and 3, and fixed-effect-only GLMs, but still
does not run the full uploaded suites. It compares fixed-effect means and SDs,
random-effect means and SDs, fitted means, and Rusty/R-INLA elapsed times for
each curated case.

For the Phase 7A validation gate, run
`tools/run-phase7a-validation.ps1`. It executes the formula contract tests,
the public `rusty_inla()` validation-error tests, the fixed-only R-INLA parity
script, and the curated supported-subset validation with worktree package
loading forced.

`tests/fixed-effects-interface.R` is the focused Phase 7A formula-contract
regression harness. It checks multi-level factors, logical columns, simple
interactions, offsets with latent terms, rank-deficient designs, unsupported
fixed transforms, invalid `f()` surfaces, and fixed-effect-only formulas.

`tests/fixed-effects-public-api-errors.R` checks the same validation boundary
through the package-level `rusty_inla()` entry point for representative
unsupported fixed terms, rank-deficient fixed-only GLMs, character columns,
non-finite fixed values, non-finite explicit offsets, empty formulas, and
unsupported `f()` arguments.

The next branch-local validation target has started as
`stress_multi_re_three_iid` in `tools/run_supported_subset_validation.R`. It is
a deterministic supported-subset proxy for the uploaded stress
`MultiRE_3Effects` surface: Poisson likelihood, fixed effects, formula offset,
and three additive `iid` latent terms.

## Fixed-Effect Reference Interpretation

The `stress_multi_re_three_iid` case currently shows a small fixed-effect
difference against R-INLA, with a maximum absolute fixed-effect difference of
about `0.0173` in the latest Phase 7A gate. A targeted decomposition showed
that this is not introduced by the three latent `iid` terms: the same scale of
fixed-effect difference is already present in the fixed-only Poisson GLM on the
same synthetic data, and adding one, two, or three `iid` terms changes it only
minimally.

For that synthetic fixed-only Poisson slice, `rustyINLA` matches base
`glm(..., family = poisson())` and an independent penalized MAP solve using
the current fixed-effect prior precision to numerical precision. R-INLA's
reported `summary.fixed` values are consistently offset from that direct
GLM/MAP target, even when using formula offsets, explicit `offset = ...`, or
`E = exposure`.

Interpretation for Phase 7A:

- treat the `stress_multi_re_three_iid` fixed-effect difference as a known
  reference-definition difference, not as evidence of multi-latent drift
- keep the R-INLA comparison in the supported-subset harness because it is
  still the primary parity reference
- use the fixed-only GLM/MAP comparator in `tests/fixed-only-parity.R` to
  distinguish direct fixed-effect numerical accuracy from R-INLA reporting
  parity
