# Phase 8 Real-Data Rolling Validation

This note records the first private real-data validation scaffold for Phase 8.
The data are sensitive and must stay outside git.

## Sensitive data location

Place private inputs under:

```text
data/real/
```

That directory is ignored by git. The current expected file is:

```text
data/real/inla_sbi_test.RDS
```

Generated reports are written under:

```text
scratch/phase8_reports/realdata_hull/
```

That directory is also ignored by git.

## Current data contract

The first hull / motor own-damage rolling test expects:

```text
period:          periodo
claims:          pt
exposure:        expuesto
iid random term: desc_armadora
fixed effects:   modeloc, desc_edo_circula, medio_emit
family:          poisson
```

The first planned time split is:

```text
base fit:
  2021_y

sequential updates:
  2022_1t
  2022_2t
  2022_3t
  2022_4t
  2023_1t
  ...
```

The model template is:

```r
pt ~ 1 +
  modeloc +
  desc_edo_circula +
  medio_emit +
  offset(log(expuesto)) +
  f(desc_armadora, model = "iid")
```

After the Phase 8H/8I multi-iid evidence work, the same reusable template can
also run several iid blocks by passing `--iid-cols`. For example, the current
two-random-effect hull diagnostic is:

```powershell
Rscript tools\run_phase8_realdata_rolling_template.R --run-fits `
  --iid-cols=desc_edo_circula,desc_armadora `
  --fixed=modeloc,medio_emit `
  --periods=2021_y,2022_1t `
  --updates=2022_1t `
  --joint-refit-schedule=all `
  --max-fit-minutes=60 `
  --report-dir=scratch\phase8_reports\realdata_two_iid_template_2021_to_2022_1t
```

That builds:

```r
pt ~ 1 +
  modeloc +
  medio_emit +
  offset(log(expuesto)) +
  f(desc_edo_circula, model = "iid") +
  f(desc_armadora, model = "iid")
```

## Poisson cell aggregation

The harness aggregates Poisson frequency data by default before fitting:

```text
periodo
modeloc
desc_edo_circula
medio_emit
desc_armadora
```

For rows with the same model covariates, the likelihood contribution is
preserved by:

```text
pt       = sum(pt)
expuesto = sum(expuesto)
```

because:

```text
y_i ~ Poisson(expuesto_i * lambda_cell)

sum_i y_i ~ Poisson(sum_i expuesto_i * lambda_cell)
```

This is the intended fitting representation for the first real-data experiment.
It reduces repeated policy rows to sufficient Poisson cells while keeping the
same frequency likelihood for the supported model.

Important ordering:

```text
1. keep rows with valid response, period, and model covariates
2. aggregate pt and expuesto by model cell
3. drop only aggregated cells whose total expuesto is <= 0
```

Do not drop row-level `expuesto == 0` observations before aggregation. In this
real extraction, many claim indicators can sit on rows with zero exposure, while
the matching exposure is carried by other rows in the same model cell. Dropping
those rows before aggregation artificially removes claims and creates a false
frequency collapse.

Disable this only for debugging:

```powershell
Rscript tools\run_phase8_realdata_rolling_template.R --schema-only --no-aggregate-poisson-cells
```

## Schema-only pass

Run this before fitting:

```powershell
Rscript tools\run_phase8_realdata_rolling_template.R --schema-only
```

The schema pass checks:

- required columns
- usable rows after missing/invalid filtering
- exposure and claim totals by period
- active, born, dormant, and re-entered `desc_armadora` levels
- number of levels for the fixed effects and random effect
- fixed-effect rank by period
- Poisson cell aggregation ratio

The fixed-effect rank report matters because a quarterly batch can miss a
factor level that was present in the full design. If that creates zero columns
or rank deficiency, the update should fail early rather than silently compare
against a different model.

For the first real-data MVP run, the low-exposure `desc_edo_circula` levels
should be removed explicitly:

```powershell
Rscript tools\run_phase8_realdata_rolling_template.R --schema-only `
  --drop-low-exposure-col=desc_edo_circula `
  --drop-low-exposure-n=5
```

The initial schema diagnostics identified these levels:

```text
EXTRANJERO
TIJUANA
APARTADO
MAZATLAN
PUERTO VALLARTA
```

They had very small exposure and no claims in the diagnostic pass. Removing
them made the fixed-effect design full-rank in every period and cumulative
refit.

## Estado-As-Iid Variant

For actuarial models where geographic state should be pooled, use
`desc_edo_circula` as the single supported iid block and retain brand influence
through an exposure-based fixed feature:

```powershell
Rscript tools\run_phase8_realdata_rolling_template.R --schema-only `
  --iid-col=desc_edo_circula `
  --fixed=modeloc,medio_emit,brand_top20 `
  --top-exposure-feature-col=desc_armadora `
  --top-exposure-feature-n=20 `
  --top-exposure-feature-name=brand_top20 `
  --top-exposure-feature-scope=base
```

The top-N feature is selected by exposure, not claims. With
`--top-exposure-feature-scope=base`, the dictionary is built from the base fit
period only, so future or newly appearing brands collapse to `OTHER`.

## Fitting pass

Only after the schema report looks sane, run:

```powershell
Rscript tools\run_phase8_realdata_rolling_template.R --run-fits
```

For the first smoke test, run only the first update:

```powershell
Rscript tools\run_phase8_realdata_rolling_template.R --run-fits `
  --drop-low-exposure-col=desc_edo_circula `
  --drop-low-exposure-n=5 `
  --updates=2022_1t `
  --max-fit-minutes=20 `
  --report-dir=scratch\phase8_reports\realdata_hull_drop_estado_low5_first_update
```

`--max-fit-minutes` is a script-level guard checked between fit stages. For
large native solver calls, also run the process with an outer timeout when a
hard wall-clock kill is required.

For each update period, the script fits:

```text
rolling update:
  current period data + previous compressed evidence state

truth comparator:
  cumulative joint refit through the same period
```

For multi-iid rolling runs, the template selects the evidence mode from the
current state. A state with multi-point `theta_evidence` uses
`fixed_iid_cross_theta_evidence`. Composed states with one theta dimension per
iid block preserve theta-dependent evidence across rolling periods by aligning
the old recursive evidence graph onto the current graph. The selected mode is
recorded in `realdata_rolling_update_times.csv`.

It writes:

```text
realdata_rolling_update_times.csv
realdata_rolling_metrics.csv
realdata_rolling_theta_proxy_vs_joint.csv
realdata_rolling_effects_proxy_vs_joint.csv
realdata_rolling_fitted_proxy_vs_joint.csv
```

`realdata_rolling_theta_proxy_vs_joint.csv` is the preferred theta diagnostic.
It compares every iid precision hyperparameter on several scales:

```text
log precision mean/mode
raw precision mean and relative drift
block-level random-effect SD, exp(-0.5 * log_precision)
precision marginal CDF KS
uncertainty-aware log-precision z drift
```

Raw precision drift can be misleading when both fits drive an iid precision
very high. The block-SD and internal log-precision columns are usually more
interpretable for actuarial shrinkage.

By default, `--joint-refit-schedule=all` runs a cumulative joint refit after
every update. For longer rolling checks, use `--joint-refit-schedule=final` to
run all sequential updates and compare only against the final cumulative joint
refit:

```powershell
Rscript tools\run_phase8_realdata_rolling_template.R --run-fits `
  --updates=2022_1t,2022_2t,2022_3t,2022_4t,2023_1t,2023_2t,2023_3t,2023_4t `
  --joint-refit-schedule=final
```

For the all-available-years estado-iid check, use the wrapper:

```powershell
.\tools\run_phase8_realdata_all_years_estado_iid.ps1
```

That runs schema-only by default. To fit all available update periods:

```powershell
.\tools\run_phase8_realdata_all_years_estado_iid.ps1 -RunFits -MaxFitMinutes 20
```

The wrapper intentionally leaves `--updates` unset, so the harness uses every
available period after `2021_y`.

Use `-JointRefitSchedule none` to measure production rolling-update timings
without running the expensive validation refit:

```powershell
.\tools\run_phase8_realdata_all_years_estado_iid.ps1 -RunFits `
  -JointRefitSchedule none `
  -ReportDir scratch\phase8_reports\realdata_hull_estado_iid_brand_top20_all_years_rolling_only
```

Core diagnostics:

- update time vs joint refit time
- update object size vs joint object size
- theta log-precision drift
- theta CDF KS distance
- fixed-effect mean drift
- random-effect mean drift by iid block
- fitted-value relative drift on the new period
- fixed and random posterior SD ratios
- interval overlap against the cumulative joint refit
- fitted-value total, quantile, and top-cell drift on the new period
- exposure/claim summaries by factor level and period, useful for sparse
  levels such as `modeloc(20,25]`

## Current Two-Iid Rolling Result

After fixing formula-offset extraction and adding multi-iid theta-evidence
composition, the two-iid all-period diagnostic ran from `2021_y` through `2025_4t`
with a final cumulative joint refit:

```text
report_dir:
  scratch/phase8_reports/realdata_two_iid_all_periods_theta_comp_final_joint

base fit:
  35.47 sec, 3.97 MB object

rolling updates:
  every period mode: fixed_iid_cross_theta_evidence
  final update 2025_4t: 20.09 sec, 5.14 MB object

final joint refit:
  190.75 sec, 60.54 MB object
```

Final-period update versus cumulative joint refit:

```text
theta internal drift max:      0.119118
theta CDF KS max:              0.092032
fixed mean max abs drift:      0.097599
random mean max abs drift:     0.051523
fitted max relative drift:     0.108167
fitted total update vs joint:  1105.16 vs 1087.55
```

By block, the final theta drift is mostly geographic:

```text
desc_edo_circula:
  log-precision mean drift: 0.088188
  log-precision mode drift: 0.119118
  block SD relative drift:  0.045081

desc_armadora:
  log-precision mean drift: 0.011210
  log-precision mode drift: 0.064118
  block SD relative drift:  0.005589
```

This result is a usable MVP diagnostic, not a final claim of exact full Bayes.
The two-iid rolling path now keeps theta-dependent evidence through every
period, but the evidence is still a compressed local Gaussian approximation and
uses guarded support interpolation rather than refitting or storing the full old
data likelihood.

## Walkthrough script

For a direct package-use example rather than the full harness, run:

```powershell
Rscript examples\phase8_two_iid_rolling_update_walkthrough.R
```

That script performs the first two-iid rolling check step by step:

```text
1. read the private RDS
2. aggregate Poisson cells
3. fit 2021_y
4. extract rusty_update_state()
5. update on 2022_1t
6. fit the cumulative joint comparator
7. write theta, effect, fitted, observed-frequency, prediction, and timing
   tables
```

It is meant to be edited locally when testing new period splits or factor
choices.

The walkthrough now writes explicit frequency-scale guardrails:

```text
walkthrough_observed_period_frequencies.csv
walkthrough_prediction_frequency_summary.csv
walkthrough_level_predictions.csv
```

`walkthrough_level_predictions.csv` includes observed claims, exposure,
period count, and empirical frequency for the exact model cell represented by
each prediction row. This is deliberate: a full Cartesian prediction grid can
include sparse or synthetic factor combinations, so the report should make it
obvious whether a surprising frequency is coming from observed experience or
from extrapolating a level combination.

The package interface also has a regression guard for formula offsets mixed
with latent `f(...)` terms. The offset must be extracted from the original
formula before the latent terms are removed for fixed-effect model-matrix
construction; otherwise `offset(log(expuesto))` can be silently lost and the
Poisson model fits claim counts instead of frequency.

## Guardrails

This harness is intentionally narrow:

- same family across periods
- same fixed-effect formula across periods
- one or more `iid` random blocks, supplied through `--iid-col` or
  `--iid-cols`
- same iid covariate names across rolling periods
- new, dormant, and re-entered iid levels are allowed in each block
- multi-iid hyperparameters are still diagnostic until the expanded theta
  support path has more real-data coverage

If the real data violates these constraints, the next step is not to force the
run. The next step is to decide whether the data should be coarsened, the fixed
design simplified, or the recursive evidence graph needs another extension.
