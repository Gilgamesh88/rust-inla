param(
    [switch]$RunFits,
    [int]$MaxFitMinutes = 20,
    [ValidateSet("final", "none", "all")]
    [string]$JointRefitSchedule = "final",
    [string]$ReportDir = "scratch\phase8_reports\realdata_hull_estado_iid_brand_top20_all_years_final",
    [string]$Rscript = "C:\Program Files\R\R-4.5.3\bin\Rscript.exe"
)

$ErrorActionPreference = "Stop"

$modeArg = if ($RunFits) { "--run-fits" } else { "--schema-only" }
$args = @(
    "tools\run_phase8_realdata_rolling_template.R",
    $modeArg,
    "--iid-col=desc_edo_circula",
    "--fixed=modeloc,medio_emit,brand_top20",
    "--top-exposure-feature-col=desc_armadora",
    "--top-exposure-feature-n=20",
    "--top-exposure-feature-name=brand_top20",
    "--top-exposure-feature-scope=base",
    "--joint-refit-schedule=$JointRefitSchedule",
    "--max-fit-minutes=$MaxFitMinutes",
    "--report-dir=$ReportDir"
)

& $Rscript @args
