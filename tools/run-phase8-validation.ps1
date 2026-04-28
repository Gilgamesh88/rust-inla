[CmdletBinding()]
param(
    [string]$Rscript
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path $PSScriptRoot -Parent

if ([string]::IsNullOrWhiteSpace($Rscript)) {
    $candidates = @(
        'C:\Program Files\R\R-4.5.3\bin\x64\Rscript.exe',
        'C:\Program Files\R\R-4.5.3\bin\Rscript.exe',
        'Rscript.exe'
    )
    $Rscript = $candidates | Where-Object {
        if ($_ -eq 'Rscript.exe') {
            $null -ne (Get-Command $_ -ErrorAction SilentlyContinue)
        } else {
            Test-Path $_
        }
    } | Select-Object -First 1
}

if ([string]::IsNullOrWhiteSpace($Rscript)) {
    throw 'Unable to locate Rscript.exe. Pass -Rscript explicitly.'
}

$env:RUSTYINLA_FORCE_WORKTREE = '1'

Write-Host ''
Write-Host '==> Phase 8 iid posterior-state experiment'
Write-Host "    $Rscript tests\posterior-state-iid-experimental.R"

Push-Location $repoRoot
try {
    & $Rscript 'tests\posterior-state-iid-experimental.R'
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host '==> Phase 8 fixed+iid Gaussian evidence experiment'
Write-Host "    $Rscript tests\posterior-state-fixed-iid-evidence.R"

Push-Location $repoRoot
try {
    & $Rscript 'tests\posterior-state-fixed-iid-evidence.R'
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host '==> Phase 8 fixed+iid cross evidence contract'
Write-Host "    $Rscript tests\posterior-state-fixed-iid-contract.R"

Push-Location $repoRoot
try {
    & $Rscript 'tests\posterior-state-fixed-iid-contract.R'
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host '==> Phase 8 theta evidence extraction shape'
Write-Host "    $Rscript tests\posterior-state-theta-evidence-shape.R"

Push-Location $repoRoot
try {
    & $Rscript 'tests\posterior-state-theta-evidence-shape.R'
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host '==> Phase 8 theta-dependent evidence objective'
Write-Host "    $Rscript tests\posterior-state-theta-objective.R"

Push-Location $repoRoot
try {
    & $Rscript 'tests\posterior-state-theta-objective.R'
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host '==> Phase 8 composed rolling update state'
Write-Host "    $Rscript tests\posterior-state-composition.R"

Push-Location $repoRoot
try {
    & $Rscript 'tests\posterior-state-composition.R'
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host '==> Phase 8 dormant iid level carry'
Write-Host "    $Rscript tests\posterior-state-dormant-iid-levels.R"

Push-Location $repoRoot
try {
    & $Rscript 'tests\posterior-state-dormant-iid-levels.R'
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host 'Phase 8 validation gate passed.'
