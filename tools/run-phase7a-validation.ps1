[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Rscript
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-Rscript {
    param(
        [string]$RequestedRscript
    )

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($RequestedRscript)) {
        $candidates += $RequestedRscript
    }
    if (-not [string]::IsNullOrWhiteSpace($env:R_HOME)) {
        $candidates += Join-Path $env:R_HOME 'bin\x64\Rscript.exe'
        $candidates += Join-Path $env:R_HOME 'bin\Rscript.exe'
    }
    $pathCommand = Get-Command Rscript.exe -ErrorAction SilentlyContinue
    if ($pathCommand) {
        $candidates += $pathCommand.Source
    }
    $candidates += @(
        'C:\Program Files\R\R-4.5.3\bin\x64\Rscript.exe',
        'C:\Program Files\R\R-4.5.3\bin\Rscript.exe',
        'C:\Program Files\R\R-4.3.0\bin\x64\Rscript.exe',
        'C:\Program Files\R\R-4.3.0\bin\Rscript.exe'
    )

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw 'Unable to locate Rscript.exe. Pass -Rscript or set R_HOME.'
}

function Invoke-RValidation {
    param(
        [string]$Label,
        [string]$ScriptPath
    )

    Write-Host ''
    Write-Host "==> $Label"
    Write-Host "    $script:RscriptExe $ScriptPath"
    & $script:RscriptExe $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$script:RscriptExe = Resolve-Rscript -RequestedRscript $Rscript
$previousForceWorktree = $env:RUSTYINLA_FORCE_WORKTREE

Push-Location $repoRoot
try {
    $env:RUSTYINLA_FORCE_WORKTREE = '1'

    Invoke-RValidation `
        -Label 'Fixed-effects formula interface contract' `
        -ScriptPath (Join-Path $repoRoot 'tests\fixed-effects-interface.R')
    Invoke-RValidation `
        -Label 'Public rusty_inla() validation errors' `
        -ScriptPath (Join-Path $repoRoot 'tests\fixed-effects-public-api-errors.R')
    Invoke-RValidation `
        -Label 'Fixed-only R-INLA parity' `
        -ScriptPath (Join-Path $repoRoot 'tests\fixed-only-parity.R')
    Invoke-RValidation `
        -Label 'Curated supported-subset validation' `
        -ScriptPath (Join-Path $repoRoot 'tools\run_supported_subset_validation.R')

    Write-Host ''
    Write-Host 'Phase 7A validation gate passed.'
} finally {
    if ($null -eq $previousForceWorktree) {
        Remove-Item Env:\RUSTYINLA_FORCE_WORKTREE -ErrorAction SilentlyContinue
    } else {
        $env:RUSTYINLA_FORCE_WORKTREE = $previousForceWorktree
    }
    Pop-Location
}
