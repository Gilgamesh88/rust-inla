[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$RHome,
    [string]$CacheRoot,
    [switch]$NoInstructions,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-LatestRHome {
    param(
        [string]$BasePath = 'C:\Program Files\R'
    )

    if (-not (Test-Path $BasePath)) {
        throw "R base path not found: $BasePath"
    }

    $candidate = Get-ChildItem -Path $BasePath -Directory |
        Where-Object { $_.Name -match '^R-\d+\.\d+\.\d+$' } |
        Sort-Object {
            [version]($_.Name -replace '^R-', '')
        } -Descending |
        Select-Object -First 1

    if (-not $candidate) {
        throw "No R installation found under $BasePath"
    }

    return $candidate.FullName
}

function Find-ToolExecutable {
    param(
        [string]$Name,
        [string[]]$SearchRoots
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) {
        return $command.Source
    }

    foreach ($root in $SearchRoots) {
        if (-not (Test-Path $root)) {
            continue
        }

        $match = Get-ChildItem -Path $root -Recurse -Filter $Name -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }

    throw "Unable to find $Name. Install Visual Studio build tools or add it to PATH."
}

function Prepend-EnvPath {
    param(
        [string]$VariableName,
        [string]$Entry
    )

    $current = [Environment]::GetEnvironmentVariable($VariableName, 'Process')
    if ([string]::IsNullOrWhiteSpace($current)) {
        [Environment]::SetEnvironmentVariable($VariableName, $Entry, 'Process')
        return
    }

    $parts = $current -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    if ($parts -contains $Entry) {
        return
    }

    [Environment]::SetEnvironmentVariable($VariableName, "$Entry;$current", 'Process')
}

function Ensure-RImportLibrary {
    param(
        [string]$ResolvedRHome,
        [string]$ResolvedCacheRoot,
        [string]$DumpbinExe,
        [string]$LibExe
    )

    $rDll = Join-Path $ResolvedRHome 'bin\x64\R.dll'
    if (-not (Test-Path $rDll)) {
        throw "R.dll not found at $rDll"
    }

    $cacheDir = Join-Path $ResolvedCacheRoot (Join-Path (Split-Path $ResolvedRHome -Leaf) 'x64')
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null

    $rLib = Join-Path $cacheDir 'R.lib'
    if (Test-Path $rLib) {
        return $cacheDir
    }

    $defPath = Join-Path $cacheDir 'R.def'
    $exports = & $DumpbinExe /exports $rDll |
        Select-String -Pattern '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+\S+' |
        ForEach-Object { ($_ -split '\s+')[-1] }

    if (-not $exports -or $exports.Count -eq 0) {
        throw "Unable to extract exports from $rDll"
    }

    @('LIBRARY R', 'EXPORTS') + $exports | Set-Content -Path $defPath -Encoding Ascii
    & $LibExe "/def:$defPath" '/machine:x64' "/out:$rLib" '/name:R.dll' | Out-Host

    if (-not (Test-Path $rLib)) {
        throw "Failed to generate R import library at $rLib"
    }

    return $cacheDir
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$resolvedCacheRootInput = if ([string]::IsNullOrWhiteSpace($CacheRoot)) {
    Join-Path $repoRoot 'scratch\build_support'
} else {
    $CacheRoot
}
$resolvedRHome = if ([string]::IsNullOrWhiteSpace($RHome)) {
    Get-LatestRHome
} else {
    $resolved = Resolve-Path -Path $RHome -ErrorAction Stop
    $resolved.ProviderPath
}

$resolvedCacheRoot = Resolve-Path -Path (New-Item -ItemType Directory -Force -Path $resolvedCacheRootInput) |
    Select-Object -ExpandProperty ProviderPath
$rInclude = Join-Path $resolvedRHome 'include'
$rBin = Join-Path $resolvedRHome 'bin\x64'

if (-not (Test-Path $rInclude)) {
    throw "R include directory not found at $rInclude"
}
if (-not (Test-Path $rBin)) {
    throw "R bin directory not found at $rBin"
}

$vsRoots = @(
    'C:\Program Files\Microsoft Visual Studio',
    'C:\Program Files (x86)\Microsoft Visual Studio'
)
$dumpbinExe = Find-ToolExecutable -Name 'dumpbin.exe' -SearchRoots $vsRoots
$libExe = Find-ToolExecutable -Name 'lib.exe' -SearchRoots $vsRoots
$importLibDir = Ensure-RImportLibrary `
    -ResolvedRHome $resolvedRHome `
    -ResolvedCacheRoot $resolvedCacheRoot `
    -DumpbinExe $dumpbinExe `
    -LibExe $libExe

[Environment]::SetEnvironmentVariable('R_HOME', $resolvedRHome, 'Process')
[Environment]::SetEnvironmentVariable('R_INCLUDE_DIR', $rInclude, 'Process')
Prepend-EnvPath -VariableName 'PATH' -Entry $rBin
Prepend-EnvPath -VariableName 'LIB' -Entry $importLibDir

Write-Host "Configured Windows R build environment:"
Write-Host "  R_HOME=$resolvedRHome"
Write-Host "  R_INCLUDE_DIR=$rInclude"
Write-Host "  PATH += $rBin"
Write-Host "  LIB += $importLibDir"

if ($Command -and $Command.Length -gt 0) {
    if ($Command.Length -eq 1) {
        & $Command[0]
    } else {
        & $Command[0] $Command[1..($Command.Length - 1)]
    }
    exit $LASTEXITCODE
}

if ($NoInstructions) {
    return
}

Write-Host ''
Write-Host 'In the same PowerShell session, run:'
Write-Host '  .\tools\with-r-build-env.ps1'
Write-Host '  cargo test --workspace'
Write-Host '  cargo clippy --workspace -- -D warnings'
Write-Host ''
Write-Host 'For simple commands without a second `--`, one-shot forwarding also works:'
Write-Host '  .\tools\with-r-build-env.ps1 cargo test --workspace'
Write-Host ''
Write-Host 'For the full Windows workspace validation wrapper, run:'
Write-Host '  .\tools\check-rust-workspace-win.ps1'
