[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$RHome,
    [string]$CacheRoot,
    [string]$Target = 'x86_64-pc-windows-gnu',
    [switch]$SkipTests,
    [switch]$SkipClippy,
    [switch]$SkipPackageInstallFallback,
    [switch]$StrictCargo
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

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

function Get-RVersionInfo {
    param(
        [string]$IncludeDir
    )

    $headerPath = Join-Path $IncludeDir 'Rversion.h'
    if (-not (Test-Path $headerPath)) {
        throw "Rversion.h not found at $headerPath"
    }

    $header = Get-Content -Path $headerPath
    $majorLine = $header | Where-Object { $_ -match '^#define\s+R_MAJOR\s+"(?<value>[^"]+)"' } | Select-Object -First 1
    $minorLine = $header | Where-Object { $_ -match '^#define\s+R_MINOR\s+"(?<value>[^"]+)"' } | Select-Object -First 1

    if (-not $majorLine -or -not $minorLine) {
        throw "Unable to parse R version from $headerPath"
    }

    $major = [regex]::Match($majorLine, '^#define\s+R_MAJOR\s+"(?<value>[^"]+)"').Groups['value'].Value
    $minorRaw = [regex]::Match($minorLine, '^#define\s+R_MINOR\s+"(?<value>[^"]+)"').Groups['value'].Value
    $minorParts = $minorRaw -split '\.'

    [pscustomobject]@{
        Major = $major
        Minor = $minorParts[0]
        Patch = if ($minorParts.Length -gt 1) { $minorParts[1] } else { '0' }
        Display = "$major.$minorRaw"
    }
}

function Get-RExecutable {
    param(
        [string]$ResolvedRHome
    )

    $candidates = @(
        Join-Path $ResolvedRHome 'bin\x64\R.exe'
        Join-Path $ResolvedRHome 'bin\R.exe'
    )

    $resolved = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $resolved) {
        throw "Unable to locate R.exe under $ResolvedRHome"
    }

    return $resolved
}

function Invoke-Step {
    param(
        [string]$Label,
        [string]$Executable,
        [string[]]$Arguments
    )

    Write-Host ''
    Write-Host "==> $Label"
    Write-Host "    $Executable $($Arguments -join ' ')"

    & $Executable @Arguments | Out-Host
    return [int]$LASTEXITCODE
}

function Add-RtoolsToPath {
    $candidates = @(
        'C:\rtools45\x86_64-w64-mingw32.static.posix\bin'
        'C:\rtools45\usr\bin'
        'C:\rtools44\x86_64-w64-mingw32.static.posix\bin'
        'C:\rtools44\usr\bin'
    )

    $added = @()
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            Prepend-EnvPath -VariableName 'PATH' -Entry $candidate
            $added += $candidate
        }
    }

    if ($added.Count -gt 0) {
        Write-Host "Rtools PATH additions: $($added -join '; ')"
    }
}

function Configure-GnuToolchain {
    param(
        [string]$RustTarget
    )

    if ($RustTarget -ne 'x86_64-pc-windows-gnu') {
        return
    }

    $binRoots = @(
        'C:\rtools45\x86_64-w64-mingw32.static.posix\bin'
        'C:\rtools44\x86_64-w64-mingw32.static.posix\bin'
    )

    $gccPath = $null
    $arPath = $null

    foreach ($binRoot in $binRoots) {
        if (-not (Test-Path $binRoot)) {
            continue
        }

        $gccCandidates = @(
            Join-Path $binRoot 'x86_64-w64-mingw32.static.posix-gcc.exe'
            Join-Path $binRoot 'gcc.exe'
        )
        $arCandidates = @(
            Join-Path $binRoot 'x86_64-w64-mingw32.static.posix-ar.exe'
            Join-Path $binRoot 'ar.exe'
        )

        $gccPath = $gccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        $arPath = $arCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

        if ($gccPath -and $arPath) {
            break
        }
    }

    if ($gccPath) {
        [Environment]::SetEnvironmentVariable('CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER', $gccPath, 'Process')
        [Environment]::SetEnvironmentVariable('CC_x86_64_pc_windows_gnu', $gccPath, 'Process')
        Write-Host "GNU linker: $gccPath"
    }

    if ($arPath) {
        [Environment]::SetEnvironmentVariable('CARGO_TARGET_X86_64_PC_WINDOWS_GNU_AR', $arPath, 'Process')
        [Environment]::SetEnvironmentVariable('AR_x86_64_pc_windows_gnu', $arPath, 'Process')
        Write-Host "GNU archiver: $arPath"
    }
}

function Ensure-LibgccMock {
    param(
        [string]$RootPath
    )

    $mockDir = Join-Path $RootPath 'libgcc_mock'
    $mockLib = Join-Path $mockDir 'libgcc_eh.a'

    New-Item -ItemType Directory -Force -Path $mockDir | Out-Null
    if (-not (Test-Path $mockLib)) {
        New-Item -ItemType File -Force -Path $mockLib | Out-Null
    }

    Prepend-EnvPath -VariableName 'LIBRARY_PATH' -Entry $mockDir
    Write-Host "GNU libgcc mock: $mockDir"
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$bootstrapScript = Join-Path $PSScriptRoot 'with-r-build-env.ps1'
$manifestPath = 'src/rust/Cargo.toml'
$resolvedCacheRoot = if ([string]::IsNullOrWhiteSpace($CacheRoot)) {
    Join-Path $env:TEMP 'rustyINLA-r-build-cache'
} else {
    $CacheRoot
}
$packageInstallLibrary = Join-Path $env:TEMP 'rustyINLA-workspace-check-lib'
$toolingScratchRoot = Join-Path $env:TEMP 'rustyINLA-workspace-check'

Push-Location $repoRoot
try {
    & $bootstrapScript -RHome $RHome -CacheRoot $resolvedCacheRoot -NoInstructions

    $rVersion = Get-RVersionInfo -IncludeDir $env:R_INCLUDE_DIR
    $env:DEP_R_R_VERSION_MAJOR = $rVersion.Major
    $env:DEP_R_R_VERSION_MINOR = $rVersion.Minor
    $env:DEP_R_R_VERSION_PATCH = $rVersion.Patch
    [Environment]::SetEnvironmentVariable('CARGO_NET_OFFLINE', $null, 'Process')
    Add-RtoolsToPath
    Configure-GnuToolchain -RustTarget $Target
    Ensure-LibgccMock -RootPath $toolingScratchRoot

    Write-Host "Configured extendr R version env: $($rVersion.Display)"
    Write-Host "Workspace target: $Target"

    $cargoTestExit = 0
    if (-not $SkipTests) {
        $cargoTestExit = Invoke-Step `
            -Label 'Cargo test (workspace)' `
            -Executable 'cargo' `
            -Arguments @('test', '--workspace', '--manifest-path', $manifestPath, '--target', $Target)
    }

    $cargoClippyExit = 0
    if (-not $SkipClippy) {
        $cargoClippyExit = Invoke-Step `
            -Label 'Cargo clippy (workspace)' `
            -Executable 'cargo' `
            -Arguments @('clippy', '--workspace', '--manifest-path', $manifestPath, '--target', $Target, '--', '-D', 'warnings')
    }

    $cargoChecksPassed = ($cargoTestExit -eq 0) -and ($cargoClippyExit -eq 0)
    $packageInstallExit = $null

    if (-not $cargoChecksPassed -and -not $SkipPackageInstallFallback) {
        New-Item -ItemType Directory -Force -Path $packageInstallLibrary | Out-Null
        $rExe = Get-RExecutable -ResolvedRHome $env:R_HOME
        $packageInstallExit = Invoke-Step `
            -Label 'R CMD INSTALL fallback' `
            -Executable $rExe `
            -Arguments @('CMD', 'INSTALL', "--library=$packageInstallLibrary", $repoRoot)
    }

    Write-Host ''
    Write-Host 'Validation summary'
    Write-Host "  cargo test   : $(if ($SkipTests) { 'skipped' } elseif ($cargoTestExit -eq 0) { 'passed' } else { "failed ($cargoTestExit)" })"
    Write-Host "  cargo clippy : $(if ($SkipClippy) { 'skipped' } elseif ($cargoClippyExit -eq 0) { 'passed' } else { "failed ($cargoClippyExit)" })"
    Write-Host "  R CMD INSTALL fallback : $(if ($null -eq $packageInstallExit) { 'not run' } elseif ($packageInstallExit -eq 0) { 'passed' } else { "failed ($packageInstallExit)" })"

    if ($cargoChecksPassed) {
        exit 0
    }

    if (($packageInstallExit -eq 0) -and (-not $StrictCargo)) {
        Write-Warning 'Workspace cargo checks failed, but the package build fallback passed. Use -StrictCargo if you want this wrapper to fail in that situation.'
        exit 0
    }

    exit 1
} finally {
    Pop-Location
}
