[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$GradleWrapper = Join-Path $RepoRoot "gradlew.bat"
$Server = Join-Path $RepoRoot "build\Windows\blackwell-release\bin\gem16-server.exe"
$MsiDirectory = Join-Path $RepoRoot "studioApp\build\compose\binaries\main\msi"

if (-not (Test-Path -LiteralPath $GradleWrapper -PathType Leaf)) {
    throw "Gradle wrapper was not found: $GradleWrapper"
}
if (-not (Test-Path -LiteralPath $Server -PathType Leaf)) {
    throw "Build the Windows CUDA release server before packaging: $Server"
}

# Compose's packaging task can otherwise remain UP-TO-DATE after the native
# server changed because the generated resource dependency is not reflected in
# every jpackage output fingerprint. Remove only prior gem16 MSI outputs and
# force the packaging tasks so a successful invocation always means a fresh MSI.
if (Test-Path -LiteralPath $MsiDirectory -PathType Container) {
    Get-ChildItem -LiteralPath $MsiDirectory -Filter "gem16-*.msi" -File |
        Remove-Item -Force
}

Write-Host "> $GradleWrapper -p $RepoRoot :studioApp:packageMsi --rerun-tasks --no-daemon"
& $GradleWrapper -p $RepoRoot :studioApp:packageMsi --rerun-tasks --no-daemon
if ($LASTEXITCODE -ne 0) {
    throw "Studio MSI packaging failed with exit code $LASTEXITCODE"
}

$packages = @(Get-ChildItem -LiteralPath $MsiDirectory -Filter "gem16-*.msi" -File)
if ($packages.Count -ne 1) {
    throw "Expected exactly one fresh gem16 MSI in $MsiDirectory, found $($packages.Count)."
}

$package = $packages[0]
$hash = Get-FileHash -LiteralPath $package.FullName -Algorithm SHA256
Write-Host "Created $($package.FullName)"
Write-Host "Size: $($package.Length) bytes"
Write-Host "SHA256: $($hash.Hash)"
