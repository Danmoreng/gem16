[CmdletBinding()]
param(
    [switch]$SkipServerBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Server = Join-Path $RepoRoot "build\Windows\blackwell-release\bin\gem16-server.exe"

. (Join-Path $PSScriptRoot "windows-toolchain.ps1")
Import-Gem16VisualStudioEnvironment
if (-not $SkipServerBuild) {
    Import-Gem16CudaEnvironment
    Push-Location $RepoRoot
    try {
        Invoke-Gem16Checked "cmake.exe" @("--preset", "blackwell-release", "-S", $RepoRoot)
        Invoke-Gem16Checked "cmake.exe" @(
            "--build", "--preset", "blackwell-release", "--target", "gem16-server", "--parallel"
        )
    } finally {
        Pop-Location
    }
}
if (-not (Test-Path -LiteralPath $Server -PathType Leaf)) {
    throw "The workspace server does not exist: $Server. Run without -SkipServerBuild first."
}

$ResolvedServer = (Resolve-Path -LiteralPath $Server).Path
$UnexpectedServers = @(
    Get-Process -Name "gem16-server" -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.Path -ne $ResolvedServer }
)
if ($UnexpectedServers.Count -gt 0) {
    $Paths = ($UnexpectedServers | ForEach-Object { $_.Path } | Sort-Object -Unique) -join ", "
    throw "A non-workspace gem16 server is already running: $Paths. Stop it before starting Studio development."
}

$env:GEM16_REPO_ROOT = $RepoRoot
$env:GEM16_STUDIO_SERVER_EXECUTABLE = $ResolvedServer
$StudioSource = Join-Path $RepoRoot "nativeStudio"
$StudioBuild = Join-Path $RepoRoot "build\Windows\native-studio"
& cmake.exe -S $StudioSource -B $StudioBuild -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
if ($LASTEXITCODE -ne 0) { throw "Native Studio configure failed" }
& cmake.exe --build $StudioBuild --config Release --target gem16-studio --parallel
if ($LASTEXITCODE -ne 0) { throw "Native Studio build failed" }
& (Join-Path $StudioBuild "bin\gem16-studio.exe")
exit $LASTEXITCODE
