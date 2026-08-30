[CmdletBinding()]
param(
    [switch]$Cuda,
    [switch]$Sanitize,
    [switch]$Test,
    [switch]$ConfigureOnly,
    [string]$PythonExecutable = "",
    [ValidateRange(0, 1024)]
    [int]$Jobs = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "windows-toolchain.ps1")

if ($Cuda -and $Sanitize) {
    throw "-Cuda and -Sanitize select different presets and cannot be combined."
}
if ($Sanitize) {
    throw "The host-sanitize preset requires GCC or Clang ASan/UBSan and is currently supported on Linux only."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$preset = if ($Cuda) { "blackwell-release" } else { "host-debug" }

Import-Gem16VisualStudioEnvironment
Assert-Gem16Command "cmake.exe" "Install CMake 3.28 or newer."
Assert-Gem16Command "ninja.exe" "Install Ninja or the Visual Studio CMake tools component."

if ($Cuda) {
    Import-Gem16CudaEnvironment
    Assert-Gem16Command "nvcc.exe" "Install the pinned NVIDIA CUDA toolkit and set CUDA_PATH."
}

Push-Location $repoRoot
try {
    $configureArguments = @("--preset", $preset, "-S", $repoRoot)
    if ($PythonExecutable) {
        $resolvedPython = (Resolve-Path -LiteralPath $PythonExecutable).Path
        $configureArguments += "-DPython3_EXECUTABLE=$resolvedPython"
    }
    Invoke-Gem16Checked "cmake.exe" $configureArguments
    if ($ConfigureOnly) {
        return
    }

    $buildArguments = @("--build", "--preset", $preset, "--parallel")
    if ($Jobs -gt 0) {
        $buildArguments += $Jobs.ToString()
    }
    Invoke-Gem16Checked "cmake.exe" $buildArguments

    if ($Test) {
        Invoke-Gem16Checked "ctest.exe" @("--preset", $preset)
    }
} finally {
    Pop-Location
}
