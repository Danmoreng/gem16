[CmdletBinding()]
param(
    [ValidateRange(0, 1024)]
    [int]$Jobs = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
. (Join-Path $repoRoot "scripts\windows-toolchain.ps1")

$expectedCommit = (Get-Content -Raw (Join-Path $PSScriptRoot "commit.txt")).Trim()
$sourceDir = if ($env:LLAMA_CPP_SOURCE) { $env:LLAMA_CPP_SOURCE } else { Join-Path $repoRoot "third_party\cache\llama.cpp" }
$buildDir = if ($env:LLAMA_CPP_BUILD_DIR) { $env:LLAMA_CPP_BUILD_DIR } else { Join-Path $repoRoot "build\Windows\llama_cpp\release" }

Import-Gem16VisualStudioEnvironment
Import-Gem16CudaEnvironment
Assert-Gem16Command "cmake.exe" "Install CMake 3.28 or newer."
Assert-Gem16Command "ninja.exe" "Install Ninja or the Visual Studio CMake tools component."
Assert-Gem16Command "nvcc.exe" "Install the pinned NVIDIA CUDA toolkit and set CUDA_PATH."

if (-not (Test-Path -LiteralPath (Join-Path $sourceDir ".git"))) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $sourceDir) | Out-Null
    Invoke-Gem16Checked "git.exe" @("clone", "--filter=blob:none", "--no-checkout", "https://github.com/ggml-org/llama.cpp.git", $sourceDir)
}

Invoke-Gem16Checked "git.exe" @("-C", $sourceDir, "fetch", "--filter=blob:none", "origin", $expectedCommit)
Invoke-Gem16Checked "git.exe" @("-C", $sourceDir, "checkout", "--detach", $expectedCommit)
$actualCommit = (& git.exe -C $sourceDir rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $expectedCommit) {
    throw "Expected llama.cpp $expectedCommit, got $actualCommit"
}
$changes = & git.exe -C $sourceDir status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0 -or $changes) {
    throw "Refusing to build a modified llama.cpp worktree at $sourceDir"
}

$configureArguments = @(
    "-S", $sourceDir,
    "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CUDA_ARCHITECTURES=120a-real",
    "-DGGML_CUDA=ON",
    "-DGGML_NATIVE=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=ON",
    "-DLLAMA_BUILD_TOOLS=ON"
)
Invoke-Gem16Checked "cmake.exe" $configureArguments

$buildArguments = @("--build", $buildDir, "--parallel")
if ($Jobs -gt 0) { $buildArguments += $Jobs.ToString() }
$buildArguments += @("--target", "llama-cli", "llama-bench", "llama-quantize")
Invoke-Gem16Checked "cmake.exe" $buildArguments
Invoke-Gem16Checked (Join-Path $buildDir "bin\llama-cli.exe") @("--version")
