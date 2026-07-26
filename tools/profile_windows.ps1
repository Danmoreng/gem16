[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$Model,
  [string]$BuildDirectory = "build\Windows\blackwell-release",
  [ValidateRange(1, 262144)]
  [int]$Context = 128,
  [ValidateRange(1, 4096)]
  [int]$DecodeTokens = 16,
  [ValidateRange(1, 100)]
  [int]$Warmups = 1,
  [ValidateRange(1, 100)]
  [int]$Repetitions = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repository = Split-Path -Parent $PSScriptRoot
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$buildPath = (Resolve-Path -LiteralPath (Join-Path $repository $BuildDirectory)).Path
$benchmark = Join-Path $buildPath "bin\gem16-bench.exe"
if (-not (Test-Path -LiteralPath $benchmark -PathType Leaf)) {
  throw "Benchmark executable not found: $benchmark"
}

$nsightRoot = Join-Path $env:ProgramFiles "NVIDIA Corporation"
$nsys = Get-ChildItem -LiteralPath $nsightRoot -Directory -Filter "Nsight Systems *" |
  ForEach-Object { Join-Path $_.FullName "target-windows-x64\nsys.exe" } |
  Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
  Sort-Object -Descending |
  Select-Object -First 1
if ($null -eq $nsys) {
  throw "Nsight Systems nsys.exe was not found under $nsightRoot"
}

$profileDirectory = Join-Path $buildPath "profiles"
New-Item -ItemType Directory -Force -Path $profileDirectory | Out-Null

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [Parameter(Mandatory = $true)]
    [string[]]$Arguments
  )
  & $Executable @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$Executable exited with code $LASTEXITCODE"
  }
}

function Invoke-Profile {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [string[]]$BenchmarkArguments
  )
  $output = Join-Path $profileDirectory $Name
  $profileArguments = @(
    "profile",
    "--trace=cuda,nvtx",
    "--sample=none",
    "--cpuctxsw=none",
    "--force-overwrite=true",
    "--output", $output,
    $benchmark
  ) + $BenchmarkArguments
  Invoke-Checked $nsys $profileArguments
  Invoke-Checked $nsys @(
    "stats",
    "--report", "nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_api_sum",
    "--force-export=true",
    "$output.nsys-rep"
  )
}

Invoke-Profile "prefill-$Context" @(
  "prefill", "--model", $modelPath,
  "--context", "$Context",
  "--warmups", "$Warmups",
  "--repetitions", "$Repetitions"
)

Invoke-Profile "decode-$Context" @(
  "decode", "--model", $modelPath,
  "--context", "$Context",
  "--tokens", "$DecodeTokens",
  "--warmups", "$Warmups",
  "--repetitions", "$Repetitions"
)
