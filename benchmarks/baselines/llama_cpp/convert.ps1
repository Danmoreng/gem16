[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$ModelDir,
    [Parameter(Mandatory = $true, Position = 1)][string]$OutputGguf,
    [Parameter(Position = 2, ValueFromRemainingArguments = $true)][string[]]$ConverterOptions = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
. (Join-Path $repoRoot "scripts\windows-toolchain.ps1")

$expectedCommit = (Get-Content -Raw (Join-Path $PSScriptRoot "commit.txt")).Trim()
$sourceDir = if ($env:LLAMA_CPP_SOURCE) { $env:LLAMA_CPP_SOURCE } else { Join-Path $repoRoot "third_party\cache\llama.cpp" }
$python = if ($env:LLAMA_CPP_CONVERT_PYTHON) { $env:LLAMA_CPP_CONVERT_PYTHON } else { Join-Path $repoRoot "third_party\cache\llama-convert-venv\Scripts\python.exe" }

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Converter Python was not found at $python. Install the pinned requirements/requirements-convert_hf_to_gguf.txt first."
}
$actualCommit = (& git.exe -C $sourceDir rev-parse HEAD 2>$null).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $expectedCommit) {
    throw "llama.cpp source is not at pinned commit $expectedCommit"
}
if (-not (Test-Path -LiteralPath (Join-Path $ModelDir "config.json") -PathType Leaf)) {
    throw "Model directory has no config.json: $ModelDir"
}

$arguments = @(
    (Join-Path $sourceDir "convert_hf_to_gguf.py"),
    "--outtype", "auto",
    "--outfile", $OutputGguf
) + $ConverterOptions + @($ModelDir)
Invoke-Gem16Checked $python $arguments
