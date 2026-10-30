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

$sourceDir = if ($env:LLAMA_CPP_SOURCE) { $env:LLAMA_CPP_SOURCE } else { if ($env:LLAMA_CPP_HOME) { $env:LLAMA_CPP_HOME } else { "C:/llama.cpp" } }
$python = if ($env:LLAMA_CPP_CONVERT_PYTHON) { $env:LLAMA_CPP_CONVERT_PYTHON } else { Join-Path $repoRoot "third_party\cache\llama-convert-venv\Scripts\python.exe" }

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Converter Python was not found at $python. Install the pinned requirements/requirements-convert_hf_to_gguf.txt first."
}
if (-not (Test-Path -LiteralPath (Join-Path $sourceDir "convert_hf_to_gguf.py"))) { throw "Shared llama.cpp converter not found at $sourceDir" }
if (-not (Test-Path -LiteralPath (Join-Path $ModelDir "config.json") -PathType Leaf)) {
    throw "Model directory has no config.json: $ModelDir"
}

$arguments = @(
    (Join-Path $sourceDir "convert_hf_to_gguf.py"),
    "--outtype", "auto",
    "--outfile", $OutputGguf
) + $ConverterOptions + @($ModelDir)
Invoke-Gem16Checked $python $arguments
