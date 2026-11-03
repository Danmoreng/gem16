[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$source = if ($env:LLAMA_CPP_SOURCE) { $env:LLAMA_CPP_SOURCE } elseif ($env:LLAMA_CPP_HOME) { $env:LLAMA_CPP_HOME } else { 'C:/llama.cpp' }
if (-not (Test-Path -LiteralPath (Join-Path $source 'convert_hf_to_gguf.py'))) {
    throw "Shared llama.cpp converter not found at $source"
}
Write-Host 'Current upstream includes mixed FP8/NVFP4 converter support; no patched checkout is required.'
Write-Host 'Run convert-patched.ps1 to use the central converter with the existing Python environment.'
