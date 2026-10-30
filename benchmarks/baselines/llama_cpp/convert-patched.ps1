[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$ModelDir,
    [Parameter(Mandatory = $true, Position = 1)][string]$OutputGguf,
    [Parameter(Position = 2, ValueFromRemainingArguments = $true)][string[]]$ConverterOptions = @()
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$oldPython = $env:LLAMA_CPP_CONVERT_PYTHON
try {
    if (-not $oldPython) {
        $env:LLAMA_CPP_CONVERT_PYTHON = Join-Path $repoRoot 'third_party\cache\unsloth-nvfp4-env\Scripts\python.exe'
    }
    Write-Host 'Using the central upstream converter; mixed FP8/NVFP4 support is built in.'
    & (Join-Path $PSScriptRoot 'convert.ps1') $ModelDir $OutputGguf @ConverterOptions
} finally {
    $env:LLAMA_CPP_CONVERT_PYTHON = $oldPython
}
