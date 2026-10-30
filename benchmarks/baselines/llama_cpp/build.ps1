[CmdletBinding()]
param([ValidateRange(0, 64)][int]$Jobs = 0)
$ErrorActionPreference = "Stop"
$root = if ($env:LLAMA_CPP_HOME) { $env:LLAMA_CPP_HOME } else { 'C:/llama.cpp' }
$updater = Join-Path $root 'update-central.ps1'
if (-not (Test-Path -LiteralPath $updater)) { throw "Shared llama.cpp updater not found: $updater" }
if ($Jobs -gt 0) { & $updater -Jobs $Jobs } else { & $updater }
