[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
. (Join-Path $repoRoot "scripts\windows-toolchain.ps1")

$expectedCommit = (Get-Content -Raw (Join-Path $PSScriptRoot "commit.txt")).Trim()
$sourceDir = if ($env:LLAMA_CPP_SOURCE) { $env:LLAMA_CPP_SOURCE } else { Join-Path $repoRoot "third_party\cache\llama.cpp" }
$targetDir = if ($env:LLAMA_CPP_PATCHED_SOURCE) { $env:LLAMA_CPP_PATCHED_SOURCE } else { Join-Path $repoRoot "third_party\cache\llama.cpp-mixed" }
$patchFile = Join-Path $PSScriptRoot "patches\0001-support-mixed-fp8-nvfp4-compressed-tensors.patch"

if (-not (Test-Path -LiteralPath (Join-Path $sourceDir ".git"))) {
    throw "Pinned clean llama.cpp source was not found at $sourceDir"
}
$actualCommit = (& git.exe -C $sourceDir rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $expectedCommit) {
    throw "Source is not at pinned commit $expectedCommit"
}
$changes = & git.exe -C $sourceDir status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0 -or $changes) {
    throw "Source worktree must be clean"
}
if (Test-Path -LiteralPath $targetDir) {
    throw "Patched target already exists: $targetDir"
}

Invoke-Gem16Checked "git.exe" @("clone", "--shared", $sourceDir, $targetDir)
Invoke-Gem16Checked "git.exe" @("-C", $targetDir, "checkout", "--detach", $expectedCommit)
Invoke-Gem16Checked "git.exe" @("-C", $targetDir, "apply", "--unidiff-zero", "--check", $patchFile)
Invoke-Gem16Checked "git.exe" @("-C", $targetDir, "apply", "--unidiff-zero", $patchFile)
Write-Host "Prepared patched converter source at $targetDir" -ForegroundColor Green
