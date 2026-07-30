$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:GEM16_REPO_ROOT = $RepoRoot
& (Join-Path $RepoRoot "gradlew.bat") -p $RepoRoot :studioApp:run
exit $LASTEXITCODE
