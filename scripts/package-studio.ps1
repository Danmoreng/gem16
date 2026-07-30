$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $RepoRoot "gradlew.bat") -p $RepoRoot :studioApp:packageDistributionForCurrentOS
exit $LASTEXITCODE
