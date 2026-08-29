[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Server = Join-Path $RepoRoot "build\Windows\blackwell-release\bin\gem16-server.exe"
if (-not (Test-Path -LiteralPath $Server -PathType Leaf)) {
    throw "Build the Windows CUDA release server before packaging: $Server"
}
$Version = (Get-Content -LiteralPath (Join-Path $RepoRoot "VERSION") -Raw).Trim()
$ServerVersion = (& $Server --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw "Unable to read the release server version" }
if ($ServerVersion -ne "gem16-server $Version") {
    throw "Release server version mismatch: expected 'gem16-server $Version', got '$ServerVersion'"
}
. (Join-Path $PSScriptRoot "windows-toolchain.ps1")
Import-Gem16VisualStudioEnvironment
$Build = Join-Path $RepoRoot "build\Windows\native-studio-package"
$Stage = Join-Path $RepoRoot "build\packages\gem16-windows-x64"
$Archive = Join-Path $RepoRoot "build\packages\gem16-windows-x64.zip"
& cmake.exe -S (Join-Path $RepoRoot "nativeStudio") -B $Build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
if ($LASTEXITCODE -ne 0) { throw "Native Studio configure failed" }
& cmake.exe --build $Build --config Release --target gem16-studio --parallel
if ($LASTEXITCODE -ne 0) { throw "Native Studio build failed" }
New-Item -ItemType Directory -Path (Join-Path $Stage "bin") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Stage "licenses") -Force | Out-Null
Copy-Item (Join-Path $Build "bin\gem16-studio.exe") (Join-Path $Stage "bin") -Force
Copy-Item $Server (Join-Path $Stage "bin") -Force
Copy-Item (Join-Path $RepoRoot "VERSION") $Stage -Force
Copy-Item (Join-Path $RepoRoot "LICENSE") $Stage -Force
Copy-Item (Join-Path $RepoRoot "nativeStudio\third_party\imgui\LICENSE.txt") (Join-Path $Stage "licenses\Dear-ImGui-MIT.txt") -Force
Copy-Item (Join-Path $RepoRoot "nativeStudio\licenses\Free-Solace-ImGui-Interface-MIT.txt") (Join-Path $Stage "licenses") -Force
Copy-Item (Join-Path $RepoRoot "third_party\miniaudio\LICENSE") (Join-Path $Stage "licenses\miniaudio-MIT-0-or-Public-Domain.txt") -Force
if (Test-Path $Archive) { Remove-Item $Archive -Force }
Compress-Archive -Path "$Stage\*" -DestinationPath $Archive
$hash = Get-FileHash -LiteralPath $Archive -Algorithm SHA256
Write-Host "Created $Archive"
Write-Host "SHA256: $($hash.Hash)"
