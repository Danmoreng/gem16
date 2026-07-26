Set-StrictMode -Version Latest

function Invoke-Gem16Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @()
    )

    Write-Host "> $FilePath $($Arguments -join ' ')"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Import-Gem16VisualStudioEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw "vswhere.exe was not found. Install Visual Studio 2022 Build Tools with the C++ workload."
    }

    $vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($vsRoot)) {
        throw "Visual Studio Build Tools with the x64 C++ toolchain were not found."
    }

    $vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path -LiteralPath $vcvars)) {
        throw "vcvars64.bat was not found at $vcvars"
    }

    Write-Host "Loading Visual Studio C++ environment from $vcvars" -ForegroundColor DarkCyan
    $environment = & cmd.exe /d /s /c "`"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) {
        throw "vcvars64.bat failed with exit code $LASTEXITCODE"
    }

    # Windows environment names are case-insensitive, while `set` can expose
    # PATH and Path entries in an order that loses the MSVC update. Import the
    # compiler-containing value last. This pattern is adapted from the
    # neighboring qwen-tts-studio Windows build helper.
    $pathCandidates = @()
    foreach ($line in $environment) {
        if ($line -match "^([^=]+)=(.*)$") {
            if ($Matches[1] -ieq "Path") {
                $pathCandidates += $Matches[2]
            } else {
                [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
            }
        }
    }

    $msvcPath = $null
    foreach ($candidate in $pathCandidates) {
        foreach ($entry in ($candidate -split [IO.Path]::PathSeparator)) {
            if ($entry -and (Test-Path -LiteralPath (Join-Path $entry "cl.exe"))) {
                $msvcPath = $candidate
                break
            }
        }
        if ($msvcPath) { break }
    }
    if (-not $msvcPath) {
        throw "vcvars64.bat did not expose cl.exe on PATH."
    }
    $env:Path = $msvcPath

    foreach ($toolDirectory in @(
        (Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"),
        (Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja")
    )) {
        if ((Test-Path -LiteralPath $toolDirectory) -and
            (($env:Path -split [IO.Path]::PathSeparator) -notcontains $toolDirectory)) {
            $env:Path = $toolDirectory + [IO.Path]::PathSeparator + $env:Path
        }
    }

    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "MSVC compiler cl.exe was not found after importing the Visual Studio environment."
    }
}

function Import-Gem16CudaEnvironment {
    $candidates = @()
    if ($env:CUDA_PATH) {
        $candidates += $env:CUDA_PATH
    }
    $candidates += Get-ChildItem Env: -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "CUDA_PATH_V*" } |
        Sort-Object Name -Descending |
        ForEach-Object { $_.Value }

    $toolkitParent = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path -LiteralPath $toolkitParent) {
        $candidates += Get-ChildItem -LiteralPath $toolkitParent -Directory |
            Sort-Object Name -Descending |
            ForEach-Object { $_.FullName }
    }

    $cudaRoot = $null
    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-Path -LiteralPath (Join-Path $candidate "bin\nvcc.exe") -PathType Leaf) {
            $cudaRoot = $candidate
            break
        }
    }
    if (-not $cudaRoot) {
        throw "The NVIDIA CUDA toolkit was not found. Install the pinned toolkit or set CUDA_PATH."
    }

    $env:CUDA_PATH = $cudaRoot
    $env:CUDAToolkit_ROOT = $cudaRoot
    $cudaBin = Join-Path $cudaRoot "bin"
    if (($env:Path -split [IO.Path]::PathSeparator) -notcontains $cudaBin) {
        $env:Path = $cudaBin + [IO.Path]::PathSeparator + $env:Path
    }
}

function Assert-Gem16Command {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$InstallHint
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found on PATH. $InstallHint"
    }
}
