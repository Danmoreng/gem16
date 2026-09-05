# Windows source update and local-model smoke — 2026-09-05

Scope: local development validation, not release qualification or a performance claim.

## Source and environment

The clean `main` checkout was updated using `git fetch origin` and
`git merge --ff-only origin/main`, from `4928f3f` to
`56e83c3906d89a7e4941af37aa3a768e912ae088`. No commit was created.
CUTLASS remains `db1c288993354c88e551c40c19a8fb93a774a241`.

Windows x64, RTX 5080 Laptop GPU (16,303 MiB), driver 596.49,
MSVC 19.44.35219.0, CUDA toolkit 13.3.33, CMake 4.1.2.
The existing native Studio build directory uses Visual Studio 17 2022;
the host/CUDA presets use Ninja. Reuse each directory's existing generator.

## Fix

`nativeStudio/src/canvas_browser.cpp` now includes Windows and COM definitions
before `WebView2.h`. The pinned WebView2 SDK's forward declarations use
`interface` before its later COM includes; the previous include order failed
with MSVC C4430/C2146. The full Windows Studio target compiles the affected
backend and now builds successfully. No inference or numerical code changed.

## Checks and commands

Run from the repository root in PowerShell:

```powershell
.\scripts\build.ps1 -Test -Jobs 8
.\scripts\build.ps1 -Cuda -Test -Jobs 6
. ./scripts/windows-toolchain.ps1
Import-Gem16VisualStudioEnvironment
cmake -S nativeStudio -B build/Windows/native-studio -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/Windows/native-studio --config Release --parallel 6
ctest --test-dir build/Windows/native-studio -C Release --output-on-failure
$env:GEM16_STUDIO_TEST_VISION_CACHE="$env:USERPROFILE/.cache/huggingface/hub"
ctest --test-dir build/Windows/native-studio -C Release -R gem16-native-studio-host --output-on-failure
$env:GEM16_12B_MODEL="$env:USERPROFILE/.cache/huggingface/hub/.gem16/snapshots/unsloth--gemma-4-12b-it-NVFP4--b1f649734b34aa5575b03d186abd1b9be3d0d5c4"
$env:GEM16_M22_RAW_DIR="$PWD/build/windows-12b-product-20260905"
ctest --preset blackwell-release -R '^gem16-12b-m22-product$' --output-on-failure
```

- Host: 5/5 passed.
- CUDA preset: 11 passed, 4 explicitly skipped; no failures. The optional
  NVFP4 consumption and historical 26B M17/M22 tests had no model configuration.
- The initially skipped 12B product test was then run with the real model and
  passed: exact output `[9503,106]`, two resident sessions, continuation,
  zero reported fallbacks and token-loop allocations.
- Studio: 3/3 passed, plus the real Compact Vision cache check passed with
  Target, Vision and Assistant ready and zero required download bytes.
- Compact Vision fixed-D2 CLI and HTTP text smokes returned `56` for 7 times 8.
- HTTP image smoke used `benchmarks/vision-v19/images/scene-square.png` and
  its suite prompt: yellow robot, green chair, ball under table, five stars;
  all four checks passed. Context 4,096, image budget 280, greedy, reasoning off.
- A separate fixed-D2 CLI smoke at the Studio default capacity of 229,120
  succeeded. This reserves capacity; it does not exercise a full-length prompt.
- Hidden Studio startup with isolated settings reached input idle and remained
  responsive. It was terminated after the check; interactive rendering, Canvas
  behavior and normal GUI shutdown were not qualified by this smoke.

The 26B server reported native Trellis35 W4A8 dispatch, FP8 Vision loaded,
one slot, and 2,114,977,792 free device bytes after admission at context 4,096.
No benchmark distribution or peak-VRAM measurement was collected.

## Cache reconciliation

The runtime locks still pin 26B revision
`6de2a057f11332420819f8e6efd08e42d7a03bc7`, already present locally.
The newer normalized Hub revision is deliberately not substituted. Runtime
views retain `model.gem16`, `vision.gem16` and the locked Assistant shard name.

Full SHA-256 checks passed for all 49 present files across the five public-profile
component locks (23,341,102,699 bytes checked). Only the Vision publication's
`NOTICE` and `README.md` were absent; neither is part of its strict four-file
runtime catalog. Existing canonical blobs and composed views already matched.
Verification markers were refreshed after hashing. No model download, payload
copy, rename, lock migration or precision conversion was needed.

Local raw logs, exact smoke commands/scripts and JSON responses remain under
`build/windows-*-20260905*`, `build/audit_model_cache_20260905.py`,
`build/prepare_model_views_20260905.py` and `build/smoke_26b_api_20260905.py`.
These are ignored local development evidence. Full SDK/Pi parity, extended
quality/long-context tests, packaging and clean-machine release gates remain open.
