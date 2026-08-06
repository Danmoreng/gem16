# Gemma 4 26B M00 repository drift report

**Date:** 2026-08-06

**Milestone:** M00 — Policy and 26B track bootstrap

**Plan anchor:** `1c4287965d318ba32a68e597f9d7b6678b883376`

**Current commit:** `b078a153772d12711032ef1b940221e825ce36e5`

**Branch:** `feat/26b-m00-policy`

**Dirty state before M00 edits:** clean

**Open GitHub PRs:** none reported by `gh pr list --state open`
**Open GitHub issues matching `26B`:** none reported by `gh issue list --state open --search 26B`

## Relevant changed files

| File/area | Anchor behavior | Current behavior | Plan impact |
|---|---|---|---|
| `AGENTS.md` | 2,438-line repository-initialization specification with a repository-wide direct-load assumption and stale feature state | 124-line policy-only contract; explicitly permits reproducible profile-specific compiled artifacts while forbidding runtime conversion, undisclosed copies and misleading provenance | Validates M00's compiled-artifact premise. The stricter 26B contract must live in the track document and accepted decision. |
| `docs/plans/gemma4-26b/` | Not yet present at the anchor | Complete M00–M25 plan package added by `0bb0dd1`, including imp amendment and M18-before-M14 quality kill gate | Use the package unchanged as the binding implementation program; M00 is the sole active milestone. |
| `src/model/config.cpp`, `manifest.cpp`, `checkpoint_loader.cpp` | One strict 12B Unified primary contract and assistant support | No changes since the anchor; still no 26B variant, router/expert inventory or compiled-artifact schema | Plan assumptions remain accurate. M02/M03/M08 are still required; M00 must not add implementation. |
| `src/runtime/memory_plan.cpp` | 12B manifest residency and hybrid separate K/V planning | No changes since the anchor | M09 remains required. Current all-regions 12B evidence strengthens the measured CUDA-visible reserve policy but does not prove 26B fit. |
| `src/cuda/engine/target_model.*`, `src/cuda/output_head.cu` | 48-layer, hidden-3840, intermediate-15360 and BF16 tied-head specialization | No changes since the anchor; constants and dense bindings remain 12B-specific | Confirms that 12B must not be generalized during M00. M02/M12/M16/M17 boundaries remain valid. |
| `src/cuda/engine/inference_engine_memory.cuh`, `inference_engine_prefill_api.cuh` | Full-chunk final prompt normalization workspace | Final RMSNorm stores only the required five-row MTP extent and normalizes the last ordinary prompt row | Current 12B memory/performance baseline changed; preserve its exact tests and accounting. No 26B plan invalidation. |
| `src/cuda/engine/inference_engine_prefill_layers.cuh`, `src/cuda/mtp/verify.*`, `src/cuda/sampling/sampling.cu` | Earlier sampled-MTP verifier boundaries and captured suppression count | Restored verifier BF16 V/O boundaries and dynamic suppression semantics; sampled D1/D2/D4 identity is corrected | Add these exact 12B sampled/MTP gates to later shared-infrastructure regression work. M00 arithmetic remains unchanged. |
| `src/cuda/nvfp4/cutlass_sm120.*` | Partial final Up epilogue tile could expose invalid packed bytes | Padded physical final M tile is validated and used while logical consumers retain the requested token count | Existing NVFP4 arithmetic/layout evidence is stronger; compiler/runtime work must preserve the qualified boundary. |
| `tests/cuda/nvfp4_reference_test.cu`, prefill golden | Earlier final-sprint fixtures | Expanded tail/boundary coverage and vLLM 0.26 prefill fixture | Later 26B changes must retain the current test baseline. |
| `CMakeLists.txt`, `CMakePresets.json` | Host and SM120a Ninja presets; no 26B option | Unchanged since the anchor; no 26B source or feature flag exists | No M00 feature flag is needed because there is no incomplete implementation to expose. Add an experimental default-off boundary before M02 exposes runtime behavior. |
| `toolchains/blackwell16gb.lock`, CUTLASS submodule | CUDA 13.3.73, driver 610.43.03, CUTLASS v4.5.2 at `db1c288...` | Lock and submodule unchanged | Compiler/kernel milestones can use the planned toolchain identity after fresh milestone verification. |
| 12B benchmark/memory docs and evidence | Pre-final-sprint references | Corrected 16K D2 qualification, Windows regression, Linux short-context closure and direct all-regions reserve are recorded | The previous blocker before M00 is closed. Current 12B behavior becomes the regression baseline. |

## Contract changes

### Model/checkpoint

- Repository policy now permits immutable upstream or reproducible project-compiled profile artifacts.
- The direct 12B source checkpoint and assistant locks remain unchanged and remain the default product.
- No 26B source lock, derived lock, compiler schema or runtime loader exists yet.
- The plan's required QAT BF16, ordinary BF16, Unsloth NVFP4 and official Q4_0 source identities remain M01 work.

### Runtime/memory

- No core memory planner change occurred since the anchor.
- The current 12B all-regions context-17,519 probe records 4,335,665,152 CUDA-visible bytes free with Target,
  assistant, fixed D2, recommended sampling, decode graphs, media workspace, FP8 KV and the final prompt plan.
- Final prompt normalization removed 125,752,320 named workspace bytes from the production 16K plan.
- None of this establishes 26B residency; M03's synthetic 32K admission and M09's real-artifact plan remain hard
  gates.

### CUDA kernels

- No 26B kernel or model dispatch was added.
- Current 12B changes are bounded correctness/performance fixes in final prompt normalization, MTP verifier
  boundaries/suppression and CUTLASS NVFP4 tail handling.
- `target_model` and output-head specialization remain exactly 12B-shaped.

### Product/tests/docs

- Image/audio, sampled MTP, server sessions and Studio are mature 12B regression surfaces compared with the plan
  snapshot.
- The corrected Linux 16K D2 and closed Linux short-context records supersede earlier performance figures.
- Host and Blackwell suites passed at the M00 parent before branch creation.
- The short-context plan was removed after durable closure evidence; 26B M00 is now the active roadmap item.

## Environment drift from the lock

| Item | Locked reference | M00 observation | Impact |
|---|---|---|---|
| Kernel | `7.1.3-arch1-3` | `7.1.5-arch1-2` | Record for M00 only; refresh or explicitly accept before promoted compiler/performance evidence. |
| CMake | `4.4.0` | `4.4.2` | No source/build-contract change observed; exact release/compiler runs remain lock-bound. |
| CUDA toolkit | `13.3.73` | `13.3.73` | Matches. |
| NVIDIA driver | `610.43.03` | `610.43.03` | Matches. |
| GPU/UUID | RTX 5080 Laptop / `GPU-93070293-2184-eb69-555b-1d856a4fdbb8` | Match | Reference hardware unchanged. |
| CUTLASS | v4.5.2 / `db1c288993354c88e551c40c19a8fb93a774a241` | Match | Dependency unchanged. |

## Plan decisions

- **Still applicable:** 12B/26B static specialization, one resident weight representation, offline deterministic
  compiler, text-only initial profile, no MTP/vision/offload, source/compiler/output locks, M18 quality kill gate.
- **Adapt current milestone:** use current `b078a15` as M00 parent and record all post-anchor 12B regression facts.
- **Prior milestone must be reopened:** none.
- **Plan assumption invalid:** none material.
- **Feature flag:** deferred to the first milestone that introduces 26B code; M00 adds no dead build option.

## New risks

- The active Arch kernel and CMake patch version differ from the toolchain lock; promoted evidence must either restore
  the lock or update it through an explicit toolchain decision.
- The current source contains many intentional 12B shape constants. Premature commonization remains a high
  regression risk.
- Distribution terms for a QAT-derived artifact are not resolved by technical provenance and remain owner/legal
  work in M01.
- Optional imp donor code has an MIT provenance plan but no copying approval; accidental architectural import must
  be prevented.

## Required owner decision

Owner acceptance is required for the concrete artifact/runtime/baseline contract in
[`docs/GEMMA4_26B.md`](../../GEMMA4_26B.md) and the matching decision entry before M00 can pass and M01 can become
ready. M00 does not approve model distribution or copying imp source.

## Evidence

Commands/diffs inspected:

```text
git status --short --branch
git log --oneline 1c4287965d318ba32a68e597f9d7b6678b883376..HEAD
git diff --name-status 1c4287965d318ba32a68e597f9d7b6678b883376..HEAD
git diff --stat 1c4287965d318ba32a68e597f9d7b6678b883376..HEAD -- <M00 touchpoints>
git diff 1c4287965d318ba32a68e597f9d7b6678b883376..HEAD -- <changed CUDA files>
git submodule status
sha256sum toolchains/blackwell16gb.lock
rg -n "3840|15360|262144|kTargetLayerCount|ModelVariant|26B" src include tests CMakeLists.txt
gh pr list --state open --limit 100
gh issue list --state open --limit 100 --search 26B
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.free --format=csv,noheader,nounits
```
