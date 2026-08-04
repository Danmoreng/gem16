# Repository file change map

This map is directional. The agent must inspect the current tree and write a drift note before using it.

## Model parsing and contracts

| Area | Expected work | Milestones |
|---|---|---|
| `src/model/config.h/.cpp` | parse MoE/shared/router/per-layer fields; variant validation | M02 |
| `src/model/manifest.h/.cpp` | 26B inventory, roles, compiled schema, omission | M03, M08 |
| `src/model/checkpoint_loader.cpp` | select contract, provenance validation | M02, M08 |
| `include/gem16/types.h` | logical role/layout/provenance metadata | M03, M08 |
| new `src/model/model_variant.*` | immutable traits and dispatch | M02 |
| new `src/model/gemma4_26b_contract.*` | exact 26B validation | M02, M03 |

## Compiler/tools

| Area | Expected work | Milestones |
|---|---|---|
| `tools/fetch_model.py` | new locks/schema support | M01, M22 |
| new `tools/compile_gemma4_26b.py` | compiler CLI | M04–M08 |
| new `tools/gem16_compile/` | plan/read/write/quantizers/reports | M04–M08 |
| new comparison/evaluation tools | A/B, quality, long context | M18–M21 |
| `models/` | four source locks and final derived lock | M01, M08, M19 |

## Runtime memory and ownership

| Area | Expected work | Milestones |
|---|---|---|
| `src/runtime/memory_plan.*` | 26B traits, selected residency, workspace | M09 |
| `include/gem16/memory.h` | named counters/profile fields | M09 |
| `src/server/session_pool.*` | slot admission and sharing | M09, M17, M22 |
| `include/gem16/engine.h` | variant/path/metrics options | M17, M22 |

## Loader/bindings

| Area | Expected work | Milestones |
|---|---|---|
| `src/cuda/engine/target_model.h/.cu` | dynamic traits, MoE bindings, quantized head | M08 |
| `src/cuda/engine/inference_engine.h/.cu` | 26B reference and optimized plans | M09–M17 |
| `src/cuda/inference.cu` | orchestration/results | M13, M17 |

## NVFP4/FP8

| Area | Expected work | Milestones |
|---|---|---|
| `src/numeric/fp8.cpp` | compiler-compatible reference | M05 |
| `src/numeric/nvfp4.cpp` | compiler-compatible reference | M06 |
| `src/cuda/fp8/*` | new dimensions/traits, existing arithmetic | M12 |
| `src/cuda/nvfp4/*` | shared/expert shapes and reuse | M11, M14, M15 |
| `src/cuda/nvfp4/sm120_layout.*` | expert-major layout validation | M06, M14 |

## New MoE area

```text
src/cuda/moe/
  moe.h
  reference.cu
  router_reference.cu
  router_sm120.cu
  decode_sm120.cu
  prefill_router.cu
  prefill_permute.cu
  prefill_sm120.cu
  prefill_reduce.cu
```

Milestones M10–M15.

## Attention/KV

| Area | Expected work | Milestones |
|---|---|---|
| `src/cuda/attention/*` | 26B head shapes/traits | M12 |
| `src/cuda/kv_cache/*` | ownership and shape tests | M12 |
| `src/cuda/rope/*` | per-layer config | M12 |

## Embedding/head

```text
src/numeric/q4_0.*
src/cuda/embedding/lookup.*
src/cuda/output_head_q4_0.cu
src/cuda/output_head_nvfp4.cu
```

Refactor current `src/cuda/output_head.*` only enough to share stable candidate/argmax semantics. M07, M16.

## Product

| Area | Expected work | Milestones |
|---|---|---|
| `src/cli/*` | model profile/options/errors | M22 |
| `src/server/*` | capabilities/admission/API | M22 |
| `studioApp/` | model catalog/download/UI | M22 |
| `docs/STUDIO.md`, `docs/SERVER.md` | user behavior | M22 |

## Build/tests

- add new sources in `CMakeLists.txt`;
- keep CUDA architecture-specific compile options localized;
- add host tests under `tests/unit/`;
- add CUDA tests under `tests/cuda/`;
- add Python compiler/evaluator tests;
- do not put unrelated refactors into the same PR.

## Documentation

Expected new repository docs:

```text
docs/GEMMA4_26B.md
docs/GEMMA4_26B_CHECKPOINT.md
docs/GEMMA4_26B_MEMORY.md
docs/GEMMA4_26B_CORRECTNESS.md
docs/GEMMA4_26B_RUNTIME.md
docs/GEMMA4_26B_QUALITY.md
docs/GEMMA4_26B_BENCHMARKING.md
docs/GEMMA4_26B_LONG_CONTEXT.md
```

Update existing `DECISIONS`, `CORRECTNESS`, `MEMORY`, `BENCHMARKING`, `PERFORMANCE_LEDGER`, `ROADMAP` at the relevant milestone, not all at the end.
