# Current repository baseline

## Snapshot

This plan is anchored to:

```text
repository: Danmoreng/gem16
branch: main
commit: 1c4287965d318ba32a68e597f9d7b6678b883376
snapshot date: 2026-08-04
```

The implementation agent must compare the actual working tree against this commit and record changed paths that affect the plan.

## Existing product shape

At the snapshot, `gem16` is a specialized Gemma 4 12B Unified runtime with:

- direct mixed FP8/NVFP4 Safetensors loading;
- 48 dense decoder layers;
- hidden size 3840 and dense intermediate size 15360;
- FP8 attention projections;
- packed NVFP4 gate/up/down MLPs;
- BF16 tied embedding/output matrix;
- FP8 or BF16 KV;
- local/global hybrid attention;
- CUDA Graph batch-one decode;
- chunked native SM120 prefill;
- optional official BF16 MTP assistant;
- text, image and audio paths;
- CLI, desktop application and OpenAI-compatible server;
- strict fixed-arena and no-token-loop-allocation policy.

The 26B track must not convert this into a generic engine that regresses the existing path.

## Important current code assumptions

### Model configuration

`src/model/config.cpp` validates one hard-coded primary 12B Unified contract. It expects:

- architecture `Gemma4UnifiedForConditionalGeneration`;
- model type `gemma4_unified`;
- text model type `gemma4_unified_text`;
- 48 layers;
- hidden 3840;
- intermediate 15360;
- 16 query heads;
- 8 local KV heads and 1 global KV head;
- local/global head dimensions 256/512;
- five local layers followed by one global layer, repeated eight times.

The 26B model instead uses `Gemma4ForConditionalGeneration`, `gemma4`, `gemma4_text`, 30 layers, hidden 2816, a 2112-wide always-active dense MLP, 128 routed experts of width 704 and top-8 routing.

### Tensor manifest

`src/model/manifest.cpp` currently:

- compiles quantization target regexes;
- classifies FP8 and NVFP4 tensor families;
- identifies text-only tensors;
- validates a dense layer inventory with one gate/up/down family;
- does not yet understand router tensors, per-expert scales or fused 3D expert storage.

This validator must be split by model variant.

### Target model binding

`src/cuda/engine/target_model.h/.cu` currently contains:

- `kTargetLayerCount = 48`;
- fixed constants for 3840/15360/262144;
- one `Nvfp4Binding` each for gate/up/down;
- one `LayerBinding` per dense layer;
- global-layer detection from `index % 6 == 5`;
- BF16 tied embedding/output binding;
- direct load-time NVFP4 transformation into Row8/K64 order.

Do not simply replace these constants with 26B values. Preserve a 12B specialization and introduce a 26B MoE specialization selected at initialization.

### Memory planning

`src/runtime/memory_plan.cpp` computes hybrid local/global K/V and requires separate physical K and V. It currently reconciles resident bytes against the full manifest payload. The 26B text-only path needs:

- an explicit residency class;
- omitted vision bytes;
- compiled-only/source-only tensor reporting;
- tied embedding alias accounting;
- per-profile output-head format accounting;
- MoE workspace accounting.

### Output head

`src/cuda/output_head.cu` assumes a BF16 matrix with hidden 3840 and vocabulary 262144. The 26B work requires:

- dimension traits;
- Q4_0 and/or NVFP4 matrix binding;
- exact softcap;
- direct candidate reduction;
- no full persistent logits;
- lowest-token tie behavior.

### Build and tests

`CMakeLists.txt` already separates CUDA source files and has host plus CUDA test executables. New 26B files should follow that structure rather than growing `inference_engine.cu` indefinitely.

## Baseline invariants to retain

Every 26B milestone must keep these invariants green unless a repository decision explicitly changes them:

- all existing host CTest targets pass;
- all existing CUDA operator tests pass;
- 12B short and long deterministic output hashes remain exact;
- 12B allocator accounting remains unchanged unless the PR intentionally changes shared infrastructure;
- no new token-loop allocation;
- no silent precision fallback;
- direct 12B source checkpoint loading remains supported;
- MTP, vision, audio, server and Studio behavior remain unchanged for 12B.

## Suggested architectural boundary

Prefer this shape:

```text
common model inspection and tokenizer
       │
       ├── DenseUnified12B traits + loader + engine
       └── Moe26BA4B traits + loader + engine
```

Share code only where the arithmetic and memory contract are truly identical. Duplicate a small amount of orchestration rather than inserting dynamic branches into every hot layer.

## Existing tooling to extend

- `tools/fetch_model.py`: immutable lock download and verification;
- `gem16-inspect`: tensor inventory and validation;
- `gem16-run`: teacher forcing, state/logit dumps and generation;
- `gem16-bench`: resident prefill/decode distributions;
- benchmark scripts: telemetry and raw-result retention;
- `docs/DECISIONS.md`: accepted choices;
- `docs/PERFORMANCE_LEDGER.md`: measured experiments;
- `toolchains/blackwell16gb.lock`: exact toolchain.

## Required drift report

Before M00 edits, create an evidence note containing:

```text
base commit from this plan
actual HEAD
files changed between them
open 26B-related PRs/issues
changes to AGENTS.md
changes to config/manifest/loader/memory/output-head/CMake
changes to pinned CUDA/CUTLASS/toolchain
changes to current reference GPU/driver
```

Update this plan only when the drift materially invalidates milestones; do not mechanically rewrite it for unrelated UI commits.
