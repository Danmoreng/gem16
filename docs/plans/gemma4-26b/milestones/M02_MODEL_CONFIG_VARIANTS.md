# M02 — Model configuration and static variant traits

## Objective

Teach model inspection to recognize Gemma 4 26B A4B MoE while preserving an independently validated 12B Unified contract and compile-time-specialized hot paths.

## Why this milestone exists

The current primary-model validator is hard-coded to 12B dimensions and semantics. Replacing constants globally would break the existing engine. The correct boundary is a common parsed model description plus explicit supported variants and static runtime traits.

## Prerequisites

- M01 source config snapshots and exact dimensions are available.
- All existing configuration tests pass on the anchored tree.

## Repository areas to inspect first

- `src/model/config.h`
- `src/model/config.cpp`
- `tests/unit/config_test.cpp`
- `include/gem16/types.h`
- `src/cuda/engine/target_model.h`
- `src/cuda/engine/inference_engine.h`
- `src/runtime/capabilities.cpp`

## Suggested additions or boundaries

- `src/model/model_variant.h`
- `src/model/model_variant.cpp`
- `tests/fixtures/gemma4_26b_config.json`

## Implementation sequence

1. Introduce an enum such as `ModelVariant::{kGemma4Unified12B,kGemma4Moe26BA4B,kAssistant}` and a pure classification function.
2. Extend `ModelConfig` with `enable_moe_block`, `moe_intermediate_size`, `num_experts`, `top_k_experts`, router-related flags and any exact per-layer fields discovered from the pinned config.
3. Parse `Gemma4ForConditionalGeneration`, `gemma4`, `gemma4_text` without weakening the 12B identifiers.
4. Implement `ValidateGemma4Moe26BContract` with exact dimensions, layer type sequence, tied embedding, attention K=V setting, no KV sharing, context, RoPE and modality metadata.
5. Keep `ValidateGemma4Unified12BContract` semantically identical to the current validator.
6. Create host-side traits or immutable execution descriptors for the two supported target variants. The descriptor may be runtime data; CUDA hot kernels must still instantiate fixed shapes.
7. Replace ad-hoc global-layer arithmetic with the validated `layer_types` array at common boundaries.
8. Add capability flags for text-only 26B and explicitly false vision/audio/MTP support.
9. Ensure inspectable-but-unsupported Gemma 4 variants still fail with a precise reason rather than being misclassified.
10. Update manifest JSON output to include `model_variant` and all MoE dimensions.

## Required tests

- Positive parse/validate fixture from the pinned 26B config.
- Negative tests for 29/31 layers, wrong hidden, wrong expert count, top-k, shared intermediate, global KV heads and layer pattern.
- 12B config tests remain exact.
- Assistant model classification remains exact.
- Unknown Gemma 4 variants are inspectable but rejected for execution.
- Overflow/zero-value tests for all new dimensions.

## Evidence and documentation outputs

- Updated configuration contract documentation.
- Config fixture hashes tied to M01 source locks.
- A model-variant capability JSON example.

## Suggested commands

```text
ctest --test-dir build --output-on-failure
```
```text
gem16-inspect --model <qat-source> --validate --json
```

## Risks to watch in this milestone

- Reference config naming may change across Transformers revisions.
- A generic descriptor can accidentally move shape checks from compile time to the token loop.

## Forbidden shortcuts

- Do not change `kTargetLayerCount` to 30 globally.
- Do not make unsupported dimensions silently dynamic.
- Do not treat `intermediate_size=2112` as the routed expert width.
- Do not infer local/global layers from modulo arithmetic after parsing.

## Exit criteria

- [ ] 12B and 26B are distinct variants.
- [ ] Every 26B architectural dimension is validated.
- [ ] No CUDA kernel is selected by loosely comparing a file name or model directory.
- [ ] Existing 12B and assistant tests pass unchanged.
- [ ] M03 and common runtime work can consume one explicit variant field.

## Downstream milestones unblocked

- M03

## Codex execution prompt

```text
You are implementing M02: Model configuration and static variant traits in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M02. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M02 exit criterion passed. Stop before starting the next milestone.
```
