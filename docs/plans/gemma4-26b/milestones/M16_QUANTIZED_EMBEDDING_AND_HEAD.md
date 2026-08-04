# M16 — Production quantized embedding and output head

## Objective

Promote the selected M07 tied-matrix format into production-quality lookup, greedy, sampling and short-batch output-head paths with one resident allocation.

## Why this milestone exists

The tied matrix saves roughly one GiB versus BF16 and is required for residency. Its output path must cover not only greedy T=1 but also sampling diagnostics and later MTP-style T≤5 verification without changing product semantics.

## Prerequisites

- M07 provisional decision
- M13 full-model reference
- M18 preliminary quantizer/quality kill gate passed
- M09 final arena contract

## Repository areas to inspect first

- `src/cuda/output_head.h`
- `src/cuda/output_head.cu`
- `src/cuda/sampling/sampling.cu`
- `src/cuda/engine/target_model.h`
- `src/cuda/engine/inference_engine.cu`
- `include/gem16/engine.h`

## Suggested additions or boundaries

- `src/cuda/embedding/lookup.cu`
- `src/cuda/output_head_q4_0.cu`
- `src/cuda/output_head_nvfp4.cu`
- `tests/cuda/gemma4_26b_output_head_test.cu`

## Implementation sequence

1. Define a format-tagged tied binding that stores one payload, its scales and logical dimensions.
2. Implement token lookup directly to the engine's required BF16/FP32 hidden boundary.
3. Implement fused projection, final logit softcap, suppression and blockwise candidate reduction for T=1.
4. Implement T=3 and T=5 batch candidate paths or a generic bounded T≤5 path for diagnostics and future MTP.
5. Implement full-logit materialization only when requested for teacher forcing or sampling; reuse the existing fixed sampling arena.
6. Preserve the lowest-token tie break and all existing suppression/EOS behavior.
7. Capture the ordinary greedy head in the whole-model decode graph using fixed addresses.
8. Benchmark the promoted format against the alternate M07 format again inside the full model, not only in isolation.
9. Record actual weight bytes and prove no BF16 head copy survives startup.
10. Retain the alternate implementation behind an explicit experimental profile until M19 closes the final decision.

## Required tests

- Lookup rows match CPU dequantization for boundary and random token IDs.
- Full logits and fused candidate reduction select the same token.
- Suppression, softcap, tie break and diagnostic logit tests.
- T=1/T=3/T=5 comparisons versus BF16 and M07 reference.
- Greedy and sampled full-model regression suite.
- Graph capture/replay and no allocation in token loop.
- One pointer/one allocation invariant for tied input/output weights.
- Process VRAM proves BF16 head removal.

## Evidence and documentation outputs

- `artifacts/m16/output-head-correctness.json`
- `artifacts/m16/output-head-full-model-ab.json`
- `artifacts/m16/output-head-memory.json`
- `artifacts/m16/output-head-nsight/`

## Suggested commands

```text
build/blackwell-release/bin/gem16-bench output-head --model "$GEM16_26B" --formats selected,alternate --rows 1,3,5 --warmups 3 --repetitions 10 --json artifacts/m16/output-head-full-model-ab.json
```
```text
python tools/validate_gemma4_26b_head.py --model "$GEM16_26B" --reference "$QAT_BF16" --output artifacts/m16/output-head-correctness.json
```

## Risks to watch in this milestone

- The isolated fastest head may not minimize whole-model ITL.
- Sampling requires full or sorted logits and can change workspace substantially.
- A W4A4 head adds activation error beyond Google's Q4_0 training target.
- Batch verifier geometry can regress T=1 if coupled carelessly.

## Forbidden shortcuts

- Keeping a hidden BF16 head for unsupported operations.
- Skipping vocabulary rows or softcap.
- Using a separate input embedding copy.
- Promoting the alternate format without M19 quality evidence.
- Making sampling silently fall back to CPU.

## Exit criteria

- [ ] Selected tied format supports lookup, greedy, sampling diagnostics and T≤5.
- [ ] One resident allocation serves embedding and output.
- [ ] No BF16 duplicate survives.
- [ ] Full-model correctness and deterministic generation remain within the accepted envelope.
- [ ] Head memory and timing are measured and reported.
- [ ] Alternate profile remains available for M19 but is not resident simultaneously.

## Downstream milestones unblocked

- M17 whole-model graph
- M19 final quality selection
- M20 performance

## Codex execution prompt

```text
You are implementing M16: Production quantized embedding and output head in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M16. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M16 exit criterion passed. Stop before starting the next milestone.
```
