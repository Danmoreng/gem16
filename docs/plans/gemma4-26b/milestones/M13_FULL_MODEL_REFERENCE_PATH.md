# M13 — Complete slow 26B text inference path

## Objective

Assemble embedding, all 30 attention layers, shared MLP, routed experts, final norm and output head into a complete deterministic text-only inference path using correctness-first operators.

## Why this milestone exists

A full, slow model establishes end-to-end correctness before native MoE optimization. It is the first point where teacher forcing, prompt rendering, cache continuity and generation drift can be evaluated together.

## Prerequisites

- M08 loader
- M09 arenas
- M11 CUDA MoE reference
- M12 attention/KV
- M07 reference head

## Repository areas to inspect first

- `src/cuda/engine/inference_engine.cu`
- `src/cuda/inference.cu`
- `include/gem16/engine.h`
- `src/runtime/chat.cpp`
- `src/model/tokenizer.cpp`
- `src/cuda/output_head.cu`
- `tools/validate_layer_checkpoint.py`
- `docs/CORRECTNESS.md`

## Suggested additions or boundaries

- `tools/validate_gemma4_26b_full_model.py`
- `benchmarks/goldens/gemma4_26b/full_model/`
- `docs/GEMMA4_26B_CORRECTNESS.md`

## Implementation sequence

1. Add an explicit 26B correctness execution plan selected once after model validation.
2. Bind the selected quantized embedding/head reference implementation and all layer-specific traits.
3. Execute prompt prefill in bounded small chunks using reference operators; performance is not an objective.
4. Execute decode with fixed-address buffers and no token-loop allocation even in correctness mode.
5. Add teacher-forcing capture of final logits, selected hidden states, router outputs and per-layer residual boundaries.
6. Integrate tokenizer, chat template, EOS, suppression and final-logit softcap without changing the existing product semantics.
7. Validate resident conversation suffix prefill and exact cache-prefix ownership.
8. Compare against pinned trusted QAT BF16, ordinary BF16, Unsloth NVFP4 and official Q4_0 reference outputs using the metrics defined in the quality spec.
9. Run a preliminary held-out-free quality kill screen on the development corpus: teacher-forced NLL/KL, router drift, greedy fixtures and a small prose-first task/perplexity subset. If the project-compiled NVFP4 experts show a material unexplained loss, block native performance work and route only to M18 diagnosis.
10. Add deterministic short and long generation fixtures. Do not require universal cross-runtime token identity where arithmetic differs; require distribution and task gates.
11. Label all timing output `correctness_reference` and `benchmark_qualified=false`.

## Required tests

- Exact tokenizer and rendered prompt IDs against the pinned model assets.
- Teacher-forced logits and state captures over a multi-prompt suite.
- Router top-8 and per-layer residual drift report.
- Preliminary development-corpus quality screen against QAT BF16, official Q4_0 and Unsloth; final held-out thresholds remain M18/M19 work.
- Greedy deterministic output over repeated fresh processes.
- Resident two-turn chat with suffix-only prefill.
- 32K allocation and short forward smoke test if runtime is sufficiently fast; full long-context qualification waits for M21.
- No NaN/Inf and no token-loop allocation.
- Every existing 12B host/CUDA/chat/server gate remains green.

## Evidence and documentation outputs

- `docs/GEMMA4_26B_CORRECTNESS.md`
- `artifacts/m13/full-model-teacher-forcing.json`
- `artifacts/m13/generation-fixtures.json`
- `artifacts/m13/allocation-trace.json`
- Pinned golden payloads with source lock and capture-tool commit.
- `artifacts/m13/preliminary-quality-screen.json` with an explicit proceed/stop result.

## Suggested commands

```text
python tools/validate_gemma4_26b_full_model.py --model "$GEM16_26B" --reference "$QAT_BF16" --mode teacher-forcing --output artifacts/m13/full-model-teacher-forcing.json
```
```text
build/blackwell-debug/bin/gem16-run --model "$GEM16_26B" --tokens-file benchmarks/prompts/gemma4-26b-smoke.json --max-context 4096 --json artifacts/m13/smoke.json
```

## Risks to watch in this milestone

- Slow reference execution can make large validation suites impractical; use selected captures without weakening coverage.
- Different head formats can dominate token-selection drift.
- A correct per-layer implementation can still fail because of prompt/template or cache-prefix mistakes.
- Debug timing may be mistakenly compared with optimized runtimes.

## Forbidden shortcuts

- Optimizing by removing diagnostics before the reference is accepted.
- Calling reference timings a performance baseline.
- Silently disabling MoE, shared MLP, global attention or quantized head.
- Falling back to CPU expert execution.
- Loosening 12B gates to accommodate 26B.

## Exit criteria

- [ ] Complete text-only prompt and generation execute on the 16 GB reference GPU.
- [ ] Teacher-forced numerical reports satisfy the provisional correctness envelope.
- [ ] Generation is deterministic under deterministic settings.
- [ ] Resident conversation and cache-prefix tests pass.
- [ ] No token-loop allocation or CPU weight offload occurs.
- [ ] The path remains explicitly unqualified for performance.
- [ ] Preliminary quality kill screen passes, or M14–M17 remain blocked while M18 diagnoses the quantizer/source/head attribution.

## Downstream milestones unblocked

- M18 converter A/B diagnosis always
- M14–M17 native performance path only after the preliminary quality kill screen passes
- M19 final quality qualification remains blocked until optimized integration and frozen thresholds

## Codex execution prompt

```text
You are implementing M13: Complete slow 26B text inference path in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M13. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M13 exit criterion passed. Stop before starting the next milestone.
```
