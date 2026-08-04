# M21 — 32K required and 64K target context qualification

## Objective

Qualify the final 26B model at 32K context and, if memory and quality allow, at 64K using FP8 KV, local rings, global extents and bounded prefill workspace.

## Why this milestone exists

Payload arithmetic suggests 32K and possibly 64K fit, but only real prompt processing, cache wrap, retrieval quality, decode and peak memory prove a usable context profile.

## Prerequisites

- M19 final quality artifact
- M20 qualified optimized runtime
- M09/M12 memory and cache contracts

## Repository areas to inspect first

- `docs/MEMORY.md`
- `docs/BENCHMARKING.md`
- `tools/benchmark_long_context_qa.py`
- `src/cuda/attention/`
- `src/cuda/engine/inference_engine.cu`

## Suggested additions or boundaries

- `tools/validate_gemma4_26b_long_context.py`
- `benchmarks/baselines/gemma4_26b_long_context/`
- `docs/GEMMA4_26B_LONG_CONTEXT.md`

## Implementation sequence

1. Create immutable exact-token prompts at 8K, 16K, 32K and 64K with early/middle/late retrieval markers and natural text.
2. Run cache correctness probes across repeated local ring wrap and global extension.
3. Measure prompt throughput, TTFT, steady decode, retrieval quality, KV bytes, workspace bytes, graph bytes and sampled process peak.
4. Use 32K as a hard product gate with at least 700 MiB free-device margin under the defined reference conditions.
5. Attempt 64K only after 32K passes. Require at least 500 MiB margin or record it as experimental/non-default.
6. Test resident continuation after the long root so suffix prefill and cache identity are exercised.
7. Validate cancellation/reset after long-context allocation.
8. Run a 1,024-token decode soak at 32K and a bounded soak at 64K.
9. Compare FP8 KV quality against a BF16 reference on shorter overlapping contexts and task retrieval at long contexts.
10. Freeze default context profile and admission reserve based on measured data.

## Required tests

- Local ring indices and chronological reads across multiple wraps.
- Global K/V append/read at chunk boundaries.
- All retrieval markers returned in correct order.
- No OOM, fallback, allocation or cache corruption.
- Resident continuation consumes only suffix tokens.
- Repeated deterministic runs have stable output hashes where expected.
- Measured KV payload equals formula and process peak meets margin.
- 32K and 64K reports distinguish quality, speed and memory.

## Evidence and documentation outputs

- `docs/GEMMA4_26B_LONG_CONTEXT.md`
- `artifacts/m21/context-32k.json`
- `artifacts/m21/context-64k.json` if attempted
- `artifacts/m21/soak-32k.json`
- Continuous telemetry and exact prompt manifests.

## Suggested commands

```text
python tools/validate_gemma4_26b_long_context.py --model "$GEM16_26B_FINAL" --contexts 8192,16384,32768,65536 --output artifacts/m21
```
```text
python tools/benchmark_long_context_qa.py --model "$GEM16_26B_FINAL" --prompt-tokens 32768 --output-tokens 1024 --output artifacts/m21/soak-32k.json
```

## Risks to watch in this milestone

- 64K can fit by weights+KV but fail from prefill workspace or CUDA context variation.
- Synthetic repeated text may overstate or understate real retrieval quality.
- Long prompt thermal behavior can change throughput and memory telemetry.
- A large default context can reduce available server slots.

## Forbidden shortcuts

- Calling allocation-only success context support.
- Reducing prompt length after tokenization without disclosure.
- Ignoring retrieval failures because generation completes.
- Advertising 64K as default if reserve is below the frozen threshold.
- Using CPU offload to complete a long-context run.

## Exit criteria

- [ ] 32K completes correctness, retrieval, soak, performance and memory gates.
- [ ] 32K retains at least the frozen device margin.
- [ ] 64K is either qualified with its own margin or clearly classified experimental/rejected.
- [ ] Default context and server admission policy are frozen.
- [ ] No cache, allocation or 12B long-context regression exists.

## Downstream milestones unblocked

- M22 product defaults
- M23 release

## Codex execution prompt

```text
You are implementing M21: 32K required and 64K target context qualification in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M21. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M21 exit criterion passed. Stop before starting the next milestone.
```
