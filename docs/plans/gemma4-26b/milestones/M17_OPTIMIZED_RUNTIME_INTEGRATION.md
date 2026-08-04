# M17 — Optimized whole-model integration and CUDA Graph decode

## Objective

Integrate the native attention, MoE, embedding/head and fixed arenas into one production 26B execution plan with whole-model batch-one decode graph replay and bounded native prefill.

## Why this milestone exists

Operator speedups are not a product result until the full model has stable ownership, fixed addresses, product semantics and end-to-end correctness.

## Prerequisites

- M12 attention
- M14 native decode
- M15 grouped prefill
- M16 production head
- M09 arenas

## Repository areas to inspect first

- `src/cuda/engine/inference_engine.cu`
- `src/cuda/inference.cu`
- `src/cuda/engine/target_model.cu`
- `include/gem16/engine.h`
- `src/runtime/chat.cpp`
- `src/server/session_pool.cpp`

## Suggested additions or boundaries

- `docs/GEMMA4_26B_RUNTIME.md`
- `tools/validate_gemma4_26b_optimized.py`

## Implementation sequence

1. Define a dedicated immutable 26B execution plan selected once from validated model traits.
2. Bind all kernel pointers and tensor addresses before graph capture; per-layer dispatch may be generated into fixed call sequences rather than virtual calls.
3. Allocate immutable weights once per `ModelRuntime`; allocate KV, routing, prefill and graph-private state per `ExecutionSlot` according to M09.
4. Assemble native prefill with bounded chunks and exact media-free text semantics.
5. Capture the complete ordinary greedy T=1 forward and selection path in one graph or the smallest proven graph sequence.
6. Route sampling through a separate explicitly captured plan without changing greedy memory or timing.
7. Integrate cache reset, suffix prefill, cancellation and resident conversation ownership.
8. Export path metadata: model variant, head format, prefill plan, decode plan, native instruction availability, weight/KV/workspace/graph bytes and fallback count.
9. Run differential validation against M13 after every integration boundary.
10. Remove only superseded 26B reference dispatch from normal execution; retain reference operators for tests.

## Required tests

- Full-model teacher-forced and generation comparison versus M13.
- Whole-model graph capture/replay over changing token, position and context.
- Resident multi-turn suffix prefill.
- Greedy and sampling product semantics.
- Cancellation and reset leave no stale router/KV state.
- Nsight CUDA API trace shows all allocations before generation.
- Weights are shared across sessions; mutable state is isolated.
- All existing 12B target/MTP/multimodal/server tests remain green.

## Evidence and documentation outputs

- `docs/GEMMA4_26B_RUNTIME.md`
- `artifacts/m17/optimized-vs-reference.json`
- `artifacts/m17/allocation-trace.json`
- `artifacts/m17/graph-topology.json`
- `artifacts/m17/runtime-capabilities.txt`

## Suggested commands

```text
python tools/validate_gemma4_26b_optimized.py --model "$GEM16_26B" --reference-path --optimized-path --output artifacts/m17/optimized-vs-reference.json
```
```text
nsys profile --trace=cuda,nvtx -o artifacts/m17/gemma4_26b_graph build/blackwell-release/bin/gem16-run --model "$GEM16_26B" --tokens-file benchmarks/prompts/gemma4-26b-smoke.json --max-generated 256
```

## Risks to watch in this milestone

- Graph capture can hide dynamic allocations in library calls.
- A shared runtime with per-session graph addresses can accidentally alias mutable state.
- Kernel fusion may alter accepted BF16/FP8/NVFP4 boundaries.
- A normal path can silently choose a reference kernel if capability dispatch is incomplete.

## Forbidden shortcuts

- Any allocation, file access or host expert decision in the token loop.
- Silent fallback to correctness kernels in benchmark mode.
- Combining 26B integration with unrelated 12B refactors.
- Removing reference paths before differential tests exist.
- Calling the path quality- or performance-qualified before M19/M20.

## Exit criteria

- [ ] Optimized full-model output stays within the M13 correctness envelope.
- [ ] Whole-model decode graph replays with fixed addresses.
- [ ] No token-loop allocations or silent fallbacks are observed.
- [ ] Weight sharing and session isolation are correct.
- [ ] All path and memory metadata are exported.
- [ ] 12B regression suites pass.

## Downstream milestones unblocked

- M18 converter A/B
- M19 quality
- M20 performance
- M21 long context

## Codex execution prompt

```text
You are implementing M17: Optimized whole-model integration and CUDA Graph decode in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M17. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M17 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

Add canonical actual-path telemetry, a retained first graph-demotion reason and the CUDA lifecycle tests in `specs/CUDA_STATE_LIFECYCLE_SPEC.md`. A predicted path is not sufficient evidence. Destroy/recreate the complete engine repeatedly in one process before exit.
