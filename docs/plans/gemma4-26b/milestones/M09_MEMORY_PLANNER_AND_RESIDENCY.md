# M09 — 26B memory planner and text-only residency

## Objective

Generalize memory planning and admission control for the 30-layer 26B MoE variant, enforce text-only residency, and prove that 32K context fits with a safe device margin before full-model CUDA execution is attempted.

## Why this milestone exists

The nominal 13.7 GiB weight target leaves little room for KV, activations, graphs, CUDA context and temporary prefill storage. Correct formulas and hard startup admission are required to prevent optimistic plans from failing late or consuming the desktop GPU.

## Prerequisites

- M08 compiled artifact loads
- M02 model traits available
- M03 tensor inventory complete

## Repository areas to inspect first

- `include/gem16/memory.h`
- `src/runtime/memory_plan.h`
- `src/runtime/memory_plan.cpp`
- `src/runtime/memory.cpp`
- `src/cuda/engine/inference_engine.cu`
- `src/server/session_pool.cpp`
- `docs/MEMORY.md`
- `tests/unit/memory_plan_test.cpp`

## Suggested additions or boundaries

- `docs/GEMMA4_26B_MEMORY.md`
- `tests/unit/gemma4_26b_memory_plan_test.cpp`
- `tools/report_model_memory.py`

## Implementation sequence

1. Make weight and scale planning operate on the selected resident manifest rather than `manifest.total_tensor_bytes` when the artifact can contain nonresident audit metadata.
2. Count 25 sliding and 5 full-attention layers from model traits; never infer the pattern from layer index modulo unless validation has already proved it.
3. Implement separate K and V FP8 payload formulas: local rings capped at 1,024 tokens and global extents sized to the requested context.
4. Model immutable weights, KV, recurrent hidden state, routing buffers, expert permutation, activation scales, CUTLASS workspace, graph-private allocations, output candidates, sampling and safety margin as separate named regions.
5. Add context profiles for 8K, 16K, 32K, 64K and optional 128K. Keep 32K as the required product gate and 64K as the target.
6. Add a 26B prefill chunk-size selector constrained by a maximum workspace budget rather than reusing the 12B 2,048-token chunk blindly.
7. Perform startup admission against current free VRAM, measured allocator deltas and a configurable but nonzero reserve. Fail before model load when an impossible profile is selected.
8. Ensure server slot scaling accounts for immutable shared weights once and mutable per-session KV/workspace separately.
9. Export named byte counters through CLI JSON, `/health` and `/metrics`.
10. Reconcile planned bytes with actual `cudaMemGetInfo` deltas using placeholder allocations before enabling full execution.

## Required tests

- Exact formula tests: local FP8 K+V is 100 MiB after the ring fills; 32K total KV is 420 MiB; 64K is 740 MiB for the validated 26B traits.
- Overflow and malformed-config tests for every multiplication and alignment.
- Head-format variants change only their selected immutable weight region.
- Vision and MTP bytes are zero and requests requiring them are rejected.
- One, two and impossible session-count admission scenarios.
- Plan versus real allocation deltas stay within an explicitly documented allocator overhead tolerance.
- 12B memory profile outputs remain byte-for-byte or semantically compatible with retained fixtures.

## Evidence and documentation outputs

- `docs/GEMMA4_26B_MEMORY.md` with formulas and measured deltas
- `artifacts/m09/memory-plans.json` for all profiles
- `artifacts/m09/allocation-reconciliation.json`
- `artifacts/m09/server-admission.json`

## Suggested commands

```text
build/blackwell-release/bin/gem16-inspect --model "$GEM16_26B" --validate --memory-profile 32768 --json artifacts/m09/plan-32k.json
```
```text
python tools/report_model_memory.py --model "$GEM16_26B" --contexts 8192,16384,32768,65536 --output artifacts/m09/memory-plans.json
```
```text
ctest --test-dir build/host --output-on-failure -R memory
```

## Risks to watch in this milestone

- CUDA context and graph allocation are not represented by simple payload formulas.
- Grouped MoE prefill can accidentally scale workspace with tokens × experts × hidden.
- A 64K context may fit by payload but fail because of CUTLASS or graph workspace.
- Server admission based only on `nvidia-smi` process samples is too coarse.

## Forbidden shortcuts

- Counting the entire remaining VRAM as KV capacity.
- Allowing CPU expert offload when admission fails.
- Allocating all 128-expert intermediate outputs simultaneously.
- Reusing the 12B workspace constants without derivation.
- Advertising 64K before measured execution passes M21.

## Exit criteria

- [ ] Every immutable and mutable region has a checked formula and named accounting.
- [ ] 32K plan fits the reference 16 GB device with the planned safety reserve.
- [ ] 64K feasibility is classified with exact headroom, not assumed.
- [ ] Startup and server admission reject impossible configurations early.
- [ ] No modality or MTP residency appears in the base profile.
- [ ] 12B memory and scheduler tests remain green.

## Downstream milestones unblocked

- M11 CUDA reference allocations
- M15 bounded prefill
- M21 long-context qualification

## Codex execution prompt

```text
You are implementing M09: 26B memory planner and text-only residency in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M09. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M09 exit criterion passed. Stop before starting the next milestone.
```
