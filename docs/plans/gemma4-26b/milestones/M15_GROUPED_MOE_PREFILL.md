# M15 — Grouped and bounded-workspace MoE prefill

## Objective

Implement a prompt-processing MoE path that groups token-expert assignments and uses native NVFP4 GEMMs while keeping workspace bounded enough for 32K and 64K context profiles.

## Why this milestone exists

Prefill must exploit matrix-level Tensor Core throughput, but a naive tokens × top-8 × hidden materialization can consume the entire remaining VRAM. This milestone treats routing, sorting, grouped GEMM and workspace as one design problem.

## Prerequisites

- M18 preliminary quantizer/quality kill gate passed
- M14 decode path or a separately accepted shared native-MoE layout contract
- M09 prefill budget
- M11 reference path

## Repository areas to inspect first

- `src/cuda/nvfp4/cutlass_sm120.cu`
- `src/cuda/nvfp4/cutlass_sm120.h`
- `src/cuda/engine/inference_engine.cu`
- `src/runtime/memory_plan.cpp`
- `CUTLASS grouped/block-scaled GEMM examples`
- `docs/PERFORMANCE_LEDGER.md`

## Suggested additions or boundaries

- `src/cuda/moe/prefill_router.cu`
- `src/cuda/moe/prefill_permute.cu`
- `src/cuda/moe/prefill_sm120.cu`
- `src/cuda/moe/prefill_reduce.cu`
- `tests/cuda/gemma4_26b_moe_prefill_test.cu`

## Implementation sequence

1. Select a conservative initial chunk size from M09 and expose it in runtime metadata.
2. Compute router outputs for the chunk and write exactly eight assignment records per token into fixed storage.
3. Build expert counts, prefix sums and a stable token permutation on device. Define deterministic ordering within each expert.
4. Run grouped W13 directly against resident expert weights. Reuse the decode-final layout if performance is acceptable; otherwise use bounded scratch conversion for one active projection at a time, never a persistent duplicate.
5. Apply activation and run grouped W2. Carry token index, top-k slot and router weight through the compact assignment stream.
6. Inverse-permute and accumulate selected expert contributions in a deterministic, documented order.
7. Execute the shared dense MLP with native batch NVFP4 GEMMs and combine branches at the exact model boundary.
8. Overlap routing/permutation and expert work only after a serial correct path passes.
9. Tune chunk size, expert scheduling and CUTLASS workspace under the hard VRAM budget; record rejected candidates.
10. Ensure the last partial chunk, zero-hit experts, heavily skewed routing and every-expert-hit cases are correct.

## Required tests

- Stable histogram/prefix/permutation round trip for synthetic assignments.
- Zero-hit and all-hit expert cases.
- Skewed routing and uniform routing fixtures.
- Chunk boundary equivalence for 1, 17, 511, 512, 1,023 and selected production chunk sizes.
- Whole-layer output versus M11 for real prompt activations.
- No assignment loss or duplicate contribution.
- Workspace does not scale with full context length and stays within M09.
- Sanitizers and deterministic repeated prefill checks.
- Adjacent 128/512/2K/8K prompt throughput measurements.

## Evidence and documentation outputs

- `artifacts/m15/moe-prefill-correctness.json`
- `artifacts/m15/moe-prefill-workspace.json`
- `artifacts/m15/moe-prefill-benchmark.json`
- `artifacts/m15/moe-prefill-nsight/`
- Performance ledger entries for chunk-size and scheduling decisions.

## Suggested commands

```text
build/blackwell-release/bin/gem16-bench moe-prefill --model "$GEM16_26B" --tokens 128,512,2048,8192 --warmups 3 --repetitions 10 --json artifacts/m15/moe-prefill-benchmark.json
```
```text
python tools/validate_moe_prefill.py --model "$GEM16_26B" --chunks 1,17,511,512,1024 --output artifacts/m15/moe-prefill-correctness.json
```

## Risks to watch in this milestone

- Sort/permutation overhead can dominate small prompts.
- CUTLASS layouts may require scale interleaving or temporary row-major views.
- A few popular experts can create poor load balance.
- Atomic inverse reduction may be nondeterministic or change rounding.
- Large chunks improve GEMM efficiency but can destroy 64K feasibility.

## Forbidden shortcuts

- Workspace proportional to context × 128 experts × intermediate size.
- Host routing or host sorting.
- Persistent duplicate expert layouts.
- Nondeterministic atomic accumulation without a separately accepted quality mode.
- Selecting a chunk size only from short-prompt throughput.

## Exit criteria

- [ ] Grouped prefill matches the CUDA reference within the approved envelope.
- [ ] Routing and permutation are deterministic and entirely GPU-resident.
- [ ] Workspace is bounded and reconciled with M09.
- [ ] 128–8K prompt benchmarks show a material win over the reference path.
- [ ] 32K allocation remains feasible with the required device margin.
- [ ] No 12B prefill regression occurs.

## Downstream milestones unblocked

- M17 optimized integration
- M20 performance
- M21 long context

## Codex execution prompt

```text
You are implementing M15: Grouped and bounded-workspace MoE prefill in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M15. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M15 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

Add the work-queue/M-tile study in `references/imp/IMP_KERNEL_STUDY_PROTOCOL.md`. Include real routing histograms and M={0,1,2,4,8,16,32,64,128,256}. A donor kernel is optional; the milestone succeeds with a better local implementation. Permanent duplicate expert bytes remain zero.
