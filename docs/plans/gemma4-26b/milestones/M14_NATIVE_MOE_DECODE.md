# M14 — Native SM120 batch-one MoE decode

## Objective

Replace the correctness MoE path for T=1 decode with a fully GPU-resident, native SM120 NVFP4 implementation covering router, top-8 routed experts, the always-active shared MLP and weighted reduction.

## Why this milestone exists

Decode latency is the primary product objective. At batch one, the kernel must minimize expert-weight traffic, launch count and intermediate writes while preserving the accepted routing and arithmetic contract.

## Prerequisites

- M11 CUDA reference accepted
- M13 complete full-model reference
- M18 preliminary quantizer/quality kill gate passed
- M06 native-compatible NVFP4 artifact

## Repository areas to inspect first

- `src/cuda/nvfp4/sm120.cu`
- `src/cuda/nvfp4/gemv.cu`
- `src/cuda/nvfp4/mlp.cu`
- `src/cuda/engine/inference_engine.cu`
- `src/cuda/mtp/verify.cu`
- `docs/PERFORMANCE_LEDGER.md`

## Suggested additions or boundaries

- `src/cuda/moe/router_sm120.cu`
- `src/cuda/moe/decode_sm120.cu`
- `src/cuda/moe/reduction_sm120.cu`
- `src/cuda/moe/moe.h`
- `tests/cuda/gemma4_26b_moe_decode_test.cu`

## Implementation sequence

1. Profile the M11 path and model expected bytes, launches and Tensor Core work before selecting geometry.
2. Implement a deterministic router kernel that computes the accepted FP32 probabilities and top-8 entirely on device; keep IDs and weights in fixed control storage.
3. Design an expert-major resident layout compatible with direct Row8/K64 access. Do not gather or copy expert weights per token.
4. Implement a native W13 path for the eight selected experts. Reuse the quantized activation fragment where profitable and keep gate/up logical ordering exact.
5. Apply the model activation and quantize expert intermediates for W2 without materializing eight full FP32 matrices.
6. Implement native W2 projections and fuse router-weight application into the final reduction when this preserves the accepted accumulation order.
7. Implement the always-active shared MLP through the same native NVFP4 family, with its distinct intermediate dimension.
8. Minimize launches experimentally, but retain a readable staged candidate until every fusion proves correctness and speed independently.
9. Capture a whole-MoE decode segment in the model CUDA Graph only after all fixed addresses and dynamic controls are stable.
10. Qualify ordinary one-token decode first. MTP T=3/T=5 reuse is outside this milestone unless it falls out without changing scope.

## Required tests

- Exact top-8 IDs and bounded weight differences versus M11.
- Per-expert W13, activation, W2 and final contribution comparisons.
- Shared MLP and combined feed-forward output comparisons.
- Adversarial router distributions and repeated-expert selection fixtures.
- Whole-layer and whole-model deterministic token checks.
- Graph capture/replay with changing token and position controls.
- Disassembly contains the required native block-scaled NVFP4 instruction family.
- No local-memory spill, out-of-bounds access, race, fallback or token-loop allocation.
- Adjacent performance A/B against M11 and a direct-source SIMT control.

## Evidence and documentation outputs

- `artifacts/m14/moe-decode-correctness.json`
- `artifacts/m14/moe-decode-nsight/`
- `artifacts/m14/moe-decode-disassembly.txt`
- `artifacts/m14/moe-decode-benchmark.json`
- Performance ledger entry for every retained or rejected fusion.

## Suggested commands

```text
build/blackwell-release/bin/gem16-bench moe-decode --model "$GEM16_26B" --warmups 3 --repetitions 10 --json artifacts/m14/moe-decode-benchmark.json
```
```text
ncu --set full --kernel-name regex:Gemma4.*Moe.* build/blackwell-release/bin/gem16-bench moe-decode --model "$GEM16_26B" --repetitions 1
```
```text
cuobjdump --dump-sass build/blackwell-release/bin/gem16-bench > artifacts/m14/moe-decode-disassembly.txt
```

## Risks to watch in this milestone

- Eight active experts make naive launch-per-expert scheduling expensive.
- T=1 is often bandwidth and scheduling bound, so theoretical FP4 FLOPs may not translate directly.
- Aggressive reduction fusion can change accumulation order and token selection.
- Expert-major tiling that helps decode may require temporary prefill conversion unless M15 handles it carefully.
- Router softmax/top-k can become a surprising fixed overhead after expert kernels improve.

## Forbidden shortcuts

- Host-side expert selection.
- Per-token expert-weight gather or device copy.
- A persistent second prefill weight layout.
- Promoting a fusion without adjacent correctness and end-to-end benchmarks.
- Reporting proposed or intermediate expert work as output throughput.

## Exit criteria

- [ ] Native T=1 MoE matches the M11/M13 correctness envelope.
- [ ] Router and all expert execution remain GPU-resident.
- [ ] Native NVFP4 instructions are proven in the selected path.
- [ ] The selected path is materially faster than the correctness reference.
- [ ] No token-loop allocations, spills or silent fallbacks occur.
- [ ] Whole-model deterministic decode remains green.

## Downstream milestones unblocked

- M17 whole-model optimized integration
- M20 performance qualification

## Codex execution prompt

```text
You are implementing M14: Native SM120 batch-one MoE decode in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M14. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M14 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

Study imp's native NVFP4 and small-M interfaces, but preserve a Gemma-specific T=1 schedule and gem16 ownership model. A copied or derived kernel requires the adoption checklist. Benchmark clean-room and donor-inspired arms on the RTX 5080. Do not add pointer registries, host expert dispatch or a second permanent expert layout.
