# M10 — CPU oracle for Gemma 4 26B MoE semantics

## Objective

Implement an independent, slow CPU reference for one complete 26B decoder layer, including shared dense MLP, router, top-8 expert selection, routed expert accumulation, normalization and residual ordering.

## Why this milestone exists

MoE introduces discontinuous routing and a second feed-forward branch that the current 12B dense layer does not have. A transparent CPU oracle is the authority needed to debug the CUDA reference and native paths.

## Prerequisites

- M02 model traits
- M03 exact tensor names/shapes
- M06 NVFP4 codec

## Repository areas to inspect first

- `src/numeric/layer.cpp`
- `src/numeric/nvfp4.cpp`
- `tests/unit/layer_test.cpp`
- `src/cuda/layer/reference.h`
- `reference Transformers Gemma4TextRouter/Experts/DecoderLayer`

## Suggested additions or boundaries

- `src/numeric/gemma4_26b_moe.cpp`
- `src/numeric/gemma4_26b_moe.h`
- `tests/unit/gemma4_26b_moe_test.cpp`
- `tools/capture_gemma4_26b_goldens.py`

## Implementation sequence

1. Transcribe the validated reference operation order into a standalone specification before coding.
2. Implement scale-free router RMSNorm, learned hidden-scale vector, `hidden_size^-0.5` factor, BF16/F32 projection semantics, FP32 softmax, deterministic top-8, selected-probability renormalization and per-expert scale.
3. Define and test tie behavior explicitly. Match the pinned trusted runtime where possible; otherwise keep a stable lowest-index rule and report tie inputs.
4. Implement shared dense MLP with the model's activation and all required pre/post norms.
5. Implement routed experts from dequantized NVFP4 weights. Support fused gate/up source representation without changing logical math.
6. Accumulate eight expert contributions in a documented order and precision. Provide an alternate high-precision accumulation mode for diagnostics.
7. Combine shared and routed branches in the exact reference order, then final feed-forward norm, residual addition and layer scalar.
8. Expose captures for router logits/probabilities, top-8 IDs, normalized weights, each selected expert contribution, combined branch and final residual.
9. Validate against pinned Transformers BF16 captures on several real tokens and layers, including first, middle and final layers.
10. Add an optional quantized-weight mode so the same oracle validates ordinary and QAT-derived artifacts.

## Required tests

- Synthetic router cases with unique ordering, exact ties, very large logits and all-equal logits.
- Softmax and top-8 normalization sum and per-expert scaling tests.
- Expert gate/up/down shape and axis tests.
- Shared-only, expert-only and combined branch fixtures.
- Real checkpoint captures for at least layers 0, 5, 15 and 29.
- Ordinary BF16, QAT BF16 and dequantized NVFP4 modes report expected drift independently.
- No use of CUDA, PyTorch or the production runtime inside the C++ oracle.

## Evidence and documentation outputs

- `artifacts/m10/cpu-oracle-goldens.json`
- `artifacts/m10/router-tie-policy.json`
- `artifacts/m10/layer-comparison.json`
- Updated `docs/CORRECTNESS.md` section for 26B MoE semantics.

## Suggested commands

```text
python tools/capture_gemma4_26b_goldens.py --model-lock models/gemma4-26b-qat-bf16.lock.json --layers 0,5,15,29 --output benchmarks/goldens/gemma4_26b/qat-bf16
```
```text
build/host/bin/gem16-unit-tests --filter gemma4_26b_moe
```

## Risks to watch in this milestone

- Reference framework top-k tie behavior can depend on backend implementation.
- BF16 rounding boundaries may occur before or after norms/projections.
- Expert accumulation order can affect final logits even when per-expert outputs match.
- The shared MLP is always active and must not be confused with a 'shared expert' selected by the router.

## Forbidden shortcuts

- Using the CUDA implementation as the CPU oracle.
- Skipping the shared dense branch.
- Comparing only final output while hiding router or per-expert drift.
- Using unordered containers to determine expert accumulation order.
- Hard-coding tensor names without the M03 manifest mapping.

## Exit criteria

- [ ] CPU oracle reproduces trusted BF16 layer captures within approved tolerances.
- [ ] Router IDs and weights are independently validated.
- [ ] Every selected expert contribution and the shared branch can be inspected.
- [ ] Quantized artifact tensors can be consumed through dequantization.
- [ ] Operation order is documented and locked for CUDA implementation.

## Downstream milestones unblocked

- M11 CUDA MoE reference
- M13 full-model correctness
- M18/M19 drift analysis

## Codex execution prompt

```text
You are implementing M10: CPU oracle for Gemma 4 26B MoE semantics in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M10. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M10 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

Freeze multi-layer router and branch goldens based on `references/imp/IMP_SEMANTIC_GOLDENS.md`. Include late-layer and near-tie cases, FP32 router input/logits, per-expert scale ordering, separate shared/expert norms and FP32 expert reduction. Compare official HF, local CPU and pinned imp; resolve discrepancies before exit.
