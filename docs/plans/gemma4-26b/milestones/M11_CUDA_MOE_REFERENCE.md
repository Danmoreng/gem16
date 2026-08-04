# M11 — CUDA correctness-first MoE path

## Objective

Implement a transparent GPU-resident reference path for router, shared MLP and routed experts that matches the M10 CPU oracle before introducing performance-oriented fusion.

## Why this milestone exists

Debugging a fused native MoE kernel without an independent GPU path would conflate routing, quantization, expert math and reduction errors. This milestone creates the device-side correctness bridge.

## Prerequisites

- M08 artifact loads
- M09 arenas available
- M10 CPU oracle accepted

## Repository areas to inspect first

- `src/cuda/nvfp4/reference.cu`
- `src/cuda/nvfp4/mlp.cu`
- `src/cuda/layer/checkpoint_probe.cu`
- `src/cuda/engine/inference_engine.cu`
- `tests/cuda/nvfp4_reference_test.cu`
- `CMakeLists.txt`

## Suggested additions or boundaries

- `src/cuda/moe/reference.h`
- `src/cuda/moe/reference.cu`
- `src/cuda/moe/router_reference.cu`
- `tests/cuda/gemma4_26b_moe_reference_test.cu`

## Implementation sequence

1. Allocate fixed reference buffers through the M09 arena; do not introduce per-forward allocations.
2. Implement router normalization/projection/FP32 softmax/top-8/renormalization/per-expert scale as separate observable kernels.
3. Use stable deterministic selection with the M10 tie policy.
4. Invoke existing correctness-oriented NVFP4 projections for the selected experts and the shared dense MLP.
5. Store each selected expert contribution in a bounded eight-row or eight-slot diagnostic region, then reduce in the locked order.
6. Implement all feed-forward norms, branch addition, residual and layer scalar with explicit BF16/FP32 boundaries.
7. Add host capture APIs only for tests and diagnostics; production generation must not copy router decisions to the CPU.
8. Compare every intermediate buffer with the CPU oracle on synthetic and real checkpoint fixtures.
9. Run compute-sanitizer memcheck, racecheck and initcheck on the dedicated test path.
10. Keep this path selectable only as an explicit correctness mode and clearly label it unqualified for performance.

## Required tests

- Bitwise or tightly bounded router score, top-8 ID and weight comparisons.
- Per-expert and shared-branch output comparisons.
- Layer final output cosine, RMS and max-error gates.
- Repeated runs produce identical IDs and output bytes under deterministic mode.
- No host synchronization between router and expert execution except diagnostic capture.
- No allocation after engine initialization.
- Sanitizer suites complete with zero errors/hazards.

## Evidence and documentation outputs

- `artifacts/m11/cuda-reference-comparison.json`
- `artifacts/m11/sanitizer/` logs
- `artifacts/m11/allocation-trace.json`
- Updated capability report naming the 26B MoE reference path separately from native kernels.

## Suggested commands

```text
ctest --test-dir build/blackwell-debug --output-on-failure -R gemma4_26b_moe
```
```text
compute-sanitizer --tool memcheck build/blackwell-debug/bin/gem16-cuda-tests --filter gemma4_26b_moe
```
```text
compute-sanitizer --tool racecheck build/blackwell-debug/bin/gem16-cuda-tests --filter gemma4_26b_moe
```

## Risks to watch in this milestone

- Materializing eight full hidden-size outputs is acceptable for a bounded reference but must not spread into production prefill.
- Softmax reduction order can differ from the framework even when selected experts are stable.
- Incorrect tensor axis binding may produce plausible but wrong expert outputs.
- Diagnostic copies can accidentally enter benchmark timing.

## Forbidden shortcuts

- Calling CPU top-k from the GPU forward path.
- Promoting the reference path based on correctness alone.
- Adding dynamic parallelism or runtime compilation.
- Hiding a BF16 fallback behind an NVFP4 capability label.
- Changing the 12B MLP path during this milestone.

## Exit criteria

- [ ] CUDA reference matches M10 at every named boundary.
- [ ] Routing stays entirely on device during normal execution.
- [ ] Sanitizers and allocation trace pass.
- [ ] The path is explicitly selectable and labeled correctness-only.
- [ ] No 12B CUDA regression occurs.

## Downstream milestones unblocked

- M13 full-model path
- M14 native decode
- M15 grouped prefill

## Codex execution prompt

```text
You are implementing M11: CUDA correctness-first MoE path in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M11. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M11 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

The CUDA reference must expose and test the precision boundaries that imp found failure-prone: FP32 router logits, FP32 selected-weight normalization, per-expert scaling after renormalization and FP32 weighted expert reduction/residual diagnostics. Add an intentional BF16-router mutation that changes at least one near-tie fixture.
