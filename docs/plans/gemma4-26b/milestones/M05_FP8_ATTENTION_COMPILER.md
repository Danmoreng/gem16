# M05 — Deterministic FP8 attention compiler

## Objective

Implement the compiler stage that converts the selected BF16 master checkpoint's attention Q/K/V/O matrices into gem16's checkpoint-compatible FP8 representation, with deterministic per-output-channel scales and complete error telemetry.

## Why this milestone exists

The production design keeps attention in FP8 so the existing SM120 projection and attention path can be reused. This milestone must prove that the project-built quantizer is reproducible and that its ordinary-BF16 output is comparable with the published Unsloth FP8 attention tensors before QAT-derived results are interpreted.

## Prerequisites

- M04 complete
- Pinned ordinary BF16, QAT BF16, and Unsloth source locks
- Manifest maps every attention module and all omitted V projections

## Repository areas to inspect first

- `src/numeric/fp8.cpp`
- `src/cuda/fp8/reference.cu`
- `src/cuda/fp8/sm120.cu`
- `src/cuda/fp8/cutlass_sm120.cu`
- `src/model/manifest.cpp`
- `docs/CHECKPOINT_FORMAT.md`
- `tests/unit/fp8_test.cpp`

## Suggested additions or boundaries

- `tools/gem16_compile/quantize_fp8.py`
- `tools/gem16_compile/fp8_report.py`
- `tests/python/test_fp8_compiler.py`
- `benchmarks/goldens/gemma4_26b/fp8/`

## Implementation sequence

1. Extract the current runtime FP8 contract into a machine-readable compiler specification: storage E4M3FN, one BF16 scale per output row, dynamic activation scale per token, and the exact dequantization equation.
2. Implement a pure host reference encoder with explicit round-to-nearest-even, saturation, zero-row handling, endian rules, and no dependence on GPU conversion intrinsics.
3. Stream source matrices by row tiles; never materialize all attention tensors in host RAM at once.
4. Emit `.weight` and `.weight_scale` tensors using the same schema consumed by `Fp8Binding`; retain source tensor name, source SHA-256, logical shape, and quantizer parameters in the compiler report.
5. Compile both ordinary BF16 and QAT BF16 sources with the identical code path and configuration.
6. Dequantize the ordinary-BF16 result and compare it tensor-by-tensor with the Unsloth FP8 checkpoint. Treat mismatch as evidence to characterize, not as automatic failure, because scale selection or rounding may differ.
7. Run real-activation operator comparisons for representative local and global Q/K/V/O shapes, including missing global V projection semantics.
8. Make compiler output deterministic across repeated runs on the reference CPU and document any platform-sensitive floating-point operation.

## Required tests

- Exhaustive or table-driven E4M3FN encoding fixtures, including ties, subnormals, saturation, infinities, NaNs, signed zero, and all-zero rows.
- Round-trip test for synthetic matrices whose rows exercise very different dynamic ranges.
- Exact byte identity across two compiler runs from the same source lock.
- C++ runtime loader accepts every emitted FP8 tensor and rejects wrong scale dtype/shape.
- Real-shape CUDA scalar and SM120 kernels consume emitted tensors and satisfy the existing numerical thresholds.
- Ordinary-BF16 compiler-versus-Unsloth report includes cosine, relative L2, maximum absolute error, scale correlation, saturation and operator-output error for every attention matrix.

## Evidence and documentation outputs

- `artifacts/m05/fp8-compiler-config.json`
- `artifacts/m05/fp8-tensor-report.json`
- `artifacts/m05/ordinary-vs-unsloth-fp8.json`
- `artifacts/m05/qat-fp8-summary.json`
- Disassembly evidence for the existing FP8 native path remains linked but is not reclassified as a 26B performance result.

## Suggested commands

```text
python tools/compile_gemma4_26b.py --source-lock models/gemma4-26b-base-bf16.lock.json --stage fp8-attention --output build/models/base-fp8-partial
```
```text
python tools/compile_gemma4_26b.py --source-lock models/gemma4-26b-qat-bf16.lock.json --stage fp8-attention --output build/models/qat-fp8-partial
```
```text
python tools/compare_quantized_checkpoints.py --family attention --left build/models/base-fp8-partial --right "$UNSLOTH_26B" --output artifacts/m05/ordinary-vs-unsloth-fp8.json
```
```text
ctest --test-dir build/host --output-on-failure && ctest --test-dir build/blackwell-release --output-on-failure
```

## Risks to watch in this milestone

- Using CUDA or NumPy casts as the reference encoder may hide rounding differences.
- Global layers can omit a stored V projection; the compiler must not invent one.
- Per-channel scale optimization may differ from Unsloth even when both outputs are valid.
- QAT training targeted Q4_0, so lower weight error under FP8 does not by itself prove end-to-end quality.

## Forbidden shortcuts

- Copying Unsloth FP8 tensors into the QAT-derived artifact.
- Using a non-deterministic calibration batch for a weight-only FP8 encoder.
- Changing activation quantization or projection accumulation order in the runtime during this milestone.
- Treating lower weight MSE as proof of better model quality.

## Exit criteria

- [ ] FP8 encoder output is byte-deterministic and fully specified.
- [ ] Both ordinary and QAT BF16 attention families compile successfully.
- [ ] Runtime schema validation and real-shape operator tests pass.
- [ ] Ordinary compiler-versus-Unsloth differences are quantified for every attention tensor.
- [ ] No 12B loader, operator, generation, or benchmark regression is introduced.

## Downstream milestones unblocked

- M08 derived artifact assembly
- M12 26B attention integration
- M18 converter A/B analysis

## Codex execution prompt

```text
You are implementing M05: Deterministic FP8 attention compiler in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M05. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M05 exit criterion passed. Stop before starting the next milestone.
```
