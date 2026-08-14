# M05 — Deterministic FP8 attention compiler

## Objective

Implement the compiler stage that converts the selected BF16 master checkpoint's attention Q/K/V/O matrices into gem16's checkpoint-compatible FP8 representation, with deterministic per-output-channel scales and complete error telemetry.

## Why this milestone exists

The production design keeps attention in FP8 so the existing SM120 projection and attention path can be reused. This milestone must prove that the project-built quantizer is reproducible and that its ordinary-BF16 output is comparable with the published Unsloth FP8 attention tensors before QAT-derived results are interpreted.

## Prerequisites

- M04 complete
- Pinned ordinary BF16, QAT BF16, and Unsloth source locks
- Manifest maps every attention module and all omitted V projections

## Current status and binding architecture

M05 is accepted at implementation commit `d91388113d68974f9ab7cec1a90ef768285c0645`. The native C++20 batch
encoder/comparator, bounded host/sanitizer/CUDA tests, exact Ordinary/QAT plans, short throughput probe, one clean
full run per source, structural verification, Ordinary-versus-Unsloth comparison, complete hashes and exact 12B
regression pass. The clean evidence and owner sign-off are retained in
[M05 acceptance](../../../evidence/gemma4_26b/m05-fp8-attention-compiler-acceptance-2026-08-11.md).
The accepted M04 Python scaffold remains the control plane for locks, exact plans, coverage, publication and
provenance. M05's promoted conversion is the native data plane; `quantize_fp8.py` is an oracle/fixture aid and cannot
be a production fallback.
Read the binding [native converter architecture](../specs/NATIVE_CONVERTER_ARCHITECTURE.md) and the version-scoped
[llama.cpp converter research](../../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md) before editing.

## Repository areas to inspect first

- `src/numeric/fp8.cpp`
- `src/cuda/fp8/reference.cu`
- `src/cuda/fp8/sm120.cu`
- `src/cuda/fp8/cutlass_sm120.cu`
- `src/model/manifest.cpp`
- `docs/CHECKPOINT_FORMAT.md`
- `tests/unit/fp8_test.cpp`

## Suggested additions or boundaries

- `src/compiler/fp8_batch_encoder.{h,cpp}` and `src/cli/fp8_compiler_main.cpp`
- `tools/gem16_compile/quantize_fp8.py` as an independent oracle/fixture aid only
- `tools/gem16_compile/fp8_report.py`
- `tests/unit/fp8_batch_encoder_test.cpp`
- `tests/python/test_fp8_compiler.py`
- `benchmarks/goldens/gemma4_26b/fp8/`
- `docs/plans/gemma4-26b/specs/NATIVE_CONVERTER_ARCHITECTURE.md`

The accepted M04 Python scaffold remains responsible for plans, source-lock verification, coverage, publication,
provenance and atomic artifact handling. M05's promoted conversion itself is native C++20; unavailable native
support fails visibly and cannot fall back to Python.

## Implementation sequence

1. Extract the current runtime FP8 contract into a machine-readable compiler specification: storage E4M3FN, one BF16 scale per output row, dynamic activation scale per token, and the exact dequantization equation.
2. Implement the versioned native C++20 batch encoder with explicit round-to-nearest-even, saturation, zero-row
   handling, endian rules, and no dependence on GPU conversion intrinsics. Retain the Python codec only as an
   independent oracle and fixture generator.
3. Stream source matrices by row tiles; never materialize all attention tensors in host RAM at once.
4. Emit `.weight` and `.weight_scale` tensors using the same schema consumed by `Fp8Binding`; retain source tensor name, source SHA-256, logical shape, and quantizer parameters in the compiler report.
5. Compile the complete 115-matrix Ordinary-BF16 attention family once and the complete 115-matrix QAT-BF16
   attention family once with the identical native code path and configuration. Do not run duplicate Python/native
   conversions or a second full M05 artifact solely for reproducibility.
6. Use the shared native data-plane comparison routines to dequantize the ordinary-BF16 result and compare it tensor-by-tensor with the Unsloth FP8 checkpoint. A small Python wrapper may select inputs and serialize evidence, but it must not run a billion-element production comparison loop. Treat mismatch as evidence to characterize, not as automatic failure, because scale selection or rounding may differ.
7. Run real-activation operator comparisons for representative local and global Q/K/V/O shapes, including missing global V projection semantics.
8. Establish determinism with exhaustive native codec tests, byte-golden rows, bounded threads-1-versus-N fixture
   identity and complete output hashes. Standalone M05 verification is structural/hash/source-lock verification and
   does not reconvert; until M08's external lock, disclose `transformation_recomputed=false`.

## Required tests

- Native exhaustive or table-driven E4M3FN encoding fixtures, including ties, subnormals, saturation, infinities,
  NaNs, signed zero and all-zero rows; retain Python oracle cross-checks for the frozen byte fixtures.
- Native row tests whose rows exercise very different dynamic ranges, plus exact scale/weight byte goldens.
- A bounded synthetic batch must be byte-identical with one versus multiple explicit native threads and must emit
  deterministic telemetry; no second full checkpoint conversion is required for this evidence.
- Runtime schema/binding tests accept the emitted FP8 tensor contract and reject wrong scale dtype/shape. Full
  loader integration remains M08.
- Real-shape CUDA scalar and SM120 kernel tests remain downstream validation and must not be claimed from M05's
  non-runtime-loadable partial artifact.
- Ordinary-BF16-versus-Unsloth report includes cosine, relative L2, maximum absolute error, scale correlation and
  saturation for every attention matrix. Large dequantization and metric accumulation use the shared native data
  plane; Python may orchestrate report files and small fixtures. Operator-output comparisons remain a separate CUDA gate.
- One native full Ordinary-BF16 run and one native full QAT-BF16 run, after a short throughput probe and explicit
  owner approval if the projected run is long. Standalone verify must not invoke conversion.

## Evidence and documentation outputs

- `artifacts/m05/fp8-compiler-config.json`
- `artifacts/m05/qat-fp8-summary.json`
- `artifacts/m05/acceptance.json`
- `artifacts/raw-evidence-index.json` for byte sizes and SHA-256 of pruned raw reports
- Disassembly evidence for the existing FP8 native path remains linked but is not reclassified as a 26B performance result.

## Suggested commands

```text
python3 tools/compile_gemma4_26b.py plan --source-lock models/gemma4-26b-base-bf16.lock.json --profile fp8-attention-partial-v1 --head-format deferred ...
```
```text
python3 tools/compile_gemma4_26b.py compile --source-lock models/gemma4-26b-base-bf16.lock.json --profile fp8-attention-partial-v1 --head-format deferred --native-encoder <gem16-fp8-compiler> --threads <N> ...
```
```text
python3 tools/compile_gemma4_26b.py verify --source-lock models/gemma4-26b-base-bf16.lock.json --profile fp8-attention-partial-v1 --head-format deferred ...
```
```text
python3 tools/compare_quantized_checkpoints.py --family attention --compiled <ordinary-fp8-partial> --compiled-source-lock models/gemma4-26b-base-bf16.lock.json --compiled-source <ordinary-source> --compiled-plan benchmarks/goldens/gemma4_26b/fp8/ordinary-compiler-plan.json --unsloth-lock models/gemma4-26b-unsloth-nvfp4.lock.json --unsloth-source <unsloth-source> --native-encoder <gem16-fp8-compiler> --threads <N> --max-host-memory <bytes> --staging-bytes <bytes> --output artifacts/raw/m05/ordinary-vs-unsloth-fp8.json
```
```text
ctest --preset host-debug && ctest --preset blackwell-release
```

Earlier stage-style examples are intentionally retired. The retained short native probe projected a short run, and
the owner authorized both the diagnostic and clean-revision full conversions recorded by M05 evidence.

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

- [x] Versioned native C++20 FP8 encoder output is byte-deterministic and fully specified; Python is oracle-only.
- [x] One clean native full Ordinary-BF16 and one clean native full QAT-BF16 attention family compile successfully.
- [x] Runtime schema/binding validation and bounded native tests pass; runtime loading remains M08 work.
- [x] Ordinary compiler-versus-Unsloth differences are quantified for every attention tensor.
- [x] M05 verify is structural/hash/source-lock-only and reports `transformation_recomputed=false`.
- [x] No 12B loader, operator, generation, or benchmark regression is introduced.

## Downstream milestones unblocked

- M06 native NVFP4 expert compiler and shared converter-data-plane extension

M08 remains blocked on M06 and M07. M12 follows the separate M10–M11 correctness branch, and M18 remains blocked on
M13 plus the complete M05–M08 compiler chain.

## Codex execution prompt

```text
You are implementing M05: Deterministic FP8 attention compiler in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, `../specs/NATIVE_CONVERTER_ARCHITECTURE.md`, the version-scoped llama.cpp research evidence, and this milestone. Work only on M05. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M05 exit criterion passed. Stop before starting the next milestone.
```
