# M06 — Deterministic NVFP4 compiler for shared and routed experts

## Objective

Implement and qualify a deterministic BF16-to-NVFP4 compiler for the always-active shared MLP and all routed expert gate/up/down weights, using the exact E2M1 plus UE4M3/E4M3 scale contract required by the SM120 native path.

## Why this milestone exists

Expert weights dominate the 26B checkpoint. Their storage and execution format decides whether the model fits and whether Blackwell's native block-scaled Tensor Cores can be used. This compiler is therefore both a quality-critical numerical component and the primary memory enabler.

## Prerequisites

- M04 complete
- Host NVFP4 codec and current 12B runtime contract understood
- Actual serialized expert tensor names and shapes discovered in M03

## Repository areas to inspect first

- `src/numeric/nvfp4.cpp`
- `src/cuda/nvfp4/reference.cu`
- `src/cuda/nvfp4/sm120_layout.cpp`
- `src/cuda/nvfp4/sm120.cu`
- `src/cuda/nvfp4/cutlass_sm120.cu`
- `src/cuda/engine/target_model.cu`
- `tests/unit/nvfp4_test.cpp`
- `tests/unit/sm120_layout_test.cpp`
- `docs/CHECKPOINT_FORMAT.md`

## Suggested additions or boundaries

- `tools/gem16_compile/quantize_nvfp4.py`
- `tools/gem16_compile/nvfp4_report.py`
- `tests/python/test_nvfp4_compiler.py`
- `benchmarks/goldens/gemma4_26b/nvfp4/`

## Implementation sequence

1. Write the canonical mathematical contract in code and documentation: two E2M1 values per byte, one local scale per 16 contracting elements, one tensor-global weight divisor, one tensor-global activation divisor, and FP32 accumulation.
2. Define the scale-selection objective explicitly. Start with an Unsloth/compressed-tensors-compatible recipe if it can be reproduced; otherwise retain the measured project recipe and label it distinctly.
3. Implement a scalar binary64 or carefully controlled FP32 reference encoder with explicit tie-breaking and exact E4M3FN encoding.
4. Support both actual storage arrangements found in source checkpoints: fused expert `gate_up` tensors or separate gate/up tensors. The output artifact may choose one canonical logical schema, but the compiler report must record every reshape/split.
5. Compile the ordinary BF16 source and compare it against Unsloth's published NVFP4 expert tensors after dequantization and at operator output.
6. Compile the QAT BF16 source with exactly the same algorithm and parameters.
7. Stream by layer, expert and row tile. Bound peak host memory and write directly to final Safetensors shards.
8. Generate both source-order payloads for audit and the final on-disk canonical payload only once. Runtime-specific Row8/K64 tiling remains an in-memory load transform unless the M00 policy explicitly changes.
9. Produce per-tensor error, scale, saturation, code-frequency and selected-real-activation output statistics.
10. Validate all divisors are finite and positive, every local scale is valid, and no NaN encoding enters the artifact.

## Required tests

- Exhaustive E2M1 nibble and E4M3FN scale codec tests.
- Rounding-tie, zero-block, subnormal, extreme-range and adversarial outlier fixtures.
- Logical shape and byte-count tests for all expert matrices, including fused gate/up splits.
- Two complete ordinary-BF16 compiler runs produce identical hashes.
- Real-shape CPU oracle, CUDA reference and direct SM120 projection comparisons for shared MLP and at least one expert per layer class.
- Ordinary compiler-versus-Unsloth report covers every routed and shared matrix, not a sample.
- A bounded host-memory test proves the compiler never loads the entire 22.8B-weight expert family at once.

## Evidence and documentation outputs

- `artifacts/m06/nvfp4-compiler-config.json`
- `artifacts/m06/ordinary-vs-unsloth-nvfp4.json`
- `artifacts/m06/qat-nvfp4-summary.json`
- `artifacts/m06/compiler-memory-telemetry.json`
- Per-layer and global byte totals reconciled with the expected approximately 12,846,366,720 routed-expert bytes and 301,086,720 shared-MLP bytes before alignment.

## Suggested commands

```text
python tools/compile_gemma4_26b.py --source-lock models/gemma4-26b-base-bf16.lock.json --stage nvfp4-mlp --output build/models/base-nvfp4-partial
```
```text
python tools/compile_gemma4_26b.py --source-lock models/gemma4-26b-qat-bf16.lock.json --stage nvfp4-mlp --output build/models/qat-nvfp4-partial
```
```text
python tools/compare_quantized_checkpoints.py --family experts --left build/models/base-nvfp4-partial --right "$UNSLOTH_26B" --output artifacts/m06/ordinary-vs-unsloth-nvfp4.json
```
```text
compute-sanitizer --tool memcheck build/blackwell-release/bin/gem16-cuda-tests
```

## Risks to watch in this milestone

- The QAT BF16 distribution may be less compatible with NVFP4 than the ordinary BF16 distribution.
- A fused `gate_up` source can be split along the wrong axis without obvious load errors.
- Global divisor conventions can be reciprocal between toolchains.
- Scale optimization can be too slow for 22.8B values unless streaming and vectorization are designed carefully.
- A compiler that writes the decode-tiled layout to disk can reduce auditability and couple artifacts to one backend.

## Forbidden shortcuts

- Inferring expert tensor axes from size alone.
- Silently clipping invalid scales or replacing NaNs.
- Calibrating ordinary and QAT models with different algorithms.
- Keeping both source-order and tiled expert weights resident on the GPU.
- Using output quality from a few chat prompts as quantizer acceptance.

## Exit criteria

- [ ] NVFP4 compiler is deterministic, documented and bounded-memory.
- [ ] Every shared and routed expert tensor compiles for ordinary and QAT sources.
- [ ] All tensor byte totals, shapes and scale relationships validate.
- [ ] Real-shape runtime operators consume the output correctly.
- [ ] Ordinary compiler-versus-Unsloth differences are fully characterized.
- [ ] Existing 12B NVFP4 tests and full generation gates remain green.

## Downstream milestones unblocked

- M08 artifact assembly
- M10 CPU MoE oracle
- M14 native expert decode
- M18 converter A/B analysis

## Codex execution prompt

```text
You are implementing M06: Deterministic NVFP4 compiler for shared and routed experts in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M06. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M06 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

Add two deliberately distinguishable source-format fixtures:

```text
llm-compressor: fp4 * local_scale / global_divisor
ModelOpt:       fp4 * local_scale * tensor_multiplier
```

Use non-unit values and mutation tests. Produce a ModelOpt-style control artifact or tensor subset for candidate G. Add W4A16 diagnostic execution so M18 can distinguish weight error from activation-quantization error.
