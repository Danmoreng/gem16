# M06 — Deterministic NVFP4 compiler for shared and routed experts

## Objective

Implement and qualify a deterministic BF16-to-NVFP4 compiler for the always-active shared MLP and all routed expert gate/up/down weights, using the exact E2M1 plus UE4M3/E4M3 scale contract required by the SM120 native path.

## Why this milestone exists

Expert weights dominate the 26B checkpoint. Their storage and execution format decides whether the model fits and whether Blackwell's native block-scaled Tensor Cores can be used. This compiler is therefore both a quality-critical numerical component and the primary memory enabler.

## Prerequisites

- M04 complete
- Host NVFP4 codec and current 12B runtime contract understood
- Actual serialized expert tensor names and shapes discovered in M03
- M05 native converter architecture accepted and the shared native data-plane boundary available

## Current status and binding architecture

M06 is dependency-gated and has not started. It must extend the shared native C++ converter data plane first
implemented by M05; do not create an independent Python numerical converter. Python may generate exact plans, small
oracle fixtures and report wrappers, but promoted BF16-to-NVFP4 arithmetic, large comparisons and telemetry belong to
the native tool. Read the binding [native converter architecture](../specs/NATIVE_CONVERTER_ARCHITECTURE.md) and the
version-scoped [llama.cpp converter research](../../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md)
before beginning M06.

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

- `src/compiler/` shared native converter support from M05
- `src/compiler/nvfp4_batch_encoder.{h,cpp}`
- `src/cli/` shared checkpoint-compiler entry point
- `tools/gem16_compile/` plan/report orchestration and small oracle fixtures only
- `tests/unit/nvfp4_batch_encoder_test.cpp`
- `benchmarks/goldens/gemma4_26b/nvfp4/`
- `docs/plans/gemma4-26b/specs/NATIVE_CONVERTER_ARCHITECTURE.md`

## Implementation sequence

1. Write the canonical mathematical contract in code and documentation: two E2M1 values per byte, one local scale per 16 contracting elements, one tensor-global weight divisor, one tensor-global activation divisor, and FP32 accumulation.
2. Define the scale-selection objective explicitly. Start with an Unsloth/compressed-tensors-compatible recipe if it can be reproduced; otherwise retain the measured project recipe and label it distinctly. The promoted implementation is a versioned native C++ extension; a Python reference is only a small oracle.
3. Implement the native scalar/reference codec and optimized bounded batch encoder with explicit tie-breaking and exact E4M3FN/UE4M3 encoding. Deterministic worker partitioning must produce identical bytes and telemetry for the supported thread counts.
4. Support both actual storage arrangements found in source checkpoints: fused expert `gate_up` tensors or separate gate/up tensors. The output artifact may choose one canonical logical schema, but the compiler report must record every reshape/split.
5. Compile the ordinary BF16 source once and compare it against Unsloth's published NVFP4 expert tensors after native dequantization and at operator output.
6. Compile the QAT BF16 source once with exactly the same native algorithm and parameters. Under the accepted partial-stage policy, do not perform duplicate Python/native conversions or a second full partial run solely to claim reproducibility; use exhaustive codec tests, bounded thread-identity fixtures, complete output hashes and small independent oracles instead.
7. Stream by layer, expert and row tile. Bound peak host memory and write directly to final Safetensors shards.
8. Generate both source-order payloads for audit and the final on-disk canonical payload only once. Runtime-specific Row8/K64 tiling remains an in-memory load transform unless the M00 policy explicitly changes.
9. Produce per-tensor error, scale, saturation, code-frequency and selected-real-activation output statistics.
10. Validate all divisors are finite and positive, every local scale is valid, and no NaN encoding enters the artifact.

## Required tests

- Exhaustive E2M1 nibble and E4M3FN scale codec tests.
- Rounding-tie, zero-block, subnormal, extreme-range and adversarial outlier fixtures.
- Logical shape and byte-count tests for all expert matrices, including fused gate/up splits.
- One complete native ordinary-BF16 run and one complete native QAT-BF16 run succeed; native bounded thread-identity fixtures, complete output hashes and small independent oracles establish determinism without duplicate full partial conversions.
- Real-shape CPU oracle, CUDA reference and direct SM120 projection comparisons for shared MLP and at least one expert per layer class.
- Ordinary compiler-versus-Unsloth report covers every routed and shared matrix, not a sample; large dequantization and metric accumulation use native data-plane routines, with Python limited to orchestration and small fixtures.
- A bounded host-memory test proves the compiler never loads the entire 22.8B-weight expert family at once.

## Evidence and documentation outputs

- `artifacts/m06/nvfp4-compiler-config.json`
- `artifacts/m06/ordinary-vs-unsloth-nvfp4.json`
- `artifacts/m06/qat-nvfp4-summary.json`
- `artifacts/m06/compiler-memory-telemetry.json`
- Per-layer and global byte totals reconciled with the expected approximately 12,846,366,720 routed-expert bytes and 301,086,720 shared-MLP bytes before alignment.

## Suggested commands

```text
python3 tools/compile_gemma4_26b.py plan --source-lock models/gemma4-26b-base-bf16.lock.json --profile nvfp4-expert-partial-v1 --head-format deferred --compiler-manifest benchmarks/goldens/gemma4_26b/nvfp4/ordinary-compiler-plan.json --max-host-memory <bytes> --staging-bytes <bytes>
```
```text
python3 tools/compile_gemma4_26b.py compile --source-lock models/gemma4-26b-qat-bf16.lock.json --profile nvfp4-expert-partial-v1 --head-format deferred --compiler-manifest benchmarks/goldens/gemma4_26b/nvfp4/qat-compiler-plan.json --native-encoder <gem16-checkpoint-compiler> --threads <N> --output build/models/qat-nvfp4-partial --max-host-memory <bytes> --staging-bytes <bytes>
```
```text
python3 tools/compare_quantized_checkpoints.py --family experts --compiled build/models/base-nvfp4-partial --reference "$UNSLOTH_26B" --output artifacts/m06/ordinary-vs-unsloth-nvfp4.json
```

These are planned action-first interfaces, not runnable M06 commands today. Earlier stage-style examples are retired.
```text
compute-sanitizer --tool memcheck build/Linux/blackwell-release/bin/gem16-cuda-tests
```

## llama.cpp lessons and explicit non-contracts

llama.cpp's group-16 E2M1/UE4M3 concepts, native C reference codec and bounded threaded scheduler are useful
engineering references. They do not define Gem16's format. In particular, do not adopt its 64-element GGUF
superblock layout, which aggregates four 16-element groups as four scale bytes followed by four contiguous 8-byte
packed subblocks; do not adopt its E4M3/2 plus doubled-LUT convention, `amax/6` and tie policy, selective
`--tensor-type nvfp4` workflow, or incomplete application of input-global scales. Any code reuse requires a pinned
source, MIT license/provenance record, differential byte tests and an explicit decision; the Gem16 scale, layout and
tie contracts always win.

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

- M07 tied embedding/output-head format experiment

M08 remains blocked on M07. M10 is independently available from M03, while M14 and M18 remain blocked on the
M13 correctness path and the M18 quality gate defined by the dependency graph.

## Codex execution prompt

```text
You are implementing M06: Deterministic NVFP4 compiler for shared and routed experts in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, `../specs/NATIVE_CONVERTER_ARCHITECTURE.md`, the version-scoped llama.cpp research evidence, and this milestone. Work only on M06. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

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
