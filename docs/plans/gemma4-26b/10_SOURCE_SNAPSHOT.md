# Source snapshot used by this plan

## Date

```text
2026-08-04
```

All mutable upstream sources must be re-resolved to immutable commits during M01.

## gem16

```text
repository: https://github.com/Danmoreng/gem16
commit: 1c4287965d318ba32a68e597f9d7b6678b883376
```

Relevant inspected paths include:

- `AGENTS.md`
- `README.md`
- `CMakeLists.txt`
- `docs/ARCHITECTURE.md`
- `docs/BENCHMARKING.md`
- `docs/CHECKPOINT_FORMAT.md`
- `docs/CORRECTNESS.md`
- `docs/DECISIONS.md`
- `docs/MEMORY.md`
- `docs/ROADMAP.md`
- `include/gem16/types.h`
- `src/model/config.*`
- `src/model/manifest.*`
- `src/model/checkpoint_loader.cpp`
- `src/runtime/memory_plan.*`
- `src/cuda/engine/target_model.*`
- `src/cuda/engine/inference_engine.*`
- `src/cuda/nvfp4/*`
- `src/cuda/output_head.*`
- `tools/fetch_model.py`
- `tests/unit/config_test.cpp`

## Google QAT BF16 source

```text
https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
```

The model card describes this as half-precision weights extracted from the QAT pipeline and intended for custom downstream compilation and research.

## Google ordinary BF16 control

```text
https://huggingface.co/google/gemma-4-26B-A4B-it
```

This is the non-QAT control source to be compiled through the exact same project quantizer.

## Official Google Q4_0 reference

```text
https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf
```

Snapshot observation:

- text GGUF: approximately 14.4 GB;
- separate multimodal projection GGUF: approximately 1.19 GB;
- repository total shown as approximately 15.6 GB.

M01 must pin the exact validated revision and file hashes.

## Unsloth NVFP4 baseline

```text
https://huggingface.co/unsloth/gemma-4-26B-A4B-it-NVFP4
```

Snapshot observation:

- repository tree reports a 16.9 GB `model.safetensors`;
- mixed quantization config:
  - FP8 attention;
  - NVFP4 routed experts and language-model MLPs;
  - group size 16 and E4M3 scale dtype;
  - router, embedding/head and vision ignored by those groups;
- model dimensions:
  - hidden 2816;
  - shared intermediate 2112;
  - routed intermediate 704;
  - 30 layers;
  - 128 experts, top-8;
  - 25 local and five global layers;
  - two global KV heads.

Do not use the short tree commit as a final lock. Resolve the full commit SHA.

## Hugging Face reference implementation

```text
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modular_gemma4.py
```

Relevant semantics at the snapshot:

- always-active dense MLP and routed expert path run in parallel from the post-attention residual;
- router uses scale-free RMSNorm, learned hidden scale and `hidden_size**-0.5`;
- projection to 128 scores;
- FP32 softmax;
- top-8;
- selected weights renormalized;
- learned per-expert scales applied;
- experts use fused gate/up storage and separate down storage in the reference module.

M01 must pin a Transformers revision used for goldens.

## Native Blackwell FP4 reference

```text
https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md
```

The snapshot documents native block-scaled narrow-precision Tensor Core paths, `nv_float4_t` with UE4M3 scales and a dense scale vector size of 16, plus supported SM120 tile shapes.

## Q4_0 reference

```text
https://github.com/ggml-org/llama.cpp
```

The reference quantizer uses 32-value blocks with one FP16 scale and 16 packed data bytes. Pin a llama.cpp commit before using it as an oracle.

## Source-use rule

This file is orientation, not a lock. M01 must generate machine-readable locks and local source snapshots. Never run a benchmark or compiler against `main`.

## imp external implementation reference

```text
repository: https://github.com/kekzl/imp
commit: a392904d4216388828d0d56317de046f4ca49627
license: MIT
```

Selected pinned paths and their intended use are defined in [`references/imp/IMP_SOURCE_MAP.md`](references/imp/IMP_SOURCE_MAP.md). Important evidence includes producer-specific NVFP4 scale semantics, Gemma 4 router/residual handling, grouped small-M kernels, actual-path dispatch recording, graph-demotion reporting, machine-readable performance baselines and quality-attribution reports.

Imp is a reference and optional narrowly scoped code donor. It is not a mutable dependency and its published RTX 5090 results are not RTX 5080 release baselines.
