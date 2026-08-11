# Source and evidence links

This package intentionally uses stable repository/model identifiers rather than embedding temporary download URLs.

## Converter research and native compiler architecture

- Binding architecture: [`../specs/NATIVE_CONVERTER_ARCHITECTURE.md`](../specs/NATIVE_CONVERTER_ARCHITECTURE.md)
- Retained local llama.cpp research: [`../../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md`](../../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md)
- Inspected clean local checkout: `third_party/cache/llama.cpp`, commit
  `0b14b87d7c20cb753b94b96854dd7b45306fc696`
- Desired benchmark pin (separate from the inspected checkout):
  `benchmarks/baselines/llama_cpp/commit.txt`, commit
  `153d324bcf86d220b235ca010eeb11213f32b5d1`
- HF/GGUF conversion entrypoint: `third_party/cache/llama.cpp/convert_hf_to_gguf.py`
- Native post-GGUF quantizer: `third_party/cache/llama.cpp/tools/quantize/quantize.cpp`,
  `third_party/cache/llama.cpp/src/llama-quant.cpp`, and
  `third_party/cache/llama.cpp/ggml/src/ggml-quants.c`
- Local mixed FP8/NVFP4 patch: `benchmarks/baselines/llama_cpp/patches/0001-support-mixed-fp8-nvfp4-compressed-tensors.patch`

Agents must treat llama.cpp as a version-scoped engineering reference. Its concepts may be adopted only after exact
Gem16 contract, numerical, provenance and license checks; its ignored cache is not a production dependency.

## gem16

- Repository: `Danmoreng/gem16`
- Analysis anchor: `1c4287965d318ba32a68e597f9d7b6678b883376`
- Key files:
  - `AGENTS.md`
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `docs/CHECKPOINT_FORMAT.md`
  - `docs/MEMORY.md`
  - `docs/CORRECTNESS.md`
  - `docs/BENCHMARKING.md`
  - `docs/DECISIONS.md`
  - `docs/PERFORMANCE_LEDGER.md`
  - `src/cuda/engine/target_model.cu`
  - `src/cuda/engine/inference_engine.cu`

## Model sources to pin in M01

- Google QAT unquantized BF16:
  `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`
- Google official Q4_0 GGUF:
  `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`
- Google ordinary instruction model:
  exact repository must be verified from the current official model family
- Unsloth NVFP4:
  `unsloth/gemma-4-26B-A4B-it-NVFP4`
- NVIDIA comparison:
  `nvidia/Gemma-4-26B-A4B-NVFP4`

Never use these identifiers with `revision=main` in evidence.

## Reference implementations

- Hugging Face Transformers:
  - `src/transformers/models/gemma4/modular_gemma4.py`
  - Mixtral expert implementation used by Gemma 4
- NVIDIA CUTLASS:
  - Blackwell functionality documentation
  - GeForce block-scaled GEMM examples
  - SM120 block-scaled builders
- llama.cpp:
  - `ggml/src/ggml-quants.c` Q4_0 reference
  - current CUDA Q4_0 kernels for optional comparison

Pin exact commits before copying semantics or code.

## Documentation rule

When repository evidence changes after this package date:

1. write a drift report;
2. prefer current primary-source code/config;
3. update the milestone assumption;
4. retain the old source identity in historical evidence.
