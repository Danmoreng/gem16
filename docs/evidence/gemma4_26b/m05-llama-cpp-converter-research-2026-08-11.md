# M05 llama.cpp converter research

Date: 2026-08-11
Status: retained read-only research; no benchmark, conversion, build, or source modification was performed by this study.
Scope: local llama.cpp conversion architecture and its relevance to the Gemma 4 26B compiler.

## Source identity

The clean checkout inspected for this report is:

```text
third_party/cache/llama.cpp
commit 0b14b87d7c20cb753b94b96854dd7b45306fc696
```

This is the local b10240-era checkout. The repository's desired current benchmark pin is recorded separately as
`benchmarks/baselines/llama_cpp/commit.txt`, naming commit
`153d324bcf86d220b235ca010eeb11213f32b5d1`. Findings in this report are therefore version-scoped to the inspected
commit and must not be described as an audit of the desired/current benchmark pin until that checkout is separately
inspected.

The local patched worktrees `third_party/cache/llama.cpp-mixed` and
`third_party/cache/llama.cpp-mixed-000547513` were also inspected for the Gemma mixed-checkpoint path. The mixed
patch is a working-tree patch, not an upstream commit and not a gem16 dependency.

## Method and concrete paths

The study read source and retained scripts only. It did not run `convert_hf_to_gguf.py`, `llama-quantize`, a model
benchmark, or a conversion job. Important inspected paths include:

- `convert_hf_to_gguf.py`: CLI output types, model loading and FP8 handling;
- `conversion/base.py`: lazy tensor loading, dtype conversion, compressed-tensors handling and output policy;
- `conversion/gemma.py`: Gemma 4 mapping and NVFP4 expert-scale folding;
- `gguf-py/gguf/utility.py`, `lazy.py`, `quants.py`, and `gguf_writer.py`: Safetensors access, lazy evaluation,
  Python/NumPy codecs and GGUF publication;
- `tools/quantize/main.cpp`, `tools/quantize/quantize.cpp`, `tools/quantize/CMakeLists.txt`;
- `src/llama-quant.cpp`, `src/llama-model-loader.cpp`, `src/llama-mmap.cpp`;
- `ggml/src/ggml-quants.c`, `ggml/src/ggml.c`, `ggml/src/ggml-common.h`, and
  `ggml/src/ggml-cpu/ggml-cpu.c`;
- `gguf-py/gguf/constants.py` and `gguf-py/gguf/quants.py`;
- `benchmarks/baselines/llama_cpp/patches/0001-support-mixed-fp8-nvfp4-compressed-tensors.patch` and
  `benchmarks/baselines/llama_cpp/convert-patched.sh`.

## Conversion architecture

llama.cpp uses two materially different stages.

### Hugging Face to GGUF

`convert_hf_to_gguf.py` is a Python control and serialization program. It selects an architecture-specific converter,
loads configuration with `trust_remote_code=False` in the normal model path, maps source names through the centralized
GGUF tensor map, and writes metadata and tensors with `gguf-py`.

The normal local Safetensors path uses a custom header/index parser and NumPy `memmap`; lazy tensor closures and
PyTorch meta tensors defer payload work. PyTorch and NumPy perform most tensor arithmetic, reshapes, casts and
vectorized quantization, while Python performs model mapping, metadata construction, iteration and file writing.
The normal output types are F32, F16, BF16, Q8_0 and selected legacy formats; this stage is not a general native
persistent FP8 writer.

This path is not a bounded gem16 compiler: `GGUFWriter` retains converted tensor records until writing, has no
absolute RSS cap, and normal output opens the final file directly. It does not provide gem16's complete lock-bound
source hashes, tensor coverage manifest, atomic `.incomplete` publication or final artifact verification. Some
converter paths also use `trust_remote_code=True` for tokenizer handling, and source index/path validation is weaker
than the gem16 input contract.

### Native GGUF quantization

The documented follow-on flow is:

```text
python3 convert_hf_to_gguf.py MODEL --outfile model-bf16.gguf --outtype bf16
llama-quantize model-bf16.gguf model-Q4_K_M.gguf Q4_K_M
```

`tools/quantize/main.cpp` forwards to the native `llama-quantize` implementation. `src/llama-quant.cpp` loads GGUF
with mmap where supported, classifies tensor types, and processes tensor payloads through native ggml quantizers.
`ggml_quantize_chunk()` dispatches the C codecs in `ggml/src/ggml-quants.c`; quantization work is divided over
explicit worker threads and whole-row chunks. Tensor processing remains sequential at the outer tensor level.

This native stage is the useful performance precedent: explicit threads, mmap-backed input, block/row work partitioning,
reference codecs, optimized codecs and quantization tests. It is not an integrity precedent. The inspected output path
writes a final-named file directly, rewrites a header placeholder, has no atomic rename or fsync publication contract,
and does not provide gem16's complete source/compiler/output provenance. Its selection code can visibly but
permissively fall back to another type, including F16, for unsupported shapes or policies; gem16 must fail or record an
explicit diagnostic fallback instead.

## FP8, MXFP4 and NVFP4 findings

### FP8

The inspected llama.cpp tree has no generic persistent GGML FP8 weight type. `convert_hf_to_gguf.py --fp8-as-q8`
converts FP8 source weights to Q8_0; otherwise FP8 source tensors are dequantized to a supported float output. The
local mixed Gemma patch therefore produces Q8 attention, not persistent FP8 attention and not gem16's rowwise
`F8_E4M3` plus BF16 scale contract.

### MXFP4

`GGML_TYPE_MXFP4` is a native C/C++ type and `MXFP4_MOE` is exposed as a standard native quantization mode. The
layout is a 32-element block with E8M0 scale information and packed E2M1 values. The native ggml codec and tests are
useful reference material for block scheduling and validation.

### NVFP4

`GGML_TYPE_NVFP4` uses 64-element superblocks: four local 16-element groups stored as four UE4M3 scale bytes
followed by four contiguous 8-byte packed E2M1 subblocks (32 payload bytes total). The C reference encoder/decode
paths live in `ggml/src/ggml-quants.c`; CUDA has runtime/decode and activation staging paths. `gguf-py/gguf/quants.py`
contains NVFP4 dequantization/repacking support but no complete ordinary-float-to-NVFP4 production quantizer
equivalent to the native C codec.

The normal `llama-quantize` CLI does not expose a simple whole-model NVFP4 mode. Selective NVFP4 can be requested with
tensor-type overrides and output-tensor options. It does not automatically establish gem16's complete companion
weight-global/input-global scale contract.

### Local mixed FP8/NVFP4 patch

The tracked patch under `benchmarks/baselines/llama_cpp/patches/` allows compressed-tensors groups containing both
`float-quantized` and `nvfp4-pack-quantized` families. Its flow repacks NVFP4 and dequantizes FP8; with
`--fp8-as-q8`, FP8 attention becomes Q8_0. It is a practical llama.cpp diagnostic path, not exact gem16 format
parity. The local repository did not perform a 26B BF16-to-GGUF conversion in this research; the retained 26B GGUF
artifacts were downloaded references.

## Exact incompatibilities with gem16 NVFP4

The llama.cpp implementation must not be copied as if it were gem16's format:

1. llama.cpp aggregates four 16-element groups in one 64-element superblock, stored as four scale bytes followed
   by four contiguous 8-byte packed subblocks; gem16's contract keeps local scales as companion tensors and uses a
   different final runtime placement.
2. llama.cpp's UE4M3 decoder compensates for doubled FP4 LUT values, effectively using an E4M3/2 convention; gem16
   uses the explicitly frozen ordinary E4M3FN/E2M1 contract.
3. llama.cpp's global/input scale handling is not equivalent to gem16's explicit global weight and activation divisor
   semantics, and the inspected graph does not visibly consume every input-scale side tensor.
4. llama.cpp's reference scale recipe and tie behavior (including `amax / 6` in its NVFP4 reference path) differ from
   gem16's versioned scale-selection and rounding contract.
5. Gemma 4 router/per-expert scale folding is model-specific. llama.cpp's folding in `conversion/gemma.py` is a
   semantic clue only; gem16 must prove the placement with its own source roles, equations and operator tests.

## Reuse guidance for future agents

Study and potentially adopt the following concepts after differential tests: separation of model mapping from native
codecs, explicit worker threads, bounded row/block processing, reference-versus-optimized codec tests, and native
per-tensor type selection.

Do not copy the following behavior: giant BF16/F16 intermediate GGUF conversion solely to reach native quantization,
permissive silent precision fallback, direct final-file overwrite, absent source/compiler/output hashes, missing
absolute memory cap, weak source/index validation, unbounded remote revision behavior, or tokenizer paths that require
`trust_remote_code`.

llama.cpp is MIT licensed. Any copied source must retain its exact license/copyright notice and record the pinned
source revision, copied files, patch/provenance and semantic differences. The ignored local cache is never a runtime
or build dependency. A future reuse decision must add a `docs/DECISIONS.md` entry and differential byte/numerical tests.

## Resulting gem16 boundary

The accepted design is a strict hybrid control plane/native data plane:

- Python may generate exact plans, verify immutable locks, orchestrate a user-facing command, publish canonical
  Safetensors/JSON and produce small independent evidence/oracle reports.
- Promoted large tensor arithmetic and billion-element comparison loops belong in the shared versioned native C++20
  compiler data plane.
- M05's `gem16-fp8-compiler` is the first native seed. A future unified `gem16-checkpoint-compiler` may extend it for
  M06 NVFP4, M07 Q4_0/NVFP4 head and M18 native comparisons; that future name is not a current runnable command.
- Gem16 reads locked Safetensors directly into final-format payloads and does not require a giant intermediate GGUF.
- M04's accepted Python `copy-v1` scaffold remains historical infrastructure evidence and performs no numerical
  conversion.

## Limitations

This is a source audit, not an execution or performance result. No conversion, benchmark, memory measurement or
quality claim is made here. The inspected commit differs from the desired benchmark pin noted above, and any future
agent must re-audit the pinned checkout before making current-upstream claims.
