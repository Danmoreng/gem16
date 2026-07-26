# AGENTS.md

## Project: gem16 — 16 GB CUDA Inference Engine

**Status:** Repository initialization specification  
**Hardware class:** NVIDIA CUDA GPUs with approximately 16 GB VRAM
**First optimization target:** Blackwell, compute capability 12.0
**Primary model:** Gemma 4 12B Unified instruction-tuned model  
**Primary checkpoint:** `unsloth/gemma-4-12b-it-NVFP4`, pinned to an exact Hugging Face revision  
**Primary workload:** Text-only, batch-one interactive inference  
**Primary competitor:** Current upstream `llama.cpp` with CUDA and native Blackwell NVFP4 support  
**Document date:** 2026-07-21

This file is the top-level instruction set for every coding agent working in this repository. Follow it before following task-specific prompts. A task-specific prompt may narrow the work, but it must not silently weaken the correctness, benchmarking, reproducibility, or memory rules in this file.

---

# 1. Mission

Build a specialized, production-quality CUDA inference engine for Gemma 4 12B Unified that:

1. Loads an existing Hugging Face NVFP4 checkpoint directly.
2. Does not require users to produce a converted, requantized, or repacked checkpoint on disk.
3. Exploits Blackwell-native NVFP4 and FP8 Tensor Core operations on the first reference GPU.
4. Is optimized first for batch-one interactive decoding.
5. Supports useful long contexts within 16 GB VRAM.
6. Preserves model quality within explicitly measured tolerances.
7. Beats a rigorously configured `llama.cpp` baseline by as much as practical.
8. Reports speedups honestly, including differences in tensor formats, quality, context, memory use, and feature support.
9. Keeps the architecture narrow enough that model-specific kernel fusion is preferred over a generic graph framework.
10. Remains debuggable, testable, reproducible, and suitable for continued optimization by coding agents.
11. Evolves into a high-performance engine for 16 GB CUDA GPUs without weakening the first Blackwell path.

The project succeeds only when measured end-to-end inference is faster than the baseline on the target GPU without hidden quality regressions or benchmark shortcuts.

---

# 2. Project priorities

When priorities conflict, use this order:

1. **Correctness**
2. **Benchmark integrity**
3. **Stable execution within 16 GB VRAM**
4. **Batch-one decode latency and throughput**
5. **Prompt-processing throughput**
6. **Long-context capability**
7. **Startup and model-loading latency**
8. **Batch throughput**
9. **Multimodal support**
10. **Portability to additional 16 GB CUDA GPUs or models**

Do not trade a higher-priority requirement for a lower-priority one without an explicit written decision in `docs/DECISIONS.md`.

---

# 3. Non-negotiable constraints

## 3.1 Direct checkpoint loading

The normal user workflow must accept a Hugging Face checkpoint directory directly:

```bash
gem16-run \
  --model /models/unsloth-gemma-4-12b-it-NVFP4 \
  --prompt "Explain why the sky is blue."
```

The engine must read, at minimum:

- `config.json`
- `model.safetensors` or a Safetensors shard index
- `tokenizer.json`
- `tokenizer_config.json`
- `generation_config.json`, when present
- `chat_template.jinja`, when present
- `processor_config.json`, when required

The engine must not require:

- GGUF conversion
- TensorRT engine generation
- an offline CUTLASS layout conversion
- requantization
- a custom binary checkpoint
- a generated plan tied to a driver version
- a second persistent copy of the weights

A small metadata cache is allowed if all of the following are true:

- It contains no model weight payload.
- It is optional and reproducible.
- Deleting it does not affect output.
- Its format is versioned.
- It records the exact source checkpoint revision and tensor checksums.

## 3.2 No silent precision fallback

Never silently run an NVFP4 tensor through BF16, FP16, FP8, or generic dequantization because a native kernel is missing.

Permitted behavior:

- fail with a clear error;
- choose a documented fallback only when the user explicitly enables it;
- emit a visible per-tensor fallback report;
- record fallback use in benchmark output.

A benchmark result is invalid if a claimed NVFP4 path silently falls back to a higher-precision implementation.

## 3.3 No allocations in the token loop

After graph preparation and cache reservation:

- no `cudaMalloc`;
- no `cudaFree`;
- no pageable host allocation;
- no growing C++ containers;
- no filesystem access;
- no dynamic kernel compilation;
- no tokenizer vocabulary mutation;
- no hidden framework workspace allocation.

All recurring buffers must come from preallocated arenas.

## 3.4 No CPU weight offload in performance results

All weights needed for each generated token must remain resident in VRAM.

CPU or system-RAM use is allowed for:

- memory-mapped source files;
- tokenizer data;
- model metadata;
- staging during startup;
- inactive multimodal tensors in text-only mode;
- benchmark orchestration;
- inactive sessions;
- optional disk-backed prompt-cache data.

Do not present a PCIe-streamed configuration as the primary high-performance result.

## 3.5 Do not optimize benchmark semantics away

Do not obtain speedups by:

- reducing generated token count;
- changing prompt tokens;
- disabling required attention layers;
- truncating context without disclosure;
- reducing vocabulary;
- approximating the output head without a separately labeled experiment;
- changing sampling parameters;
- using a different chat template;
- skipping the final logit softcap;
- using a lower-quality checkpoint without disclosure;
- excluding warm-up from one engine but not another;
- timing only favorable kernels while calling it end-to-end inference;
- reporting MTP proposed tokens as accepted output tokens;
- comparing a batch-one result against a batched baseline;
- using prompt-cache hits for one engine only.

---

# 4. Ground-truth model contract

## 4.1 Source of truth

Never hard-code architecture assumptions solely from this document. At model load, parse `config.json`, inspect the tensor manifest, and validate all required invariants.

The initial checkpoint is expected to describe:

- architecture: `Gemma4UnifiedForConditionalGeneration`;
- model type: `gemma4_unified`;
- text model type: `gemma4_unified_text`;
- 48 decoder layers;
- hidden size 3840;
- MLP intermediate size 15360;
- 16 query heads;
- 8 local KV heads;
- 1 global KV head;
- local attention head dimension 256;
- global attention head dimension 512;
- sliding window 1024;
- maximum position count 262144;
- vocabulary size 262144;
- tied input and output embeddings;
- `attention_k_eq_v = true`;
- five sliding-attention layers followed by one full-attention layer, repeated eight times;
- final layer using full attention;
- final-logit softcap 30.0;
- local RoPE theta 10000;
- global proportional RoPE theta 1000000;
- global partial rotary factor 0.25;
- GELU tanh approximation in the MLP.

If the checkpoint differs, fail validation unless support for the difference was deliberately implemented and tested.

## 4.2 The checkpoint is mixed precision

Do not refer to the selected checkpoint as “all NVFP4.”

The initial Unsloth checkpoint currently declares:

- attention projection weights: FP8, per-channel;
- attention input activations: dynamic FP8, per token;
- MLP gate/up/down weights: packed NVFP4;
- MLP input activations: dynamic/local NVFP4;
- NVFP4 group size: 16;
- NVFP4 block-scale type: FP8 E4M3;
- global tensor scales in addition to local block scales;
- FP8 KV-cache quantization metadata;
- selected embeddings, output, and multimodal projection tensors excluded from NVFP4.

Treat `quantization_config` as executable schema, not descriptive metadata.

## 4.3 NVFP4 numerical model

The engine must model NVFP4 as hierarchical scaling:

```text
x_real = x_e2m1 * block_scale_e4m3 * global_scale
```

For the target recipe:

- element format: signed E2M1, 4 bits;
- two elements packed per byte;
- one local E4M3 scale per 16 consecutive elements along the contracting dimension;
- one tensor-global scale;
- accumulator: FP32 unless a proven hardware path and correctness test justify another choice.

Do not confuse:

- NVFP4 with generic INT4;
- NVFP4 with MXFP4;
- E4M3 block scales with E8M0 block scales;
- a packed storage format with an MMA operand layout;
- weight-only FP4 with W4A4 execution.

## 4.4 Tensor inventory is authoritative

The first repository tool to implement is `gem16-inspect`.

It must print and optionally export JSON containing:

- tensor name;
- shape;
- logical shape;
- storage dtype;
- quantization class;
- byte offset;
- byte length;
- alignment;
- source Safetensors shard;
- expected role;
- local scale tensor;
- global scale tensor;
- input scale tensor;
- transposition or layout metadata;
- whether loaded in text-only mode;
- whether aliased because of tied embeddings;
- total bytes by tensor class.

No kernel implementation may guess tensor names or shapes that can be obtained from the manifest.

---

# 5. Checkpoint policy

## 5.1 Initial checkpoint

Start with:

```text
unsloth/gemma-4-12b-it-NVFP4
```

At repository initialization, resolve and pin:

- the full Hugging Face commit SHA;
- every downloaded filename;
- byte size;
- SHA-256 checksum;
- Xet/LFS object identity if available;
- model-card revision;
- config revision;
- tokenizer revision.

Store the lock in:

```text
models/gemma4-12b-nvfp4.lock.json
```

Never use `revision=main` in an automated benchmark.

## 5.2 Candidate secondary checkpoint

A second checkpoint may be supported after the primary one is stable, for example:

```text
AxionML/Gemma-4-12B-NVFP4
```

Do not assume that two repositories labeled “NVFP4” use the same:

- quantized tensor set;
- scale naming;
- packing;
- attention precision;
- KV-cache recipe;
- calibration;
- global scales;
- tensor dimensions;
- tokenizer revision.

Add a loader adapter only after manifest-level comparison.

## 5.3 NVIDIA checkpoint wording

As of the document date, do not claim that NVIDIA publishes an official dense Gemma 4 12B NVFP4 checkpoint unless a current official repository is verified.

NVIDIA recipes or larger official checkpoints may inform implementation, but provenance must remain accurate.

## 5.4 Offline conversion policy

For the engine:

- offline conversion is forbidden as a runtime prerequisite;
- offline requantization is forbidden for primary results;
- persistent repacked weight files are forbidden for the default path.

At load time, an in-memory layout transformation is permitted only if:

1. native consumption of source layout is not practical;
2. quantized values and scales are preserved exactly;
3. no second persistent GPU copy remains;
4. peak host and device memory are measured;
5. the transformation is streamed into final GPU allocations;
6. model-load time includes the transformation;
7. the engine still accepts the original checkpoint directory;
8. the transformation is documented in `docs/WEIGHT_LAYOUT.md`.

The preferred implementation consumes the existing packed tensor representation directly.

---

# 6. Hardware target

## 6.1 Required target

The product hardware class is a single NVIDIA CUDA GPU with approximately 16 GB of VRAM. The first optimized and
validated backend is Blackwell:

```text
NVIDIA Blackwell GPU
Approximately 16 GB VRAM
Compute capability 12.0
Fifth-generation Tensor Cores
```

Do not bake a board name, board form factor, nominal memory bandwidth, or power envelope into engine semantics.
Record those as benchmark-machine facts. Initial releases may fail fast on non-Blackwell devices until
their architecture-specific backend is implemented and validated.

## 6.2 Runtime capability check

At startup, print:

- GPU name;
- UUID;
- compute capability;
- VRAM total and free;
- driver version;
- CUDA runtime version;
- CUDA driver version;
- compiled architectures;
- native NVFP4 kernel availability;
- FP8 kernel availability;
- CUDA Graph availability;
- memory-pool configuration;
- clocks and power limit when queryable.

If the binary lacks the architecture-accelerated NVFP4 code path, exit instead of benchmarking a fallback.

## 6.3 Compilation target

Prefer architecture-specific translation units for the native path.

The build must:

- compile the first Blackwell path for SM 12.0;
- enable the architecture-specific target required for block-scaled NVFP4 MMA when supported by the pinned CUDA toolkit;
- make native-path compilation visible in build logs;
- expose a runtime `--print-kernel-capabilities` command;
- avoid embedding unnecessary architectures in performance builds;
- support a debug build with device assertions and line information;
- support a release build with LTO where beneficial.

Do not guess toolkit flags. Pin a toolchain known to emit the required SM120/SM120a instructions, disassemble representative kernels, and verify the expected MMA instructions are present.

## 6.4 Toolchain lock

Create:

```text
toolchains/blackwell16gb.lock
```

Record:

- Linux distribution;
- kernel;
- compiler;
- CMake;
- Ninja;
- CUDA toolkit;
- NVIDIA driver;
- CUTLASS commit if used;
- Python version for tools;
- `llama.cpp` commit;
- model revision;
- GPU VBIOS if relevant.

The first validated toolchain becomes the reference toolchain. Updates require fresh correctness and benchmark runs.

---

# 7. Scope

## 7.1 Phase-one supported behavior

Required:

- Linux x86-64;
- one approximately 16 GB Blackwell CUDA GPU;
- text input;
- text output;
- batch size one;
- greedy decoding;
- temperature sampling;
- top-k and top-p sampling;
- repetition penalty;
- prompt lengths through at least 64K;
- generation lengths through at least 1K tokens;
- direct Safetensors loading;
- native mixed FP8/NVFP4 execution;
- FP8 or correctness-mode BF16 KV cache;
- tied embedding/output projection;
- standard Gemma chat template;
- thinking enabled and disabled;
- deterministic benchmark mode;
- CUDA Graph decode path.

## 7.2 Stretch scope

Later:

- 128K production context;
- 256K experimental context;
- native Gemma MTP assistant;
- image input;
- audio input;
- video frame input;
- continuous batching;
- multiple concurrent sessions;
- prefix caching;
- prompt-cache persistence;
- Windows;
- additional 16 GB NVIDIA CUDA architecture generations;
- additional Blackwell boards and power envelopes;
- other Gemma 4 sizes.

## 7.3 Explicit non-goals for initial development

Do not spend initial milestones on:

- AMD;
- Apple Silicon;
- CPU inference;
- multi-GPU;
- training;
- fine-tuning;
- arbitrary Transformers architectures;
- a generic tensor graph compiler;
- OpenAI-compatible server APIs;
- Kubernetes;
- distributed serving;
- speculative decoding with an unrelated draft model;
- custom quantization recipes;
- a GUI.

---

# 8. Performance objective

The primary objective is not “high Tensor Core utilization.” It is:

> Maximum correct end-to-end output tokens per second and minimum inter-token latency on one 16 GB CUDA GPU at batch size one, compared fairly against current upstream llama.cpp.

Optimize separately for:

1. model load;
2. prompt ingestion;
3. first-token production;
4. steady-state ordinary decode;
5. MTP-assisted decode;
6. long-context decode.

Do not assume the same kernel is optimal for prefill and decode.

---

# 9. llama.cpp baseline contract

## 9.1 Why llama.cpp is the primary baseline

`llama.cpp` is the practical local-inference competitor and now contains a merged native Blackwell NVFP4 CUDA path.

The project must beat what a careful `llama.cpp` user can run, not an intentionally weak configuration.

## 9.2 Baseline verification gate

Before claiming a parity benchmark, prove all of the following for the exact model snapshot:

1. The current `llama.cpp` converter accepts the source checkpoint.
2. NVFP4 values and block scales are preserved for NVFP4 tensors.
3. Mixed FP8 tensors are mapped and recorded.
4. The resulting GGUF loads under the Gemma 4 architecture.
5. Native Blackwell NVFP4 kernels are invoked.
6. No unexpected CPU offload occurs.
7. Tensor type counts are exported.
8. Output quality is within the accepted comparison threshold.

Generic NVFP4 kernel support alone is not sufficient proof of exact Gemma 4 mixed-checkpoint parity.

## 9.3 Baseline tiers

Maintain three separate baseline labels.

### A. Same-source closest-parity baseline

- Same pinned Hugging Face source checkpoint.
- Convert only for `llama.cpp`.
- Preserve NVFP4 tensors where supported.
- Record how FP8 and BF16 tensors map.
- Record any dequantization or Q8 conversion.
- Do not call this “exact format parity” unless the tensor inventory proves exact parity.

### B. Native-NVFP4 llama.cpp baseline

- Uses GGUF NVFP4 tensors.
- Proves Blackwell native NVFP4 MMA execution.
- May differ in non-NVFP4 tensor handling.
- Used to compare native FP4 kernel and scheduling quality.

### C. Fastest practical llama.cpp baseline

- Best quality-acceptable GGUF that fits.
- Uses recommended CUDA options.
- May be Q4, Q5, Q6, or mixed.
- Represents the result an expert local user would choose.
- Quality and size differences must be disclosed.

## 9.4 Baseline pinning

Store:

```text
benchmarks/baselines/llama_cpp/
  README.md
  build.sh
  convert.sh
  run.sh
  commit.txt
  build-info.txt
  tensor-inventory.json
  quality.json
```

Record:

- full commit SHA;
- PRs or patches;
- build flags;
- compiler;
- CUDA toolkit;
- command line;
- environment variables;
- GGUF checksum;
- tensor types;
- GPU-layer count;
- context;
- batch settings;
- flash-attention setting;
- KV-cache type;
- graph settings;
- thread count;
- tokenizer;
- chat template;
- sampling options.

## 9.5 Speedup wording

Use:

```text
speedup = our_engine_metric / llama_cpp_metric
```

for throughput, and:

```text
latency_reduction = 1 - our_engine_latency / llama_cpp_latency
```

for latency.

Every headline must identify:

- prefill or decode;
- batch size;
- context;
- output length;
- ordinary or speculative decoding;
- checkpoint;
- precision;
- quality result;
- VRAM use.

Never publish a single unexplained “X times faster” number.

---

# 10. Benchmark methodology

## 10.1 Benchmark modes

Implement:

```bash
gem16-bench model-load
gem16-bench prefill
gem16-bench decode
gem16-bench end-to-end
gem16-bench kernel
gem16-bench memory
gem16-bench quality
gem16-bench mtp
```

Output machine-readable JSON and a concise terminal summary.

## 10.2 Core prompt-processing matrix

At minimum:

| Prompt tokens | Output tokens | Batch |
|---:|---:|---:|
| 128 | 1 | 1 |
| 512 | 1 | 1 |
| 2048 | 1 | 1 |
| 8192 | 1 | 1 |
| 32768 | 1 | 1 |
| 65536 | 1 | 1 |
| 131072 | 1 | 1, when supported |

Report prompt tokens per second and time to first token.

## 10.3 Core decode matrix

At minimum:

| Existing context | Generated tokens | Batch |
|---:|---:|---:|
| 128 | 256 | 1 |
| 2048 | 256 | 1 |
| 8192 | 256 | 1 |
| 32768 | 256 | 1 |
| 65536 | 256 | 1 |
| 131072 | 256 | 1, when supported |

Report:

- average output tokens/s;
- median inter-token latency;
- p95 inter-token latency;
- p99 inter-token latency;
- minimum and maximum;
- time to first generated token;
- peak VRAM;
- steady-state VRAM;
- average board power;
- average core clock;
- temperature.

## 10.4 Repetition

Default:

- 3 warm-up runs;
- 10 measured runs;
- 95% confidence interval;
- raw run output retained;
- median reported as primary;
- mean and standard deviation also reported.

For noisy results, increase repetitions instead of selecting the best run.

## 10.5 Environment control

Before benchmark collection:

- close unrelated GPU workloads;
- record whether the GPU drives a display;
- set a stable power mode when permitted;
- set persistence mode when permitted;
- lock graphics clocks when permitted;
- record power limit;
- record ambient and GPU temperature;
- wait for thermal steady state;
- use the same conditions for all engines;
- disable prompt-cache reuse unless explicitly benchmarking it;
- use identical prompt token IDs.

If clocks cannot be locked, record them continuously.

## 10.6 Timing boundaries

Provide both:

### Core GPU timing

Excludes:

- model download;
- model load;
- tokenizer initialization;
- prompt text tokenization;
- terminal output.

Includes:

- input-token upload where applicable;
- model forward;
- sampling on GPU;
- recurrent KV/cache updates.

### End-to-end timing

Includes:

- prompt processing;
- CPU scheduling;
- GPU execution;
- sampling;
- token transfer to the host.

Tokenization time must be separately reported.

## 10.7 Benchmark data

Store every run under:

```text
benchmarks/results/<date>/<git-sha>/<machine-id>/
```

Include:

- `system.json`
- `model.json`
- `engine.json`
- `commands.txt`
- `raw.jsonl`
- `summary.json`
- `nsys/`, when collected
- `ncu/`, when collected

Never overwrite a prior result.

---

# 11. Quality and correctness contract

## 11.1 Reference hierarchy

Use several references:

1. A trusted Python implementation that understands the exact compressed checkpoint.
2. The original BF16 Gemma 4 model when memory and compute permit.
3. The selected checkpoint in a supported reference runtime.
4. `llama.cpp` after successful conversion and validation.
5. CPU reference implementations for individual operators.

No single runtime is assumed infallible.

## 11.2 Required validation levels

### Level 0: File integrity

- checksum;
- header parse;
- offset bounds;
- no overlapping tensors;
- correct byte lengths;
- supported dtype;
- scale presence;
- shape divisibility;
- tied-weight identity.

### Level 1: Operator tests

Test:

- BF16 RMSNorm;
- FP8 quantize/dequantize;
- NVFP4 unpack;
- NVFP4 block/global scale application;
- NVFP4 activation quantization;
- FP8 GEMM;
- NVFP4 GEMM;
- GeGLU;
- RoPE;
- local attention;
- global attention;
- softmax;
- KV append/read;
- final logit softcap;
- top-k;
- top-p;
- sampling.

### Level 2: Layer tests

For representative local and global layers:

- compare pre-norm;
- projection outputs;
- Q/K/V or K=V behavior;
- attention scores;
- attention output;
- post-attention residual;
- MLP gate/up;
- activation product;
- down projection;
- final residual.

### Level 3: Model logits

Compare:

- first-token logits;
- logits across 32 decode steps;
- top-1 agreement;
- top-5 overlap;
- top-20 overlap;
- cosine similarity;
- KL divergence;
- maximum absolute error;
- RMS error.

### Level 4: Generation

Compare greedy generation on a fixed prompt suite.

A small numerical difference may alter later greedy tokens. Therefore, record both token agreement and stepwise logit metrics.

### Level 5: Task quality

Run a stable task subset and a perplexity-style evaluation.

## 11.3 Tolerance policy

Do not invent global tolerances before gathering reference distributions.

Start with strict operator tolerances, then establish per-operation budgets. Store them in:

```text
tests/tolerances.yaml
```

Any relaxation requires:

- failing example;
- numerical explanation;
- quality impact;
- review;
- updated golden data;
- no performance-only justification.

## 11.4 Performance changes require correctness evidence

Every optimization PR must include:

- tests passing;
- before/after logit metrics;
- before/after benchmark;
- peak VRAM;
- kernel path confirmation;
- profiling evidence;
- explanation of numerical reordering.

A speedup with unexplained quality drift is not accepted.

---

# 12. Memory contract

## 12.1 Nominal budget

The card has 16 GB nominal VRAM. Do not budget against all of it.

Initial planning target:

| Allocation | Target |
|---|---:|
| Persistent model tensors | Measured, preferably below 10.0 GB in text-only mode |
| KV cache at 64K | Below 1.0 GB |
| Activation arenas | Below 1.0 GB |
| CUDA Graph/private pools | Below 0.8 GB |
| Kernel workspaces | Below 0.8 GB |
| CUDA context/libraries | Measured |
| Safety margin | At least 0.7 GB |
| Total peak | Below 15.3 GB |

These are targets, not assumptions. Export actual allocator accounting.

## 12.2 Text-only loading

The initial engine is text-only.

Do not upload tensors used only for:

- image patch projection;
- audio projection;
- video processing;
- modality preprocessing.

Keep unused tensors memory-mapped on the host or leave them unopened.

The tensor manifest must explicitly show which checkpoint bytes were excluded from GPU residency.

## 12.3 KV-cache formula

Implement a cache-size calculator from parsed architecture metadata.

For a one-byte FP8 cache, calculate both possibilities until model semantics are proven:

```text
shared K/V:
  bytes = layers * tokens * kv_heads * head_dim

separate K and V:
  bytes = 2 * layers * tokens * kv_heads * head_dim
```

Sliding-window layers allocate only the active window plus required metadata. Global layers allocate through the active context.

Do not use the maximum 262144-token allocation unless requested. Reserve by context tier.

## 12.4 Context tiers

Provide explicit profiles:

| Profile | Context target | Purpose |
|---|---:|---|
| `interactive` | 8192 | Lowest latency |
| `standard` | 32768 | Default |
| `long` | 65536 | Production long context |
| `xlong` | 131072 | Supported after optimization |
| `max` | 262144 | Experimental |

Each profile has a deterministic arena plan.

## 12.5 Allocator

Implement a device arena with named regions:

- immutable model weights;
- scales;
- KV cache;
- activations A/B;
- logits;
- sampling workspace;
- graph workspace;
- temporary prefill workspace;
- optional MTP workspace.

Requirements:

- alignment appropriate to global loads and MMA operands;
- allocation report;
- canary/guard mode in debug builds;
- no fragmentation after initialization;
- high-water mark reporting;
- deterministic addresses for graph capture.

---

# 13. Repository layout

Initialize approximately:

```text
.
├── AGENTS.md
├── CMakeLists.txt
├── LICENSE
├── README.md
├── cmake/
│   ├── CompilerWarnings.cmake
│   ├── CudaArchitectures.cmake
│   └── Sanitizers.cmake
├── include/
│   └── gem16/
│       ├── engine.h
│       ├── model.h
│       ├── sampling.h
│       ├── status.h
│       └── types.h
├── src/
│   ├── cli/
│   │   ├── run_main.cpp
│   │   ├── bench_main.cpp
│   │   └── inspect_main.cpp
│   ├── model/
│   │   ├── config.cpp
│   │   ├── manifest.cpp
│   │   ├── safetensors.cpp
│   │   ├── tokenizer.cpp
│   │   └── checkpoint_loader.cpp
│   ├── runtime/
│   │   ├── engine.cpp
│   │   ├── memory_plan.cpp
│   │   ├── kv_cache.cpp
│   │   ├── cuda_graphs.cpp
│   │   └── scheduler.cpp
│   └── cuda/
│       ├── dispatch.cu
│       ├── nvfp4/
│       ├── fp8/
│       ├── attention/
│       ├── norm/
│       ├── rope/
│       ├── sampling/
│       └── fused/
├── tests/
│   ├── unit/
│   ├── cuda/
│   ├── integration/
│   ├── golden/
│   └── tolerances.yaml
├── tools/
│   ├── fetch_model.py
│   ├── inspect_checkpoint.py
│   ├── generate_golden.py
│   ├── compare_logits.py
│   ├── benchmark.py
│   └── llama_cpp/
├── benchmarks/
│   ├── baselines/
│   ├── prompts/
│   ├── results/
│   └── schemas/
├── models/
│   └── gemma4-12b-nvfp4.lock.json
├── toolchains/
│   └── blackwell16gb.lock
├── docs/
│   ├── ARCHITECTURE.md
│   ├── BENCHMARKING.md
│   ├── CHECKPOINT_FORMAT.md
│   ├── CORRECTNESS.md
│   ├── DECISIONS.md
│   ├── MEMORY.md
│   ├── PERFORMANCE_LEDGER.md
│   ├── ROADMAP.md
│   └── WEIGHT_LAYOUT.md
└── third_party/
    └── README.md
```

Do not create abstractions for hypothetical models until a second real model requires them.

---

# 14. Language and dependency policy

## 14.1 Runtime language

- C++20
- CUDA C++
- C for narrow system interfaces only
- Python for tooling, reference generation, and benchmark orchestration

## 14.2 Runtime dependency policy

Prefer a small dependency surface.

Allowed candidates:

- CUDA Runtime and Driver APIs;
- CUTLASS/CuTe, pinned, for native kernel construction or reference;
- a small JSON parser;
- a small, audited Safetensors parser or an in-repo implementation;
- SentencePiece/tokenizer support only when needed;
- NVTX for profiling;
- GoogleTest or Catch2 for host tests.

Avoid in the final runtime:

- PyTorch;
- Transformers;
- vLLM;
- TensorRT-LLM;
- a generic graph framework;
- dynamic Python embedding;
- JIT compilation in the generation path.

Reference tools may use these.

## 14.3 Vendoring

Every vendored dependency requires:

- pinned commit;
- license review;
- source URL;
- update instructions;
- reason it is vendored;
- no local modifications without a patch file or fork reference.

---

# 15. C++ and CUDA coding rules

## 15.1 General C++

- Use RAII for CUDA handles and allocations.
- Mark fallible returns `[[nodiscard]]`.
- Prefer explicit `Status`/`Result<T>` over exceptions in runtime code.
- Use fixed-width integer types for file formats.
- Check integer overflow in tensor-size calculations.
- No undefined behavior for packed nibble access.
- No pointer reinterpretation without alignment proof.
- Use spans/views instead of raw pointer-length pairs when practical.
- Avoid virtual dispatch in hot paths.
- Avoid shared ownership unless required.
- No global mutable state.
- No hidden singletons.
- Keep public headers minimal.

## 15.2 CUDA

- Check every CUDA API result.
- Check launch errors in debug and test builds.
- No `cudaDeviceSynchronize()` in the hot path.
- Use stream-local events for dependencies.
- Use asynchronous copies where useful.
- Avoid unnecessary host round trips.
- Use NVTX ranges around model phases and major kernels.
- Document block dimensions, tile sizes, shared memory, and expected occupancy.
- Record register count and spill status for hot kernels.
- Do not accept local-memory spills in a hot decode kernel without benchmark evidence.
- Use `__restrict__` only when aliasing is proven.
- Keep an unfused reference kernel for every complex fused kernel.
- Keep an environment or command-line switch to disable each optimization family.

## 15.3 Generated code

Generated kernel tables are allowed only when:

- the generator is checked in;
- generation is deterministic;
- output contains a provenance header;
- CI verifies generated files are current;
- hand editing is prohibited.

---

# 16. Loader architecture

## 16.1 Safetensors parser

Implement:

- memory mapping;
- header length validation;
- JSON schema validation;
- little-endian interpretation;
- offset bounds;
- shape product overflow checks;
- dtype support;
- shard index support;
- duplicate tensor rejection;
- tensor-name normalization only through explicit maps.

The loader must not copy the full model into host RAM.

## 16.2 Quantization schema parser

Parse:

- `quant_method`;
- global format;
- config groups;
- regex targets;
- weight bit width;
- activation bit width;
- group size;
- scale dtype;
- strategy;
- dynamic/static flags;
- ignored tensors;
- KV-cache scheme.

Compile regex target rules once at startup.

## 16.3 Tensor classification

Classify tensors into:

- BF16;
- FP16;
- FP32;
- FP8 E4M3;
- packed NVFP4;
- NVFP4 local scales;
- NVFP4 global scales;
- NVFP4 input scales;
- tokenizer/metadata;
- unsupported.

Unsupported tensors cause a descriptive error containing name, shape, dtype, and relevant quantization group.

## 16.4 Device upload

Preferred:

- allocate final device region;
- stream source bytes directly from pinned staging windows;
- preserve packed representation;
- transform scale layout only when required;
- release each staging window promptly;
- avoid full-file pinned copies.

If scales require swizzling for MMA, evaluate:

1. swizzle once at load into final scale storage;
2. preserve original scales and swizzle per launch;
3. fuse scale loading/reordering into the kernel.

Choose based on measured load time, VRAM, and kernel speed. Document the choice.

## 16.5 Weight aliasing

Tied embeddings must use one resident allocation where the checkpoint permits.

Do not allocate a duplicate LM head.

---

# 17. Execution architecture

## 17.1 Separate prefill and decode plans

Create distinct execution plans.

### Prefill plan

Optimized for:

- many tokens;
- large GEMMs;
- high Tensor Core occupancy;
- tiled attention;
- chunked long-context processing;
- larger temporary workspace;
- fewer kernel launches through fusion.

### Decode plan

Optimized for:

- one token;
- narrow GEMMs/GEMVs;
- weight bandwidth;
- minimal launch count;
- persistent addresses;
- CUDA Graph replay;
- fused residual and normalization;
- no large temporary tensors;
- direct sampling.

Never force decode through a prefill-optimized graph solely for code reuse.

## 17.2 Execution plan immutability

After plan creation:

- tensor pointers do not change;
- kernel dispatch choices do not change unless an adaptive mode is explicitly enabled;
- graph-captured shapes are fixed;
- all workspace offsets are fixed;
- cache capacity is fixed.

## 17.3 CUDA streams

Start with one compute stream for correctness.

Add streams only for measured overlap, such as:

- asynchronous host token transfer;
- prefetch of metadata;
- independent output-head work where valid.

Do not add stream complexity without an Nsight timeline showing a benefit.

---

# 18. Kernel roadmap

## 18.1 RMSNorm

Implement:

- BF16 input/output reference;
- FP32 reduction;
- vectorized loads;
- fused residual add where numerically correct;
- optional fused quantization epilogue.

Candidate fusions:

```text
residual add -> RMSNorm -> FP8 quantize
residual add -> RMSNorm -> NVFP4 quantize
```

Validate exact epsilon and Gemma normalization semantics.

## 18.2 FP8 attention projections

The selected checkpoint uses FP8 attention projections.

Implement:

- per-channel FP8 weight scales;
- dynamic per-token activation scaling;
- FP32 or appropriate accumulation;
- fused input quantization;
- fused Q/K/(V)/O scheduling where useful;
- exact handling of `attention_k_eq_v`.

Do not assume a V tensor exists. Derive the projection graph from the manifest and reference implementation.

## 18.3 NVFP4 MLP projections

Implement native Blackwell W4A4 for:

- gate projection;
- up projection;
- down projection.

Requirements:

- group size 16;
- E4M3 local scales;
- global tensor scale;
- exact source nibble ordering;
- correct contracting-dimension layout;
- native block-scaled MMA;
- no full BF16 dequantized weight tile in global memory.

Provide:

- correctness/reference kernel;
- prefill kernel;
- batch-one decode kernel;
- fused gate/up scheduling;
- fused activation/down path where practical.

## 18.4 Activation quantization

Activation quantization is part of the critical path.

Optimize:

- amax calculation;
- global scale;
- local block scales;
- E2M1 packing;
- E4M3 scale conversion;
- layout generation for native MMA;
- fusion with RMSNorm or preceding epilogue.

Measure quantization cost separately. A fast GEMM with slow activation quantization is not a successful path.

## 18.5 MLP fusion

Gemma uses GELU tanh approximation.

Target pipeline:

```text
RMSNorm
-> activation quantization
-> gate projection
-> up projection
-> GELU_tanh(gate) * up
-> activation quantization
-> down projection
-> residual add
```

Explore:

- combined gate/up launch;
- shared input quantization;
- fused GELU/multiply;
- quantized intermediate without materializing BF16;
- down-projection epilogue with residual.

Keep intermediate precision choices explicit.

## 18.6 RoPE

Support two regimes.

### Sliding attention

- standard RoPE;
- theta 10000;
- configured head dimension.

### Full attention

- proportional RoPE;
- theta 1000000;
- partial rotary factor 0.25;
- configured global head dimension.

Create reference vectors from the trusted implementation.

Fuse RoPE with Q/K projection epilogues when validated.

## 18.7 Sliding-window attention

There are 40 sliding layers in the expected 48-layer pattern.

Optimize for:

- fixed 1024-token window;
- circular KV storage;
- batch one;
- FP8 cache;
- no cache compaction;
- fused score/softmax/value accumulation;
- K=V sharing where exact;
- decode-specialized kernel.

The local decode kernel should not scan tokens outside the active window.

## 18.8 Global attention

There are 8 expected full-attention layers.

Optimize separately:

- one global KV head;
- global head dimension 512;
- unified K/V semantics;
- growing context;
- chunked prefill;
- online softmax;
- FP8 cache;
- partial RoPE.

At long context, global attention may dominate. Profile by context tier.

## 18.9 KV cache

Implement:

- BF16 correctness mode;
- FP8 production mode;
- paged or contiguous global cache selected by benchmark;
- circular local cache;
- per-layer descriptors;
- cache reset;
- cache clone only if later required;
- exact position tracking;
- no per-token allocation.

Quantization scales must be stored with sufficient granularity to reproduce the reference recipe.

## 18.10 Embedding and output head

The tied embedding/output matrix is large and may remain BF16 in the selected checkpoint.

This is a critical decode bottleneck.

Implement:

- embedding gather;
- dedicated batch-one output projection;
- chunked vocabulary traversal;
- fused final logit softcap;
- fused top-k selection;
- optional fused repetition penalty;
- no full FP32 logits buffer unless needed;
- exact fallback path producing all logits for tests.

Do not approximate vocabulary search in primary quality results.

## 18.11 Logit softcap

Implement the exact Gemma final-logit softcap from the reference implementation.

Do not infer the formula from the value alone. Add a golden test.

Fuse it into the output projection or top-k pass.

## 18.12 Sampling

Implement on GPU:

- greedy;
- temperature;
- top-k;
- top-p;
- min-p if later supported;
- repetition penalty;
- deterministic seeded RNG.

Transfer only the selected token and small statistics to the host.

## 18.13 CUDA Graphs

Capture decode graphs for common profiles.

At minimum:

- ordinary decode, batch one;
- selected context descriptor shapes;
- greedy;
- sampled;
- later MTP verification lengths.

Graph capture must not include invalid dynamic allocations.

Measure graph launch improvement against direct launches.

## 18.14 Persistent kernels

Explore only after a stable graph path.

Candidates:

- persistent decode scheduler;
- fused norm/projection sequences;
- persistent output-head reduction.

Do not build a monolithic persistent kernel before profiling launch overhead and synchronization.

---

# 19. Long-context strategy

## 19.1 Default context

Use 32768 as the default user profile after correctness.

## 19.2 Production targets

- 64K must fit and run stably.
- 128K is the next optimization target.
- 256K is experimental and may have high prefill latency.

## 19.3 Chunked prefill

Implement chunked prefill to cap workspace.

Requirements:

- exact attention semantics;
- no loss of prefix information;
- stable cache positions;
- configurable chunk size;
- benchmark chunk sizes;
- no quadratic temporary allocation.

## 19.4 Context benchmark honesty

A model supporting 256K positions does not imply practical 256K ingestion.

Report:

- prefill wall time;
- peak memory;
- average tokens/s;
- global-attention cost;
- cache size;
- whether prompt chunking was used.

---

# 20. MTP roadmap

MTP is not part of the first correctness milestone.

When ordinary decode is stable:

1. Identify the official or compatible Gemma 4 12B assistant checkpoint.
2. Pin it separately.
3. Load it directly from Safetensors.
4. Share target embeddings and cache state where the architecture allows.
5. Quantize nothing beyond the published assistant checkpoint for the first result.
6. Measure assistant VRAM separately.
7. Implement verification lengths 1, 2, and 4.
8. Report proposed tokens, accepted tokens, rejected tokens, acceptance length, and effective output tokens/s.
9. Fall back adaptively when acceptance is poor.
10. Compare MTP against ordinary decode on the same prompts.

Never compare our MTP result against non-MTP `llama.cpp` without also reporting ordinary decode.

MTP acceptance quality matters more than proposed-token throughput.

---

# 21. Profiling policy

## 21.1 Tools

Use:

- Nsight Systems for timelines;
- Nsight Compute for hot kernels;
- NVTX annotations;
- CUDA events;
- `nvidia-smi` or NVML for power/clocks;
- SASS/PTX inspection for native instruction confirmation.

## 21.2 Required NVTX ranges

At minimum:

- tokenize;
- upload;
- prefill;
- decoder layer;
- local attention;
- global attention;
- attention projections;
- MLP;
- output head;
- sampling;
- graph replay;
- MTP propose;
- MTP verify.

## 21.3 Performance ledger

Maintain `docs/PERFORMANCE_LEDGER.md`.

For each optimization:

- date;
- commit;
- hypothesis;
- changed kernels;
- benchmark configuration;
- before;
- after;
- percentage;
- quality delta;
- VRAM delta;
- profile link;
- decision: keep/revert.

Do not keep an optimization because it “should be faster.”

---

# 22. Milestones and gates

## Milestone 0: Reproducible baselines

Deliver:

- pinned model;
- model checksum;
- reference runtime output;
- pinned `llama.cpp`;
- baseline build;
- converted GGUF when possible;
- tensor mapping report;
- prompt and decode benchmark;
- system information capture.

Gate:

- another developer can reproduce within expected variance.

## Milestone 1: Checkpoint inspector and loader

Deliver:

- Safetensors parser;
- config parser;
- tensor manifest;
- direct loading;
- text-only tensor selection;
- memory plan;
- no inference yet.

Gate:

- byte-perfect tensor reads against Python;
- complete classification;
- deterministic memory report.

## Milestone 2: CPU/reference operators

Deliver:

- reference quantization;
- reference dequantization;
- RMSNorm;
- RoPE;
- attention;
- MLP;
- softcap;
- tokenizer and chat template;
- layer-level golden fixtures.

Gate:

- golden tests pass.

## Milestone 3: Unfused CUDA correctness engine

Deliver:

- BF16/FP8/NVFP4 basic kernels;
- one-token and short-prefill execution;
- full 48-layer model;
- greedy output;
- BF16 KV mode;
- no performance claim.

Gate:

- logit tolerances pass;
- stable 512-token generation;
- no leaks.

## Milestone 4: Native Blackwell NVFP4

Deliver:

- native block-scaled MMA;
- source-layout handling;
- activation quantization;
- MLP prefill;
- MLP decode;
- instruction-path verification.

Gate:

- no hidden dequant fallback;
- correct output;
- native path faster than reference.

## Milestone 5: Attention specialization

Deliver:

- FP8 projections;
- local attention;
- global attention;
- RoPE fusion;
- FP8 KV cache.

Gate:

- 32K correctness;
- lower memory than BF16 cache;
- faster than unfused attention.

## Milestone 6: Decode fusion and CUDA Graphs

Deliver:

- fused norm/projection paths;
- fused MLP epilogues;
- output-head top-k fusion;
- graph replay;
- no token-loop allocation.

Gate:

- ordinary batch-one decode beats the pinned `llama.cpp` closest-parity baseline or has a documented bottleneck plan.

## Milestone 7: Long context

Deliver:

- 64K stable;
- 128K supported if memory permits;
- chunked prefill;
- long-context benchmarks.

Gate:

- no cache corruption;
- deterministic position semantics;
- quality checks pass.

## Milestone 8: MTP

Deliver:

- assistant loader;
- proposal;
- verification;
- adaptive draft length;
- acceptance reporting.

Gate:

- effective output speed improves on representative workloads;
- quality remains exact under verification;
- memory remains within budget.

## Milestone 9: Multimodal

Only after text goals are met.

---

# 23. Testing

## 23.1 Test categories

- parser unit tests;
- malformed-file tests;
- quantization bit-pattern tests;
- scale conversion tests;
- kernel shape tests;
- randomized operator tests;
- layer golden tests;
- model smoke tests;
- long-generation tests;
- cache wraparound tests;
- graph replay tests;
- out-of-memory tests;
- determinism tests;
- benchmark schema tests.

## 23.2 Small synthetic shapes

Every CUDA kernel needs small tests that run without the full model.

Cover:

- non-multiple dimensions where legal;
- exact tile multiples;
- minimum sizes;
- large representative sizes;
- odd sequence lengths;
- window wrap;
- context boundary;
- NaN and infinity input where applicable;
- zero scales;
- maximum representable E2M1 values;
- subnormal/rounding behavior.

## 23.3 Full-model smoke suite

Use fixed prompts covering:

- plain chat;
- system prompt;
- thinking enabled;
- thinking disabled;
- code;
- multilingual text;
- long repeated context;
- tool-call template;
- stop tokens;
- EOS.

Store token IDs rather than only raw text to isolate tokenizer changes.

## 23.4 Soak tests

Run:

- 4096 generated tokens;
- repeated context resets;
- alternating short and long sessions;
- 100 model loads in a process test where practical;
- graph capture/release cycles;
- cache wraparound.

Track memory after each iteration.

---

# 24. Continuous integration

## 24.1 CPU CI

Every PR:

- formatting;
- static analysis;
- host unit tests;
- parser fuzz corpus;
- Python tool tests;
- generated-file checks;
- benchmark schema validation;
- license checks.

## 24.2 GPU CI

When a 16 GB Blackwell CI runner is available:

- native capability check;
- CUDA kernel unit tests;
- full-model smoke;
- fixed short benchmark;
- correctness metrics;
- VRAM peak;
- SASS instruction check for representative NVFP4 kernel.

Performance CI should alert on regressions, not automatically fail on tiny noisy changes. Define a statistically meaningful threshold.

## 24.3 Sanitizers

Use where applicable:

- AddressSanitizer for host code;
- UndefinedBehaviorSanitizer;
- compute-sanitizer memcheck;
- compute-sanitizer racecheck for relevant kernels;
- initcheck;
- synccheck.

---

# 25. Agent workflow

Every coding agent must:

1. Read this file.
2. Read `docs/ROADMAP.md`.
3. Read `docs/DECISIONS.md`.
4. Inspect the current git status.
5. Identify the narrow task.
6. Locate relevant tests and benchmarks.
7. State assumptions in the commit or PR description.
8. Implement the smallest coherent change.
9. Run required tests.
10. Benchmark when touching a hot path.
11. Update documentation and performance ledger.
12. Report unresolved uncertainty.

Do not start a broad rewrite because a local component is inconvenient.

## 25.1 Before editing a hot kernel

Record:

- current benchmark;
- current profile;
- bottleneck hypothesis;
- expected limiting resource;
- numerical behavior;
- fallback path.

## 25.2 After editing a hot kernel

Record:

- compiler output;
- register count;
- shared memory;
- occupancy estimate;
- spills;
- instruction path;
- benchmark distribution;
- correctness distribution;
- VRAM change.

## 25.3 Commit discipline

Prefer one logical change per commit.

Commit messages:

```text
loader: parse compressed-tensors NVFP4 groups
cuda: add sm120a NVFP4 gate/up decode kernel
attention: fuse local RoPE and KV write
bench: add 64K context decode comparison
```

Do not mix formatting sweeps with kernel changes.

---

# 26. Definition of done

A feature is done only when:

- implemented;
- tested;
- documented;
- errors are actionable;
- memory use is accounted;
- benchmark impact is measured when relevant;
- correctness impact is measured;
- no silent fallback exists;
- build is reproducible;
- source checkpoint revision is pinned;
- code paths are observable through logs or counters.

An optimization is done only when it wins on a representative end-to-end benchmark, not merely a microbenchmark.

---

# 27. Benchmark-result review checklist

Before accepting a claimed speedup, verify:

- same GPU;
- same power/clocks;
- same model source;
- known tensor-format differences;
- same prompt token IDs;
- same output-token count;
- same context;
- same batch;
- same sampling;
- same cache precision or disclosed difference;
- no CPU offload;
- no prompt-cache mismatch;
- adequate warm-up;
- adequate repetitions;
- thermal state stable;
- quality result attached;
- peak VRAM attached;
- raw data retained;
- baseline commit pinned;
- engine commit pinned.

---

# 28. Prohibited shortcuts

Do not:

- copy a `llama.cpp` result from another machine;
- compare against an old release when current upstream is faster;
- use a broken baseline configuration;
- call file-size parity execution-format parity;
- infer NVFP4 execution from a model filename;
- infer native MMA use from high throughput;
- ignore scale tensors;
- use only random data to validate quantized kernels;
- tune exclusively for one prompt length;
- hide model-load repacking;
- keep two persistent weight layouts without reporting memory;
- duplicate tied embeddings;
- materialize full dequantized model weights;
- merge an optimization with no switch to disable it during validation;
- delete reference kernels after fusion;
- weaken tolerances solely to make a test pass.

---

# 29. Documentation requirements

Maintain:

## `README.md`

- what works;
- hardware;
- quick start;
- model download;
- build;
- run;
- known limitations;
- benchmark headline with full context;
- license/provenance.

## `docs/ARCHITECTURE.md`

- model execution graph;
- prefill/decode split;
- kernel map;
- data types;
- graph capture;
- cache design.

## `docs/CHECKPOINT_FORMAT.md`

- exact tensor names;
- packed layout;
- scale relationships;
- source checkpoint revision;
- unsupported variants.

## `docs/BENCHMARKING.md`

- fair-comparison rules;
- commands;
- profiles;
- result interpretation.

## `docs/CORRECTNESS.md`

- references;
- tolerances;
- golden generation;
- known numerical differences.

## `docs/MEMORY.md`

- allocations;
- context formulas;
- peak measurements;
- text-only exclusions.

## `docs/DECISIONS.md`

Use short ADR-style entries:

```text
Date:
Decision:
Context:
Alternatives:
Consequences:
Evidence:
```

---

# 30. Initial task queue

Execute in this order unless blocked.

## Task 1: Repository skeleton

- CMake;
- targets;
- status/error framework;
- logging;
- tests;
- formatting;
- CI;
- docs skeleton.

## Task 2: Model lock and fetch tool

- resolve exact HF revision;
- download required files;
- hash;
- lock;
- verify.

## Task 3: Checkpoint inspector

- parse config;
- parse Safetensors;
- print tensor inventory;
- classify precision;
- total bytes;
- identify modality-only tensors;
- identify tied weights;
- export JSON.

## Task 4: Reference baseline

- run supported reference runtime;
- capture token IDs;
- capture selected hidden states and logits;
- create golden fixtures.

## Task 5: llama.cpp baseline

- pin upstream;
- build for Blackwell SM120/SM120a;
- convert source checkpoint;
- inspect GGUF tensor types;
- verify native NVFP4;
- run quality;
- run baseline benchmark.

If the exact Gemma 4 mixed checkpoint cannot run, document the blocker precisely and maintain the closest valid baseline without falsely calling it parity.

## Task 6: Memory planner

- text-only resident tensor set;
- GPU arena;
- context profiles;
- accounting report.

## Task 7: Reference CUDA operators

- norm;
- FP8;
- NVFP4 unpack/scale;
- GEMM;
- RoPE;
- attention;
- MLP;
- output head.

## Task 8: Full unfused model

- short prompt;
- one token;
- greedy sequence;
- golden comparison.

## Task 9: Native NVFP4 optimization

- verify SASS;
- prefill;
- decode;
- activation quantization;
- MLP fusion.

## Task 10: Attention and cache

- local;
- global;
- FP8 KV;
- 32K/64K.

## Task 11: CUDA Graphs and fusion

- fixed decode plan;
- output sampling;
- benchmark.

## Task 12: MTP

Only after ordinary decode is competitive.

---

# 31. Suggested command interface

## Inspect

```bash
gem16-inspect \
  --model /models/gemma-4-12b-it-NVFP4 \
  --json manifest.json \
  --validate
```

## Run

```bash
gem16-run \
  --model /models/gemma-4-12b-it-NVFP4 \
  --context-profile standard \
  --prompt "Write a CUDA reduction kernel." \
  --max-tokens 512 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 64
```

## Deterministic correctness

```bash
gem16-run \
  --model /models/gemma-4-12b-it-NVFP4 \
  --input-token-ids tests/golden/prompt_001.tokens \
  --greedy \
  --max-tokens 64 \
  --dump-logits logits.bin \
  --deterministic
```

## Benchmark

```bash
gem16-bench end-to-end \
  --model /models/gemma-4-12b-it-NVFP4 \
  --suite benchmarks/prompts/core.json \
  --context-profile standard \
  --warmup 3 \
  --repetitions 10 \
  --output benchmarks/results/run.json
```

## Capabilities

```bash
gem16-run --print-kernel-capabilities
```

---

# 32. Logging and observability

At model load, log:

- checkpoint revision;
- tensor counts by type;
- bytes by type;
- text-only skipped bytes;
- direct vs transformed layout;
- native kernels selected;
- persistent allocations;
- context profile;
- KV-cache bytes;
- CUDA Graph status.

At benchmark end, log:

- actual native kernel counts;
- fallback counts;
- tokens processed;
- accepted MTP tokens;
- peak VRAM;
- timing;
- clocks;
- power.

Provide `--trace-kernels` for development, but keep it disabled in performance runs.

---

# 33. Security and robustness

Treat model files as untrusted input.

Protect against:

- malicious JSON sizes;
- integer overflow;
- out-of-bounds offsets;
- overlapping tensor ranges;
- path traversal in shard indexes;
- unsupported dtype spoofing;
- tensor-size mismatch;
- excessive allocation requests;
- tokenizer malformed data;
- invalid UTF-8 where prohibited.

Never execute code from a model repository.

Do not use `trust_remote_code` in the runtime.

---

# 34. Licensing and provenance

Keep code, model, and dependency licensing separate.

For copied or adapted code:

- verify license compatibility;
- preserve notices;
- cite source file and commit;
- describe modifications.

`llama.cpp` may be studied and benchmarked. Any copied implementation must retain required attribution and must be clearly identified.

Model users remain responsible for the checkpoint’s license and terms.

---

# 35. External facts and references

These references motivated the initial design. Pin revisions where possible and revalidate if the project begins later.

- Gemma 4 12B model card:  
  https://huggingface.co/google/gemma-4-12B

- Initial direct-load checkpoint:  
  https://huggingface.co/unsloth/gemma-4-12b-it-NVFP4

- Initial checkpoint config:  
  https://huggingface.co/unsloth/gemma-4-12b-it-NVFP4/blob/main/config.json

- Initial checkpoint recipe:  
  https://huggingface.co/unsloth/gemma-4-12b-it-NVFP4/blob/main/recipe.yaml

- NVIDIA CUDA compute-capability table:  
  https://developer.nvidia.com/cuda/gpus

- NVIDIA NVFP4 documentation:  
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html

- CUTLASS Blackwell block-scaled data types:  
  https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html

- llama.cpp native Blackwell NVFP4 PR:  
  https://github.com/ggml-org/llama.cpp/pull/22196

- llama.cpp repository:  
  https://github.com/ggml-org/llama.cpp

---

# 36. Final directive to agents

Build the smallest correct engine that can execute the pinned model, then optimize only what profiling identifies.

The expected path to beating `llama.cpp` is not merely implementing NVFP4 matrix multiplication. `llama.cpp` already has a native Blackwell NVFP4 path. The advantage must come from specialization:

- direct mixed-checkpoint loading;
- Gemma 4-specific execution;
- batch-one-specific kernels;
- separate prefill and decode plans;
- fewer launches;
- fused norm and quantization;
- fused MLP stages;
- local/global attention specialization;
- exact fixed-window cache handling;
- tied-output-head specialization;
- GPU sampling;
- CUDA Graph replay;
- later, native MTP integration.

Every claimed improvement must survive correctness tests and a fair, pinned, reproducible benchmark against current upstream `llama.cpp`.
