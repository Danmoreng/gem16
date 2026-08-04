# imp reference assessment

## Reference identity

```text
repository: https://github.com/kekzl/imp
commit: a392904d4216388828d0d56317de046f4ca49627
license: MIT
reference date: 2026-08-04
```

At the pinned revision, imp is a from-scratch C++/CUDA engine aimed primarily at RTX 5090 / `sm_120a`, with a `compute_120f` PTX fallback for other consumer Blackwell GPUs. It supports a much broader runtime surface than gem16: GGUF and SafeTensors, multiple model families, paged KV, continuous batching and several quantization formats.

## Why it is valuable

### Gemma 4 MoE correctness history

The source contains concrete fixes and diagnostics around:

- scale-free router RMSNorm rather than reusing the shared-MLP norm;
- the additional `1/sqrt(hidden_size)` factor;
- FP32 router inputs and logits for late-layer expert-selection stability;
- top-k renormalization and per-expert output scales;
- separate shared-MLP and routed-expert norm paths;
- FP32 residual diagnostics and exact post-FFN ordering.

These are high-value differential checks for M10/M11. The official Transformers implementation remains the normative semantic source; imp is a second implementation that exposes failure modes and precision-sensitive boundaries.

### Consumer-Blackwell NVFP4 work

The pinned tree includes:

- direct `mma.sync ... mxf4nvf4 ... m16n8k64` paths;
- grouped small-M NVFP4 GEMM with M tiles 16/32/64/128;
- native UE4M3 scale handling;
- persistent work-queue concepts;
- CUTLASS and hand-written paths;
- actual-path dispatch recording.

This makes imp a useful source of candidate kernel schedules and negative results. It does not prove those schedules are optimal for a 16 GB RTX 5080 or for gem16's final Row8/K64 layout.

### Quality evidence

Imp's own Gemma 4 quality audit reports that its ModelOpt NVFP4 SafeTensors checkpoint was materially worse than a UD-Q4_K_M checkpoint on a plain-prose corpus, while decode speed was close. Its follow-up attributes most of the gap to the NVFP4-quantized expert weights/checkpoint recipe rather than to one imp execution path.

This evidence strengthens, rather than weakens, the plan to:

- use Google QAT BF16 as the single master source;
- compile NVFP4 ourselves;
- compare ordinary-BF16 and QAT-BF16 through the exact same compiler;
- retain official Q4_0 and Unsloth as external references;
- add ModelOpt NVFP4 as a negative/control arm.

It does **not** prove that QAT-BF16→NVFP4 will be better. That remains an experiment.

### Engineering process

Useful process patterns include:

- machine-readable performance and VRAM baselines;
- actual-path dispatch records;
- central graph-demotion reasons;
- a settled-priors ledger with source anchors;
- mutation-validated tests;
- explicit disclosure of noisy or non-comparable measurements.

## Local 16 GB characterization

The pinned source was built on the reference RTX 5080 host with GCC 15.3 and CUDA 13.3 and tested against the
official 26B A4B QAT Q4_0 GGUF. The model generates coherent text, but imp offloads 3.59 GiB of experts, disables
graphs, and reaches only 1,533.75 pp512 tok/s and 51.64 tg256 tok/s. An adjacent fully resident llama.cpp b10240
run reaches 5,087.77 and 169.762 tok/s. This rejects imp's Q4_0 host-offload route as a product baseline and
confirms that broad runtime adoption is inappropriate.

The result is not a native-NVFP4 kernel comparison: imp reports `legacy_fallback` for MoE prefill, and its published
NVFP4 configuration targets a 32 GB RTX 5090 where all experts fit. Keep the isolated native-kernel study in M14/M15.

## Limits of the reference

- Published headline figures are on RTX 5090, not the target RTX 5080 16 GB card.
- `compute_120f` compatibility is not equivalent to a directly tuned/tested 5080 path.
- imp's broad dispatch/cache architecture has different memory and serving goals.
- its GGUF and SafeTensors paths may use different weight layouts and sidecars.
- its quality findings depend on exact checkpoints, corpora and runtime options.
- fast-moving `main` must never be used as an unpinned oracle.

## Net plan impact

**Architecture:** unchanged.
**Reference set:** expanded.
**Quality risk:** raised.
**Implementation order:** stricter.
**Telemetry and provenance:** expanded.
**Potential kernel reuse:** conditional and isolated.
