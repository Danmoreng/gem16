# Trusted reference runtime and golden-capture specification

## Purpose

The 26B implementation needs independent evidence at three levels:

1. source BF16 semantics;
2. official Q4_0 behavior;
3. published Unsloth NVFP4 behavior.

No single runtime is trusted for every arithmetic format.

## Reference roles

| Reference | Role |
|---|---|
| Pinned Transformers implementation with Google BF16/QAT BF16 | architecture and high-precision semantic reference |
| llama.cpp with official Q4_0 GGUF | official QAT-target-format reference |
| vLLM or another verified direct compressed-tensors runtime with Unsloth NVFP4 | practical mixed FP8/NVFP4 reference |
| gem16 CPU/CUDA reference paths | independent implementation and differential oracle |

## Version locking

Every capture records:

- reference code commit/package version;
- model lock hash;
- tokenizer/template hashes;
- CUDA/driver/PyTorch versions;
- device;
- dtype and cache format;
- attention implementation;
- seed and determinism settings;
- exact input token IDs;
- output file hash.

Do not use a floating package install.

## Required captures

### Configuration and tensor inventory

- parsed model configuration;
- layer traits;
- tensor names/shapes/dtypes;
- quantization groups;
- tied/omitted tensors.

### Tokenization

- rendered prompts;
- token IDs;
- special/EOS/suppression IDs;
- thinking on/off;
- tool-template fixtures if product scope includes tools.

### Layer boundaries

For selected prompts and positions:

- input embedding;
- post-input-norm;
- Q/K/V projection outputs;
- normalized/rotated Q/K and normalized V;
- attention output and post-attention residual;
- shared MLP gate/up/product/down;
- router pre-norm, logits, probabilities, top-8 IDs/weights;
- selected expert gate/up/product/down;
- combined feed-forward branch;
- final layer output;
- final norm and logits.

Capture at least local and global layers and first/middle/final MoE layers.

### Full model

- full-vocabulary logits for selected teacher-forced positions;
- top-20 logits for broader positions;
- greedy token streams;
- sampled streams for fixed seeds;
- router summaries across prompts;
- loss/NLL where available.

## Capture precision

Store raw little-endian arrays with metadata, not decimal text alone.

Suggested:

```text
golden.json
arrays/
  layer_00_position_0031_router_logits.f32
  layer_00_position_0031_top8_ids.u32
  layer_00_position_0031_top8_weights.f32
  ...
SHA256SUMS
```

Compressing raw arrays is acceptable only with deterministic settings and checksums.

## Input corpus

Maintain disjoint sets:

- synthetic operator fixtures;
- compiler calibration, if needed;
- development diagnostics;
- held-out quality test;
- performance prompts.

Every manifest contains exact token IDs and content license/source.

## Cross-runtime caveat

Exact tokens can diverge from small logit differences. Therefore:

- exact identity is required within the same runtime/config across repetitions;
- exact identity is required for intentionally bit-compatible reference operators;
- cross-runtime acceptance uses logits, rank, KL, router and task metrics unless an exact-parity claim is made.

Never lower operator correctness requirements merely because full generations can diverge.

## Failure localization

When full-model drift exceeds threshold, capture the first layer/position where:

- residual NRMSE crosses threshold;
- router top-8 set changes;
- attention output diverges;
- selected-token rank changes.

Reports must distinguish source-weight difference, quantization difference, kernel arithmetic difference and autoregressive sequence drift.

## Reference execution safety

High-precision 26B may require a larger GPU or CPU/RAM. Such execution is allowed for offline goldens, not for the 16 GB product benchmark. Record the hardware and do not compare its speed.

## Golden update policy

Goldens change only when:

- source lock changes;
- trusted runtime changes;
- capture bug is fixed;
- model contract intentionally changes.

Every update requires an explanation and retains prior goldens when useful for history.
