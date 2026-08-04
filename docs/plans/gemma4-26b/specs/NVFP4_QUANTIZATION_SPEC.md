# NVFP4 quantization specification

## Role

NVFP4 is the production format for:

- routed expert gate/up/down weights;
- always-active shared dense MLP;
- optionally the tied embedding/output head.

Its primary reason is native Blackwell block-scaled Tensor Core execution combined with approximately 4.5 bits per weight including local scales.

## Logical representation

For matrix `W[rows, K]`, with `K % 16 == 0`:

- values: signed E2M1, 4 bits each;
- two values per U8;
- one E4M3FN local scale per 16 consecutive K values;
- one F32 tensor-global weight divisor or scale;
- one F32 input-global divisor used by dynamic activation quantization.

The current compressed-tensors interpretation in gem16 uses stored global values as divisors:

```text
W_real = W_e2m1 * local_scale_e4m3 / weight_global_divisor
A_scaled = A_real * input_global_divisor
A_scaled ≈ A_e2m1 * activation_local_scale_e4m3
Y = sum(A_e2m1 * A_local * W_e2m1 * W_local)
    / (input_global_divisor * weight_global_divisor)
```

M06 must verify this exact convention against the pinned source and trusted runtime. If the compiled artifact uses multipliers instead, it must use a different versioned profile.

## Block layout

Canonical logical grouping:

```text
for each row:
  for k_block in range(K / 16):
    16 source values
    → 16 E2M1 codes
    → 8 packed bytes
    → 1 E4M3FN local scale
```

Nibble order must be locked by fixtures. Runtime Row8/K64 order changes only placement, not codes/scales.

## Scale selection

Version the scale algorithm. It must define:

1. tensor-global divisor selection;
2. local scale selection;
3. E4M3FN rounding;
4. E2M1 rounding/saturation;
5. zero block behavior;
6. tie behavior;
7. whether scale search minimizes max error, L2 error or another objective.

The ordinary BF16 → own NVFP4 versus Unsloth comparison determines whether the project can reproduce the published recipe. Byte identity is desirable but not assumed.

## Activation quantization

Activations are W4A4, not weight-only FP4.

For every token and every 16-value K group:

- apply tensor input divisor;
- choose local E4M3FN scale;
- encode E2M1 values;
- store packed values and scales in fixed preallocated buffers.

All-zero group behavior must be explicit and deterministic.

## Expert tensors

Actual source may store expert gate/up fused:

```text
[experts, 2 * intermediate, hidden]
```

and down:

```text
[experts, hidden, intermediate]
```

The compiler and runtime must preserve expert axis, gate/up split axis and row order. Shape validation must use M03 evidence.

## Shared dense MLP

The always-active branch uses ordinary gate/up/down matrices with intermediate 2112. It is not routed and must execute every layer.

## Error and code reports

Per tensor:

- source/reconstructed statistics;
- relative L2, cosine, max error, SQNR;
- local/global scale distributions;
- E2M1 code histogram;
- saturation count;
- zero blocks;
- invalid E4M3 encodings;
- expert-by-expert summary.

Per operator:

- output NRMSE/cosine/max;
- activation code/scales match;
- gate/up and post-activation drift;
- down output and residual drift.

## Runtime storage

The device keeps one final weight layout. Preferred 26B layout must support:

- direct T=1 expert access;
- bounded-scratch prefill conversion or direct grouped prefill;
- no expert gather;
- no second persistent layout.

`persistent_repack_bytes` must remain zero.

## Native instruction proof

For release:

- disassembly contains the intended NVFP4 block-scaled MMA;
- real kernels dispatch on SM120/SM120a;
- scale vector size and layout match the instruction;
- no SIMT/reference fallback;
- no local-memory spills that invalidate expected performance.

## Quality caveat

Google's QAT master was trained for Q4_0 behavior, not necessarily NVFP4 W4A4. A QAT-derived NVFP4 artifact is an empirical hypothesis. Do not call it “NVFP4 QAT” unless it has actually undergone NVFP4-aware training.

## Tests

- exhaustive E2M1/E4M3 codecs;
- divisor convention;
- zero and outlier blocks;
- fused gate/up axis mapping;
- all experts/layers manifest;
- CPU/CUDA/native projection;
- ordinary versus Unsloth;
- QAT versus ordinary activation drift;
- deterministic artifact bytes.

## Producer-specific global-scale semantics

“NVFP4” is not a complete reconstruction contract. The manifest and compiler must distinguish at least:

```text
llm-compressor divisor: real = fp4 * local_scale / global_scale
ModelOpt multiplier:    real = fp4 * local_scale * tensor_scale
```

Never infer the direction from a tensor suffix alone. Require producer metadata or a locked project schema, and fail visibly on ambiguity.
