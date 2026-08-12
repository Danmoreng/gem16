# NVFP4 quantization specification

## Role

NVFP4 is implemented as a future native extension of the shared converter architecture in
[`NATIVE_CONVERTER_ARCHITECTURE.md`](NATIVE_CONVERTER_ARCHITECTURE.md). Python may provide small reference fixtures
only; it is not a promoted NVFP4 conversion path.

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

M06 must freeze one versioned algorithm before its full QAT conversion. The compiler configuration records:

1. tensor-global divisor derivation;
2. local scale derivation and search objective;
3. E4M3FN and E2M1 rounding/saturation;
4. zero-block and tie behavior;
5. exact deterministic sampled Ordinary/Unsloth tensors and diagnostic acceptance conditions.

The sample must cover shared and routed Gate, Up and Down roles plus fused-axis handling across local/global layer classes. It is a convention check, not full causal attribution: scale direction, tensor relationships, shapes and finite operator reconstruction must pass the frozen conditions. A failed check blocks the QAT full run. Byte identity with Unsloth is desirable but not required. Complete Ordinary conversion remains conditional M18 work.

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

The native C++ backend must be the normative implementation and must emit coupled weight/scales/divisors from one
bounded source traversal. Test the same backend with explicit thread counts and never silently fall back to Python.

- exhaustive E2M1/E4M3 codecs;
- divisor convention;
- zero and outlier blocks;
- fused gate/up axis mapping;
- all experts/layers manifest;
- CPU/CUDA/native projection;
- sampled Ordinary-versus-Unsloth diagnostics for M06 convention checks;
- complete Ordinary/QAT attribution only when conditional M18 is triggered;
- deterministic artifact bytes.

## llama.cpp reference boundary

The local llama.cpp research is useful for native E2M1 packing, 16-element grouping, UE4M3 codec tests and threaded
block work, but it is not this contract. Its GGML NVFP4 layout combines four groups into a 64-element block, applies a
UE4M3/2 convention with a doubled FP4 lookup, and its CPU reference uses producer-specific scale behavior including an
`amax/6` policy. Its selective `--tensor-type nvfp4` path does not establish Gem16's complete mixed artifact or
activation/global-scale semantics. Do not copy its bytes, scale direction or tie rules without a versioned differential
test and provenance decision. See [`NATIVE_CONVERTER_ARCHITECTURE.md`](NATIVE_CONVERTER_ARCHITECTURE.md) and the
version-scoped llama research evidence linked there.

## Producer-specific global-scale semantics

“NVFP4” is not a complete reconstruction contract. The manifest and compiler must distinguish at least:

```text
llm-compressor divisor: real = fp4 * local_scale / global_scale
ModelOpt multiplier:    real = fp4 * local_scale * tensor_scale
```

Never infer the direction from a tensor suffix alone. Require producer metadata or a locked project schema, and fail visibly on ambiguity.
