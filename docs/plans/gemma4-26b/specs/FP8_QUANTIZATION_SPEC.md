# FP8 attention quantization specification

## Role

FP8 stores attention projection weights for Q, K, V and O. Runtime input activations are quantized dynamically per token. Accumulation is FP32, followed by model-required scaling and BF16 boundary behavior.

## Weight representation

For a logical matrix:

```text
W: [N output rows, K contracting elements]
```

store:

```text
weight:       F8_E4M3 [N, K]
weight_scale: BF16    [N, 1]
```

Mathematical reconstruction:

```text
W_real[n, k] = decode_e4m3(weight[n, k]) * decode_bf16(weight_scale[n])
```

The scale is per output row/channel.

## Activation representation

For each token row `m`:

```text
input_scale[m] = max(abs(A_real[m, :])) / max_finite_e4m3
A_q[m, k] = round_e4m3fn(A_real[m, k] / input_scale[m])
```

All-zero input uses `input_scale = 1.0`.

Projection:

```text
Y[m, n] =
  sum_k decode(A_q[m,k]) * decode(W_q[n,k])
  * input_scale[m]
  * weight_scale[n]
```

Accumulation is FP32. The precise order of multiplying scales and casting to BF16 is an execution contract and must match the accepted runtime path.

## Weight scale selection

Version 1 uses a deterministic rowwise max-absolute scale unless M05 proves and documents another published-compatible recipe.

Requirements:

- finite positive scale for nonzero row;
- `1.0` or another explicitly specified scale for all-zero row;
- no NaN/Inf payload;
- explicit E4M3FN rounding;
- explicit BF16 scale rounding.

If a scale search minimizes reconstruction error, the objective, candidate set and tie rule must be versioned.

## Rounding

Reference encoder requirements:

- round to nearest, ties to even;
- deterministic finite saturation;
- explicit behavior for NaN/Inf source values: reject source tensor;
- preserve or canonicalize signed zero according to the locked codec;
- little-endian payload.

Do not delegate the normative encoder to a GPU intrinsic or unspecified language cast.

## Missing V projection

For global attention layers with `attention_k_eq_v` and no stored V projection:

- do not synthesize a V weight;
- bind raw K projection as the shared projection result;
- later K and V normalization/RoPE paths remain distinct.

## Error statistics

Per matrix report:

- min/max/RMS source;
- min/max scale;
- relative L2 reconstruction error;
- cosine;
- maximum absolute error;
- SQNR;
- saturation count;
- zero rows;
- E4M3 code histogram.

Operator report with real activations:

- output NRMSE;
- output cosine;
- max absolute error;
- BF16 boundary mismatch count;
- downstream Q/K norm and RoPE drift where applicable.

## Compiler/runtime separation

The compiler writes canonical row-major FP8 payload and row scales. Runtime may transform only storage order if needed; it may not requantize weights.

Dynamic activation quantization remains runtime work because activation scale is token-dependent.

## Native path evidence

The release path must prove:

- selected CUDA objects contain the intended FP8 Tensor Core instruction family;
- actual kernels dispatch for real 26B shapes;
- no fallback;
- scales and BF16 boundaries are applied exactly once.

## Tests

- exhaustive codec boundaries;
- row scale fixtures;
- all-zero rows;
- synthetic matrix round trip;
- ordinary source versus Unsloth comparison;
- real local/global projection shapes;
- T=1 and batch prefill;
- disassembly and runtime dispatch.
