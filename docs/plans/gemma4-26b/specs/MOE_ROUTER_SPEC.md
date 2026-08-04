# MoE router specification

## Input transform

For each token hidden vector `x[H]`:

```text
r = rms_norm_without_learned_weight(x)
r = r * learned_hidden_scale
r = r * H^(-0.5)
logits = W_router @ r
probabilities = softmax_fp32(logits)
```

The exact order of the learned scale and scalar factor is locked by the trusted reference. `H = 2816` for the expected 26B profile.

## Parameters

Expected router tensors include:

- projection `[128, 2816]`;
- learned hidden scale `[2816]`;
- per-expert scale `[128]`;
- any epsilon/scalar metadata.

Names and dtypes come from M03.

Router projection remains BF16 in the first production profile because routing is discontinuous and the memory cost is small.

## Softmax

Requirements:

- logits converted/accumulated in specified precision;
- max subtraction;
- exponent/sum in FP32;
- no fast approximation unless separately proven;
- NaN/Inf detection;
- deterministic reduction strategy.

## Top-8

Return:

```text
ids[8]
weights[8]
```

Procedure:

1. select eight largest probabilities;
2. preserve accepted tie behavior;
3. normalize selected values so their sum is one;
4. multiply each by `per_expert_scale[id]`.

Do not renormalize after applying per-expert scale unless the reference does.

## Tie behavior

Top-k ties are rare but must be deterministic. The plan should prefer lower expert ID on equal probability unless the pinned trusted runtime demonstrates another stable policy.

Tests must include exact equal logits.

## Output layout

Decode:

```cpp
struct RoutedExperts {
  std::uint16_t ids[8];    // or uint32 if simpler
  float weights[8];
};
```

Prefill:

- one record per token/top-k slot;
- stable token and slot identity retained through sorting;
- no host readback.

## GPU design

Decode router may be one or more kernels:

- norm/scale/projection;
- softmax/top-k;
- selected normalization.

Fusion is allowed only after the staged reference is exact and faster.

With 128 experts, a single block or bounded CTA set can hold logits. Profile occupancy, shared memory and reduction cost.

## Determinism

Avoid:

- unordered atomic winner selection;
- approximate top-k with unstable ties;
- host library sort;
- architecture-dependent uninitialized padding.

Repeated identical inputs must produce identical IDs and weight bits in deterministic mode.

## Drift metrics

Compare candidates using:

- full probability KL;
- maximum probability error;
- top-8 set Jaccard;
- ordered top-8 agreement;
- first differing rank;
- selected weight L1/L2;
- downstream weighted expert-output drift.

A changed eighth expert can be more significant than small aggregate probability error.

## Error handling

Reject or stop diagnostic execution on:

- nonfinite logits/probabilities;
- fewer than 8 experts;
- duplicate selected IDs;
- nonpositive selected sum;
- invalid per-expert scale;
- output ID out of range.

Production release should make impossible states visible rather than silently repairing them.

## imp-derived precision gate

The official model contract remains normative, but the pinned imp implementation documents a concrete failure mode when the router path is truncated to FP16. M10/M11 must therefore retain FP32 router input/logit fixtures at late layers and near top-8 boundaries. The production path may use a lower precision only after proving expert-ID and quality equivalence.
