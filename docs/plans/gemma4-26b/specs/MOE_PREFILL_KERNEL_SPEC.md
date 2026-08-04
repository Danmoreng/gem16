# Grouped MoE prefill kernel specification

## Objective

Process many prompt tokens efficiently while bounding memory independently of full context length.

## Chunking

Prefill operates on chunks selected by the memory planner. Candidate sizes may include:

```text
128, 256, 512, 1024
```

The final size is measured. It is not inherited from 12B.

Workspace scales with chunk size, not total prompt length.

## Routing records

For `T` tokens and top-k 8:

```cpp
struct Assignment {
  uint16_t expert_id;
  uint16_t topk_slot;
  uint32_t token_id;
  float weight;
};
```

Exactly `8T` records.

## Pipeline

```text
router(T)
→ assignments(8T)
→ histogram(128)
→ exclusive prefix sum(129)
→ stable permutation(8T)
→ grouped W13
→ activation + W2 input quantization
→ grouped W2
→ inverse permutation
→ deterministic weighted reduction per token
→ combine shared branch
```

## Stable permutation

Within each expert, preserve a deterministic order:

```text
token index ascending, then top-k slot ascending
```

This supports deterministic reduction and reproducible diagnostics.

## Expert scheduling

Experts with zero assignments are skipped without host intervention. Heavily used experts may be split into multiple CTA tiles.

Track active expert count and assignment skew as telemetry.

## W13/W2 buffers

Avoid:

```text
T × 8 × full hidden
T × 128 × intermediate
```

Permitted bounded buffers:

- assignments `8T`;
- permuted expert inputs `8T × H` only if memory model accepts it;
- preferably indices plus reused token input;
- W13 product `8T × I_expert`;
- W2 partial output may be tiled/reduced rather than materialized in full;
- shared branch `T × I_shared`.

M15 must choose the smallest performant representation.

## Weight layout

Use the one resident decode layout if possible. If CUTLASS prefill requires another order:

- transform one projection or bounded tile into reusable scratch;
- use scratch immediately;
- overwrite for next projection;
- no persistent second copy;
- report scratch bytes.

## Reduction

Atomic `index_add` is not automatically acceptable. Options:

1. stable segmented reduction in assignment order;
2. one token-owned CTA gathering its eight contributions;
3. deterministic two-stage reduction.

The selected method must preserve deterministic output and accepted rounding.

## Shared branch

Run dense shared MLP for all T tokens. It can use standard batch NVFP4 GEMMs and may have different optimal geometry from routed experts.

## Small prompt crossover

Grouped sorting can lose for very small T. Define measured crossover:

```text
T < threshold → repeated native decode-like path
T ≥ threshold → grouped prefill
```

The selector is fixed at plan creation or chunk dispatch and reported.

## Memory telemetry

Report:

- chunk size;
- assignments;
- histogram/prefix;
- permutation;
- activation payload/scales;
- product;
- W2 partial/reduction;
- shared branch;
- CUTLASS workspace;
- total reusable bytes.

## Tests

- all experts zero except one;
- every expert hit;
- highly skewed load;
- partial chunk;
- stable permutation;
- no duplicate/lost assignment;
- deterministic reduction;
- reference output;
- 32K/64K workspace invariant;
- sanitizer and performance matrix.
