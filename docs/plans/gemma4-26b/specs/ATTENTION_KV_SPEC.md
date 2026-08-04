# Gemma 4 26B attention and KV specification

## Expected traits

Validated target:

| Property | Value |
|---|---:|
| Layers | 30 |
| Sliding layers | 25 |
| Full layers | 5 |
| Query heads | 16 |
| Local KV heads | 8 |
| Global KV heads | 2 |
| Local head dimension | 256 |
| Global head dimension | 512 |
| Sliding window | 1024 |
| Maximum positions | 262144 |

The exact per-layer table comes from config validation.

## Projection formats

- Q/K/V/O weights: FP8 per output channel;
- input activations: dynamic FP8 per token;
- accumulation: FP32;
- output boundaries: accepted BF16 behavior;
- global V can reuse raw K projection when no V tensor exists.

## Local attention

- D256;
- 8 KV heads, 16 Q heads;
- 2 query heads per KV head;
- 1,024-position circular cache;
- separate physical K and V;
- chronological reads across wrap;
- causal/local mask.

## Global attention

- D512;
- 2 KV heads, 16 Q heads;
- 8 query heads per KV head;
- full configured context extent;
- separate physical K and V;
- proportional/partial RoPE according to config;
- causal mask.

## K equals V semantics

`attention_k_eq_v` permits projection reuse only.

```text
raw = K_projection(x)
K = learned_K_norm(raw) + RoPE
V = scale_free_V_norm(raw), no RoPE
```

Therefore:

- raw projection can be shared;
- final states cannot alias;
- KV payload remains K+V.

## Cross-layer KV sharing

If `num_kv_shared_layers` is nonzero:

- identify producer layers by validated reference semantics;
- consumers omit K/V matrices and cache writes;
- producer states may need full-length retention even for later sliding consumers;
- ownership is explicit in `LayerTraits`;
- memory formulas count physical producer caches, not naive layer count.

The provisional memory arithmetic assumes the validated expected ownership. M12 must update formulas if source config differs.

## FP8 cache

Store one byte per value plus fixed per-layer scale metadata.

Local separate K+V after ring fills:

```text
25 × 1024 × 8 × 256 × 1 × 2 = 104,857,600 bytes = 100 MiB
```

Global separate K+V:

```text
5 × context × 2 × 512 × 1 × 2
```

If cross-layer sharing reduces physical producers, record both logical and physical counts.

## RoPE

Parse rope parameters from config by attention type. Do not hard-code only:

```text
local theta 10,000
global theta 1,000,000
partial factor 0.25
```

unless validation confirms them.

Precompute exact tables only when memory and accepted arithmetic permit. Table storage belongs in the arena report.

## Cache commit

For ordinary forward:

1. project and normalize current K/V;
2. attention reads prior state plus staged current state;
3. commit current state after attention;
4. advance position.

For tentative/MTP future paths, use transactional rows and commit only accepted state.

## Tests

- layer table;
- projection shapes;
- missing V;
- K/V distinctness;
- local wrap;
- global boundaries;
- RoPE positions;
- shared-KV ownership;
- FP8 encode/decode;
- prompt chunk boundaries;
- long-context retrieval;
- exact byte accounting.
