# Memory budget and residency

## Reference model dimensions

```text
layers                         30
local attention layers         25
global attention layers         5
hidden size                  2816
shared dense intermediate    2112
routed expert intermediate    704
routed experts                128
active experts                  8
query heads                    16
local KV heads                  8
global KV heads                 2
local/global head dim       256/512
vocabulary                 262144
```

## Resident weight estimate

### Routed experts

Each expert has gate, up and down matrices:

```text
30 × 128 × 3 × 2816 × 704
= 22,837,985,280 weights
```

NVFP4 storage at 8 packed bytes plus one E4M3 scale byte per 16 values:

```text
0.5625 byte/weight
= 12,846,366,720 bytes
= 12,251.25 MiB
= 11.9641 GiB
```

### Always-active shared/dense MLP

```text
30 × 3 × 2816 × 2112 × 0.5625
= 301,086,720 bytes
= 287.14 MiB
```

### Attention FP8

Accounting for 25 local layers with Q/K/V/O and five global layers with Q/K/O under `attention_k_eq_v=true`:

```text
1,110,179,840 FP8 weight bytes
+ approximately 0.64 MiB BF16 channel scales
= approximately 1,059.39 MiB
```

### Tied embedding/output head

```text
262144 × 2816 = 738,197,504 weights
```

- BF16: 1,407.99 MiB
- Q4_0 or NVFP4 at 0.5625 byte/weight: 396.00 MiB

The matrix must be tied physically, not merely logically.

### Router projection

```text
30 × 128 × 2816 × 2 BF16 bytes
= 20.625 MiB
```

Router scale, per-expert scale, norms, layer scalars, final norm, quantizer global scales, metadata and 256-byte alignment add only a small fraction compared with experts, but must still be counted exactly.

### Planning total

Before small tensors and alignment:

```text
14,695,166,976 bytes
= 14,014.40 MiB
= 13.68594 GiB
```

Planning bands:

| Band | Immutable arena |
|---|---:|
| Stretch target | ≤ 13,950 MiB |
| Primary target | ≤ 14,100 MiB |
| Warning | 14,101–14,300 MiB |
| Hard stop and redesign | > 14,300 MiB |

The compiler and loader must report both decimal GB and binary MiB/GiB to avoid ambiguous “15.6 GB” claims.

## FP8 KV payload

Separate physical K and V are required.

### Local layers after window fill

```text
25 × 1024 × 8 × 256 × 1 byte × 2
= 100 MiB
```

### Global layers

```text
5 × context × 2 × 512 × 1 byte × 2
```

| Context | Local K/V | Global K/V | Total payload |
|---:|---:|---:|---:|
| 8K | 100 MiB | 80 MiB | 180 MiB |
| 16K | 100 MiB | 160 MiB | 260 MiB |
| 32K | 100 MiB | 320 MiB | 420 MiB |
| 64K | 100 MiB | 640 MiB | 740 MiB |
| 128K | 100 MiB | 1,280 MiB | 1,380 MiB |
| 256K | 100 MiB | 2,560 MiB | 2,660 MiB |

These are payloads, not process peaks.

## Reference-device capacity

The RTX 5080 Laptop reports approximately 16,303 MiB as nominal board memory, but the current CUDA runtime exposes
only approximately **15,881 MiB** through allocation/free-memory accounting. Budgets and admission gates must use
the lower runtime-visible capacity and direct `cudaMemGetInfo` measurements, not nominal board memory.

The current CUDA context alone accounts for roughly 216 MiB before model-specific allocations. Re-measure this
value after driver or toolkit changes.

## Required 32K budget

At the maximum immutable target:

```text
CUDA-visible capacity          15,881 MiB
immutable weights target      -14,100 MiB max
32K K/V payload                  -420 MiB
remaining before context,
workspace, graphs and margin     1,361 MiB
```

At the current 14,014.40 MiB planning estimate, 1,446.60 MiB remains before context and execution allocations.
After a representative 216 MiB context and the required 700 MiB margin, only approximately 445 MiB at the maximum
weight target—or 531 MiB at the planning estimate—remains for all workspace, graph, allocator and metadata costs.

Required production gate:

- `cudaMemGetInfo` free margin at least 700 MiB after slot initialization and warm generation;
- sampled process telemetry is retained, but nominal-total-minus-process-peak is not used as a substitute for the
  direct free-memory gate;
- on the current runtime, approximately 15,181 MiB used is the largest compatible observed value before a 700 MiB
  free margin, subject to reconciliation with context/driver accounting;
- allocator report reconciles weights, KV, reusable workspace and graph-private bytes;
- no transient startup device peak above safe capacity.

M03 must run a preliminary synthetic reservation/admission probe using the frozen role inventory and conservative
workspace estimates before compiler implementation proceeds. M09 repeats the gate with the real compiled artifact
and final named arenas.

## Target 64K budget

```text
CUDA-visible capacity          15,881 MiB
immutable weights target      -14,100 MiB max
64K K/V payload                  -740 MiB
remaining before context,
workspace, graphs and margin     1,041 MiB
```

At the 14,014.40 MiB planning estimate, 1,126.60 MiB remains before context and execution allocations.

Target gate:

- at least 500 MiB directly measured free margin;
- prompt chunking selected so prefill workspace remains bounded;
- no full prompt-sized expert activation or permutation buffer beyond documented compact routing arrays.

64K is a high-risk target, not permission to reduce the 32K safety margin.

## Workspace policy

### Forbidden

- `[tokens, 128 experts, ...]` activation materialization;
- dequantized expert copies;
- all-expert output tensors;
- persistent duplicate source/runtime layouts;
- separate resident Q4_0 and NVFP4 heads;
- full vocabulary logits retained across steps;
- unbounded CUB/temp-storage query allocation during execution;
- per-layer `cudaMalloc`.

### Allowed

- one fixed prompt chunk;
- compact 8 assignments per token;
- expert histogram and prefix offsets;
- stable permutation indices;
- W13 activation for only routed rows in the chunk;
- inverse scatter/reduction buffer;
- one shared CUTLASS workspace sized to the maximum selected shape;
- fixed candidate buffers for output sampling.

### Initial chunk candidates

Measure:

```text
256 tokens
512 tokens
1024 tokens
```

A 2048-token chunk is considered only if the 64K memory gate remains healthy. Chunk size must be fixed for an execution plan and reported.

## Startup peak policy

The compiler may use host RAM, not device RAM. Runtime load must:

1. calculate the complete arena;
2. allocate the final arena once;
3. memory-map one source shard at a time;
4. stream directly into final device addresses;
5. release host staging before allocating the largest execution slot when possible;
6. avoid a raw device copy followed by a tiled device copy.

Record:

- free VRAM before CUDA initialization;
- after context creation;
- after immutable weights;
- after KV/workspace;
- after graph capture;
- after warm prefill/decode;
- maximum sampled process VRAM.

## Admission policy

For a 16 GB card, first release supports one 26B slot. Server startup must reject `--max-sessions > 1` unless a measured plan proves it fits. Do not inherit 12B multi-slot defaults.

## Memory regression thresholds

Any PR that changes:

- immutable bytes by more than 1 MiB;
- reusable workspace by more than 8 MiB;
- graph-private bytes by more than 1 MiB;
- 32K process peak by more than 32 MiB;

must explain the delta and update the memory ledger.
