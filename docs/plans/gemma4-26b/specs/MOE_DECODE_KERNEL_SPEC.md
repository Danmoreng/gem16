# Native MoE decode kernel specification

## Objective

Minimize batch-one feed-forward latency for one token while preserving the accepted MoE semantics and using native SM120 NVFP4 execution.

## Work per layer

```text
router
shared W13
shared activation
shared W2
8 selected expert W13
8 expert activations
8 selected expert W2
weighted reduction
branch combine/norm/residual
```

## Weight layout

Preferred immutable device organization:

```text
layer
  shared_gate/up/down
  expert_gate_up[expert]
  expert_down[expert]
```

Within each projection, use the established Row8/K64 block order:

```text
[row tile 8][K block 64][row][packed E2M1]
[row tile 8][K block 64][row][4 local scales]
```

The expert axis must allow direct pointer arithmetic from selected ID. Never gather selected weights into temporary memory.

## Router and control

A fixed device control region carries:

- current hidden pointer;
- top-8 IDs;
- top-8 weights;
- dynamic activation divisors/scales;
- output pointer;
- diagnostic flags.

Graph replay changes values, not addresses.

## W13 design

Candidate strategies:

1. one kernel per selected expert with shared activation;
2. one persistent kernel processing eight expert pointers;
3. grouped block-scaled MMA with selected experts as a batch dimension;
4. specialized direct T=1 kernels.

Benchmark each. Avoid assuming a single grouped GEMM is best at T=1.

Requirements:

- gate/up logical split exact;
- activation fragment reuse when safe;
- no full eight-expert FP32 output tensor;
- no local-memory spill;
- native NVFP4 MMA.

## Intermediate quantization

For each selected expert:

```text
gate/up → required cast
activation product
dynamic NVFP4 quantization in groups of 16
```

Prefer bounded per-expert or interleaved storage sized:

```text
8 × 704
```

plus scales, not `128 × 704`.

Shared intermediate is `2112`.

## W2 and reduction

Candidate:

- compute expert W2 into per-expert partial hidden rows;
- multiply by router weights;
- reduce eight rows in locked order.

A fused W2 weighted accumulation can reduce traffic, but acceptance requires:

- exact or approved output drift;
- deterministic order;
- no atomics with unstable order;
- material end-to-end win.

## Shared branch

Execute every layer. It can overlap with routed experts only after correctness and stream/graph ownership are proven. Overlap must not require duplicate hidden or weight buffers beyond the memory plan.

## Launch budget

Track:

- router launches;
- quantization launches;
- W13/W2 launches;
- norms/reductions;
- total graph nodes.

Launch count is evidence, not the objective. A larger fused kernel that reduces occupancy can lose.

## Profiling metrics

- duration per family;
- DRAM bytes/throughput;
- Tensor Core utilization;
- issue stalls;
- occupancy;
- registers;
- local memory;
- shared memory;
- L2 hit rate;
- activation quantization fraction;
- router fraction.

## Correctness

Compare:

- top-8;
- quantized activation bytes/scales;
- W13 outputs;
- products;
- W2 outputs;
- weighted sum;
- shared branch;
- final layer.

## Acceptance

A production decode candidate must:

- beat the CUDA reference materially;
- preserve full-model quality envelope;
- use native NVFP4;
- have zero token-loop allocation;
- have zero fallback;
- capture/replay;
- stay within memory and register limits;
- pass sanitizer and deterministic tests.
