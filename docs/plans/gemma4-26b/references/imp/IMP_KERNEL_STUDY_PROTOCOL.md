# imp grouped-NVFP4 kernel study protocol

## Purpose

Evaluate whether imp's grouped small-M NVFP4 design can accelerate Gemma 4 26B prefill or multi-row verification on the RTX 5080 without importing imp's general runtime architecture or a second permanent expert layout.

## Preconditions

- M13 full-model reference path passes.
- M14 T=1 decode is correct.
- Final expert weight layout is frozen provisionally.
- MIT/provenance policy is accepted.
- A 5080 benchmark host is available.

## Study arms

1. gem16 baseline grouped implementation.
2. Clean-room implementation of imp's work-queue/M-tile policy on gem16 layout.
3. Optional isolated MIT port of the relevant imp kernel.
4. CUTLASS reference using the same arithmetic and scales.
5. Serial expert oracle.

## Required shape matrix

Projections:

```text
K=2816 → N=704   gate/up
K=704  → N=2816  down
K=2816 → N=2112  shared gate/up
K=2112 → N=2816  shared down
```

Token rows per expert:

```text
M = 0, 1, 2, 4, 8, 16, 32, 64, 128, 256
```

Routing distributions:

- balanced;
- one hot expert;
- Zipf-like skew;
- real prompt histograms at 128/512/2K/8K tokens;
- worst observed tail.

## Layout gate

Document for A, B, SFA and SFB:

- logical axes;
- packed nibble order;
- row/column order;
- K-block size;
- scale vector size and encoding;
- CTA tile order;
- alignment and TMA descriptor requirements.

A port is rejected if it requires a second permanent copy of all expert weights. A bounded temporary view for active prompt experts may be evaluated only when included in peak memory and TTFT.

## Correctness gate

- byte-exact activation quantizer fixtures;
- FP32 reference output;
- per-expert alpha/global-scale contract;
- real-shape random and checkpoint activations;
- all routing-skew cases;
- compute-sanitizer and racecheck where supported;
- deterministic output checksum.

## Performance gate

Measure:

- kernel time;
- end-to-end MoE layer time;
- prompt throughput and TTFT;
- launch count;
- DRAM/L2 traffic;
- tensor-pipe activity;
- workspace and graph-private bytes;
- cold/warm initialization cost.

The kernel is retained only if it improves a real prompt profile on the RTX 5080 and does not harm the bounded-memory release profile.

## SASS gate

Disassemble and prove that the retained path contains the expected consumer-Blackwell block-scaled `mxf4nvf4` MMA. Merely compiling a source intrinsic is insufficient.
