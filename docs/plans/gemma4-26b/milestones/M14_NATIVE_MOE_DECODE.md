# M14 — Native batch-one MoE decode

Status: accepted at `9a374c3dda10b7ae870c712cd70a60aa0a9e2c52`
Class: parallel
Unblocks: M17

Normative inputs: [MoE decode kernel](../specs/MOE_DECODE_KERNEL_SPEC.md), [CUDA state lifecycle](../specs/CUDA_STATE_LIFECYCLE_SPEC.md).

## Outcome

Replace the correctness MoE decode path with an SM120-native all-GPU implementation for one token.

## In scope

- deterministic router and compact top-8 descriptors;
- native NVFP4 Gate/Up/Down execution;
- bounded shared/expert scheduling and FP32 reduction;
- fixed-address buffers and graph-safe launch behavior;
- adjacent correctness and microbenchmark A/B against M11.

## Exit gate

- [x] Boundary outputs match M11 within frozen tolerances.
- [x] No fallback, host routing or token-loop allocation occurs.
- [x] Actual native dispatch/instructions are recorded.
- [x] Real-layer and end-to-end decode improve or a bounded retained rationale is recorded.
- [x] M17 can select the path once at initialization.

## Implementation evidence (2026-08-22)

The device-selected SM120 path consumes the accepted expert-major row8/K64
payload directly. Real layer 0 is bitwise identical at every captured shared,
routed and output boundary and improves from 1.09499 ms to 0.542123 ms. The
focused CUDA test captures, instantiates and replays the complete native MoE
layer and observes the same output. Compact evidence is in
`artifacts/m14/acceptance.json`.
