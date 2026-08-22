# M14 — Native batch-one MoE decode

Status: ready; M11 and M13 accepted
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

- [ ] Boundary outputs match M11 within frozen tolerances.
- [ ] No fallback, host routing or token-loop allocation occurs.
- [ ] Actual native dispatch/instructions are recorded.
- [ ] Real-layer and end-to-end decode improve or a bounded retained rationale is recorded.
- [ ] M17 can select the path once at initialization.
