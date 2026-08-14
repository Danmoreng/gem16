# M10 — CPU MoE semantic oracle

Status: ready next after accepted M09
Class: parallel-critical
Depends on: M03; quantized adapter depends on M06
Unblocks: M11

Normative inputs: [MoE semantics](../specs/MOE_SEMANTICS_SPEC.md), [Router semantics](../specs/MOE_ROUTER_SPEC.md).

## Outcome

Create an independent transparent authority for router, shared MLP, top-8 experts, reduction, norms and residual order.

## Phase A — start now

- BF16 mathematical path from locked traits/inventory;
- FP32 router normalization/projection/softmax;
- deterministic top-8 and explicit tie policy;
- selected-probability normalization and expert scaling;
- shared branch, expert contributions and final residual captures;
- synthetic and real BF16 goldens.

## Phase B — after M06

Add dequantized NVFP4 artifact consumption without changing semantic code.

## Exit gate

- [ ] Trusted BF16 captures match at named boundaries.
- [ ] Router IDs, weights and tie behavior are independently tested.
- [ ] Shared and each selected expert contribution are inspectable.
- [ ] Quantized tensors can be consumed through an independent dequantizer.
- [ ] No production CUDA implementation is used as the oracle.
