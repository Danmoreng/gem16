# M10 — CPU MoE semantic oracle

Status: accepted 2026-08-14 at implementation commit `eac6b443b239d5e04c5be5daef3dd659d57d5de9`
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

- [x] Trusted BF16 captures match at named boundaries.
- [x] Router IDs, weights and tie behavior are independently tested.
- [x] Shared and each selected expert contribution are inspectable.
- [x] Quantized tensors can be consumed through an independent dequantizer.
- [x] No production CUDA implementation is used as the oracle.

## Implemented evidence

`tools/gemma4_26b_moe_oracle.py` is an independent CPU implementation of the locked BF16 path and a bounded
Safetensors/NVFP4 decoder. The real replay covers layers 0, 5, 6 and 29 at positions 0 and 17: eight shared outputs,
eight router boundaries, 24 post-norm boundaries and 64 individually hashed and compared selected-expert
contributions. The compact result is
`artifacts/m10/diagnostic-summary.json`; synthetic contract coverage is in `tests/python/test_m10_moe_oracle.py`.
The clean commit-bound acceptance record is `artifacts/m10/acceptance.json`.

The oracle freezes exact ties to lower expert ID and expert accumulation to top-k slot order in FP32. The pinned
PyTorch capture orders one exact-probability tie differently and reduces by expert ID into BF16. Its top-8 set remains
identical, while the independently recombined routed boundary has worst relative L2 below 0.005 and cosine above
0.99999. This explicit divergence is retained for M11 differential testing rather than hidden.
