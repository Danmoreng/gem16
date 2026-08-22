# M11 — CUDA correctness-first MoE

Status: implementation complete; clean commit-bound acceptance pending
Class: critical
Depends on: M08, M09 and M10
Unblocks: M13, M14 and M15

Normative inputs: [MoE semantics](../specs/MOE_SEMANTICS_SPEC.md), [CUDA state lifecycle](../specs/CUDA_STATE_LIFECYCLE_SPEC.md).

## Outcome

Implement a fixed-address GPU reference path that matches M10 before performance fusion.

## In scope

- observable router kernels and deterministic top-8;
- existing correctness-oriented NVFP4 projections;
- bounded eight-expert diagnostic contributions and locked-order reduction;
- shared branch, norms, residual and layer scalar;
- no host routing or per-forward allocation;
- sanitizer coverage and repeated deterministic runs.

## Exit gate

- [ ] Named CUDA boundaries match M10.
- [ ] Repeated runs produce identical IDs and outputs.
- [ ] No allocation or CPU routing occurs after initialization.
- [ ] memcheck/racecheck/initcheck pass for the targeted path.
- [ ] The path is labeled correctness-only, not performance-qualified.

## Implementation evidence (2026-08-22)

The isolated 26B CUDA reference path now owns an immutable single-arena artifact
binding and a fixed-address one-token MoE layer. Router softmax/top-8 remains on
the device, uses lower expert ID for exact probability ties, and each of the
eight fixed slot launches reads its selected expert ID on the device. Shared and
routed branches expose the M10 boundaries and reduce weighted contributions in
top-k slot order with FP32 accumulation. No 12B engine orchestration was changed.

The dirty diagnostic at `artifacts/m11/diagnostic-summary.json` uses real layer 0,
position 0 M10 captures and the complete locked M08 arena. Fixed quantized-versus-
BF16 gates are: router probability max-abs 0.003; selected weight max-abs 0.005;
shared relative-L2/cosine 0.20/0.985; every expert contribution 0.30/0.96; and
routed sum 0.20/0.985. The diagnostic passes at shared 0.136865/0.993189,
worst expert 0.248573/0.968645, and routed sum 0.106035/0.995562. Four real
forwards are bitwise identical and report no CUDA-visible allocation delta.
Ignored raw evidence is retained only by byte count and SHA-256 in the compact
summary.

The focused synthetic path additionally fixes the all-equal router result to
expert IDs 0 through 7. Compute Sanitizer memcheck, racecheck and initcheck pass
that complete targeted layer path. Clean acceptance must bind these gates to the
implementation commit before the checklist is marked complete.
