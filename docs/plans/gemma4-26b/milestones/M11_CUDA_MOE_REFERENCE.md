# M11 — CUDA correctness-first MoE

Status: ready next after accepted M10
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
