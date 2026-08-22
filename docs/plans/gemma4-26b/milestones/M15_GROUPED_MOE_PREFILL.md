# M15 — Grouped bounded-workspace MoE prefill

Status: ready; M09, M11 and M13 accepted
Class: parallel
Unblocks: M17

Normative inputs: [MoE prefill kernel](../specs/MOE_PREFILL_KERNEL_SPEC.md), [Memory arena](../specs/MEMORY_ARENA_SPEC.md).

## Outcome

Group token-expert assignments for native NVFP4 prefill without workspace growth that invalidates 32K/64K.

## In scope

- compact 8 assignments per token;
- histogram/prefix/permutation and inverse reduction;
- fixed 256/512/1024 chunk candidates;
- native grouped expert execution;
- exact token/expert mapping and deterministic reduction;
- memory/performance A/B against the reference path.

## Exit gate

- [ ] Outputs match M11/M13 fixtures.
- [ ] Workspace is named, fixed and inside the M09 cap.
- [ ] 32K admission remains healthy.
- [ ] Prefill improves on real prompts or the candidate is rejected with evidence.
- [ ] No prompt×128-expert activation is materialized.
