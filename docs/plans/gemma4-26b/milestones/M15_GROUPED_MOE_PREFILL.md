# M15 — Grouped bounded-workspace MoE prefill

Status: accepted at `9a374c3dda10b7ae870c712cd70a60aa0a9e2c52`
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

- [x] Outputs match M11/M13 fixtures.
- [x] Workspace is named, fixed and inside the M09 cap.
- [x] 32K admission remains healthy.
- [x] Prefill improves on real prompts or the candidate is rejected with evidence.
- [x] No prompt×128-expert activation is materialized.

## Current implementation slice (2026-08-22)

The bounded planner selects among 128/256/512/1024-token candidates with
checked, 256-byte-aligned region arithmetic. At 1024 tokens it records exactly
8192 assignments, uses 165,120 bytes of permutation workspace and 165,728,256
bytes of MoE workspace, inside the accepted 64/192 MiB caps without a
prompt×128 activation. Routing, histogram/prefix, stable permutation and inverse
mapping are device-only. Native grouped W13 and W2 execute selected experts
directly and deterministic slot-order reduction preserves the M11 BF16
boundaries.

On the accepted real layer-0 fixture, a 128-token grouped launch is bitwise
identical to 128 individual M14 launches, including deterministic replay and a
valid stable permutation. It measures 2.47318387 ms versus 77.5195084 ms
(31.344×). Recurring launches have zero observed allocation delta and the full
path captures, instantiates and replays as a CUDA graph. The workspace consumes
the already accepted M09 fixed MoE/permutation reserves, so the accepted 32K
818,741,248-byte free margin is unchanged. Compact commit-bound evidence is in
`artifacts/m15/acceptance.json`.
