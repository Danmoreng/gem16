# M09 — One-slot residency, 32K gate and context feasibility

Status: accepted 2026-08-14 at implementation commit `6c3b9e456bc7fed68e2e90a51ba20c1c895fd085`
Class: critical
Unblocks: M11, M13, M15 and M21

Normative inputs: [Memory arena](../specs/MEMORY_ARENA_SPEC.md), [Session ownership](../specs/SESSION_OWNERSHIP_AND_CONCURRENCY.md).

## Outcome

Prove the real artifact fits as one fully resident 26B slot at 32K and quantify 64K/max-context feasibility.

## Phase A — may run before M08

- checked formulas for weights, separate FP8 K/V, workspace and graphs;
- 8K/16K/32K/64K profile structures;
- one-slot positive admission and second-slot rejection;
- overflow/alignment tests;
- a fixed named prefill-workspace cap; executable 256/512/1024 chunk selection is deferred to M15 with the first
  prefill path.

## Phase B — after M08

- load the real artifact;
- reconcile named regions with direct `cudaMemGetInfo` deltas;
- measure after context creation, real weights and touched slot/graph/workspace reserves;
- pass 32K with at least 700 MiB free;
- classify 64K and derive a measured max candidate, without advertising it yet.

## Implemented diagnostic evidence — 2026-08-14

- the validated 16-shard M08 artifact uploads all 1,285 tensors into one 14,696,668,160-byte immutable device arena;
- runtime-layout transforms happen during bounded upload with at most 4 MiB host staging and no persistent device
  repack;
- the named fixed regions total 469,762,048 bytes and the separate FP8 32K K/V region is 440,401,920 bytes;
- one real 32K residency slot leaves 818,741,248 bytes directly visible to CUDA, exceeding the 700 MiB gate;
- the owner-approved 400 MiB base-model long-context gate admits 64K with 483,196,928 bytes (460.81 MiB) free, leaving
  63,766,528 bytes (60.81 MiB) above the residency threshold; it remains subject to warm execution/correctness
  qualification;
- a second slot is rejected before allocation with a zero-byte CUDA allocation delta;
- protected 12B inspect plus 32K/64K memory-planner regressions pass.

The compact record is `artifacts/m09/diagnostic-summary.json`; ignored raw reports remain under `artifacts/raw/m09/`.
This run binds a dirty worktree and therefore is not formal acceptance. It reserves and touches graph/workspace
regions but does not claim captured graphs or warm model execution; those begin in M11/M12, the prefill selector is
measured in M15, and the 32K margin must be revalidated after the first warm executable path.

The clean commit-bound probe repeats every gate above and is accepted in `artifacts/m09/acceptance.json`.

## Out of scope

- `/health`, `/metrics` and Studio;
- positive two-slot scenarios;
- MTP assistant residency beyond feasibility estimates;
- long-context correctness qualification.

## Acceptance gate

- [x] Every region has checked named accounting.
- [x] One real 32K slot passes the direct 700 MiB margin.
- [x] A second 26B slot fails clearly before partial initialization.
- [x] 64K passes the direct 400 MiB base-model residency margin and its limiting regions are reported.
- [x] No modality/MTP bytes enter the base slot.
- [x] 12B memory/scheduler regressions remain green.
