# M09 — One-slot residency, 32K gate and context feasibility

Status: phase A parallel-ready; final gate waits for M08
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
- bounded prefill chunk selector.

## Phase B — after M08

- load the real artifact;
- reconcile named regions with direct `cudaMemGetInfo` deltas;
- capture after context creation, weights, slot, graphs and warm execution;
- pass 32K with at least 700 MiB free;
- classify 64K and derive a measured max candidate, without advertising it yet.

## Out of scope

- `/health`, `/metrics` and Studio;
- positive two-slot scenarios;
- MTP assistant residency beyond feasibility estimates;
- long-context correctness qualification.

## Exit gate

- [ ] Every region has checked named accounting.
- [ ] One real 32K slot passes the direct 700 MiB margin.
- [ ] A second 26B slot fails clearly before partial initialization.
- [ ] 64K feasibility and limiting regions are reported.
- [ ] No modality/MTP bytes enter the base slot.
- [ ] 12B memory/scheduler regressions remain green.
