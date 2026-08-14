# Master implementation plan — Fast Track R4

## Mission

Deliver Gemma 4 26B A4B on one approximately 16 GB Blackwell GPU with fully resident weights, a 32K first context gate, a measured 64K-or-higher single-user profile where feasible, and an exact Target-verified MTP path. Preserve the mature 12B implementation.

The active contract is [`ACTIVE_CONTRACT.md`](ACTIVE_CONTRACT.md). The detailed wave plan is [`FAST_TRACK_EXECUTION_PLAN.md`](FAST_TRACK_EXECUTION_PLAN.md).

## Accepted history

M00–M05 are accepted. Their files and evidence are historical records and are not rewritten by R4.

## Critical path

```text
M06 NVFP4 experts
 → M07 provisional NVFP4 tied head
 → M08 complete artifact and direct loader
 → M09 real one-slot 32K residency
 → M11 CUDA MoE reference + M12 runtime attention/KV
 → M13 complete slow model and early quality gate
 → M14/M15/M16 in parallel, integrated continuously by M17
 → M19/M20/M21 in parallel on one frozen artifact
 → M22 CLI/server
 → M23 base target freeze
 → M25 MTP final target
```

M10 semantic work begins in parallel with M06. M18 is conditional diagnosis. M24 is optional Q4_0 work.

## First useful checkpoints

| Checkpoint | Meaning |
|---|---|
| M08 | a complete artifact can be validated and loaded |
| M09 | the real artifact fits at 32K with the required margin |
| M13 | a complete deterministic reference generation works |
| M17 | the optimized all-resident path works |
| M23 | the base target is qualified and frozen |
| M25 | MTP and its maximum safe context are qualified |

## Promotion gates

- Artifact integrity: complete tensor coverage, source/compiler provenance, two clean M08 builds with identical hashes.
- Runtime correctness: independent MoE oracle, CUDA reference, attention/KV tests and full-model captures.
- Memory: one slot, 32K with at least 700 MiB free; base-model 64K+ with at least 400 MiB free; MTP 64K+ keeps 500 MiB.
- Early quality: M13 development-screen pass.
- Final qualification: M19 quality, M20 performance and M21 context on one frozen hash.
- MTP: compatible assistant, exact Target verification, transactional state and separately measured MTP context.

## Execution discipline

Use small, reviewable commits and parallel workstreams with disjoint ownership. Full conversions and publication-grade runs require a clean committed worktree. Do not create evidence by repeating unchanged expensive runs.

## Noncritical work

An internal Q4_0 backend, full causal attribution, positive multi-slot admission, Studio polish and vision do not block the vertical path.
