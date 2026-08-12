# Dependency graph — Fast Track R4

## Normal critical path

```text
M05 accepted
 └─ M06 NVFP4 expert compiler
     └─ M07 provisional NVFP4 tied head
         └─ M08 complete artifact and loader
             └─ M09 real 32K residency

M03 accepted ──> M10 CPU MoE oracle ─┐
M08 + M09 ──────────────────────────────> M11 CUDA MoE reference ─┐
M03 accepted ──> M12 phase A traits ─┐                           │
M08 + M09 ───────────────────────────> M12 phase B runtime ───────┤
M07 + M08 + M09 ──────────────────────────────────────────────────┤
                                                                  └─> M13 complete slow model + early quality gate
                 ├─ M14 native decode ─┐
                 ├─ M15 native prefill ├─ M17 rolling integration
                 └─ M16 production head┘
                         ├─ M19 quality
                         ├─ M20 performance
                         ├─ M21 context/max-fit
                         └─ M22 CLI/server
                              └─ M23 base target freeze
                                   └─ M25 MTP final target
```

## Conditional and optional lanes

```text
M13 or M19 failure ──> M18 attribution/diagnosis ──> corrective milestone
M13 or M23 ─────────> M24 optional Q4_0 backend
M03/M09 prework ────> M25 phase A feasibility; runtime integration waits for M23
```

## Parallel windows

- During M06: M10 phase A, M12 phase A, M09 prework, harness work and M25 feasibility.
- After M13 pass: M14, M15 and M16 in parallel; M17 integrates incrementally.
- After M17 hash freeze: M19, M20, M21 and M22 can proceed in parallel on the same artifact/profile contract.

## Gate ownership

| Gate | Owner | Blocks |
|---|---|---|
| native NVFP4 expert artifact | M06 | M07/M08 |
| provisional head reference | M07 | M08 |
| complete reproducible artifact | M08 | runtime binding |
| real one-slot 32K fit | M09 | full runtime promotion |
| MoE semantic parity | M10/M11 | M13/native MoE |
| attention/KV parity | M12 | M13 |
| early quality pass | M13 | normal M14–M17 path |
| optimized deterministic target | M17 | qualification |
| quality/performance/context | M19–M21 | base release |
| base evidence freeze | M23 | MTP integration |
| exact MTP and MTP context | M25 | program completion |

M18 never blocks M17 when M13 has passed. M17 never blocks M18 diagnosis. This removes the former cyclic wording.
