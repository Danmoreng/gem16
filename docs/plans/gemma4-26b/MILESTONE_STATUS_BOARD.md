# Milestone status board — Fast Track R4

Derived from [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Policy: [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md). Last synchronized: 2026-08-12.

## Accepted

`M00 M01 M02 M03 M04 M05`

## Active and parallel-ready

| Milestone/slice | State | Role |
|---|---|---|
| M06 | ACTIVE | critical native NVFP4 compiler |
| M10 phase A | PARALLEL READY | BF16 MoE oracle |
| M12 phase A | PARALLEL READY | attention/trait fixtures |
| M09 phase A | PARALLEL PREWORK | formulas and one-slot tests |
| M25 phase A | PARALLEL FEASIBILITY | assistant compatibility/memory only |
| Harness lane | PARALLEL READY | future report/test infrastructure |

## Next vertical sequence

`M06 → M07 → M08 → M09 final → M11/M12 runtime → M13`

## Conditional/optional

- M18: only on quality failure, unresolved head-format uncertainty or explicit attribution request.
- M24: optional internal Q4_0 backend.
- Vision: outside this program.

Program completion is M25, not M23. The first practical target is the vertical M06→M09→M13 path; later qualification remains explicit follow-up rather than a prerequisite for early experiments.
