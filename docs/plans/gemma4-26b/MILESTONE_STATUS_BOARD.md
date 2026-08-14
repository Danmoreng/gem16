# Milestone status board — Fast Track R4

Derived from [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Policy: [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md). Last synchronized: 2026-08-14.

## Accepted

`M00 M01 M02 M03 M04 M05 M06 M07 M08 M09 M10`

## Active / next

| Milestone/slice | State | Role |
|---|---|---|
| M11 | READY_NEXT | CUDA correctness-first MoE against accepted M10 |
| M12 phase A | PAUSED | attention/trait fixtures |
| M25 phase A | PAUSED | assistant compatibility/memory only |
| Harness lane | PAUSED | future report/test infrastructure |

## Next vertical sequence

`M11/M12 runtime → M13`

## Conditional/optional

- M18: only on quality failure, unresolved head-format uncertainty or explicit attribution request.
- M24: optional internal Q4_0 backend.
- Vision: outside this program.

Program completion is M25, not M23. M10 is accepted; M11 is the next practical vertical slice. Later qualification
remains explicit follow-up rather than a prerequisite for early experiments.
