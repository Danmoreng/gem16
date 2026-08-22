# Milestone status board — Fast Track R4

Derived from [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Policy: [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md). Last synchronized: 2026-08-22.

## Accepted

`M00 M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M11 M12 M13`

## Active / next

| Milestone/slice | State | Role |
|---|---|---|
| M14 | READY_NEXT | native batch-one MoE decode |
| M15 | READY_NEXT | grouped bounded-workspace prefill |
| M16 | READY_NEXT | production T=1 tied head or recorded retained-path decision |
| M17 | READY_NEXT | rolling optimized-runtime integration |
| M25 phase A | PAUSED | assistant compatibility/memory only |
| Harness lane | PAUSED | future report/test infrastructure |

## Next vertical sequence

`M14/M15/M16 → rolling M17 integration`

## Conditional/optional

- M18: only on quality failure, unresolved head-format uncertainty or explicit attribution request.
- M24: optional internal Q4_0 backend.
- Vision: outside this program.

Program completion is M25, not M23. M13 passed with `proceed`; M14–M16 and rolling M17 are now unblocked. Later qualification
remains explicit follow-up rather than a prerequisite for early experiments.
