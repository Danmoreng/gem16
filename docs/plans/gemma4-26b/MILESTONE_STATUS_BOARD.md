# Milestone status board — Fast Track R4

Derived from [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Policy: [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md). Last synchronized: 2026-08-14.

## Accepted

`M00 M01 M02 M03 M04 M05 M06 M07 M08`

## Active / next

| Milestone/slice | State | Role |
|---|---|---|
| M09 final | READY | real artifact upload, named CUDA regions and 32K residency |
| M10 phase A | PAUSED | BF16 MoE oracle |
| M12 phase A | PAUSED | attention/trait fixtures |
| M25 phase A | PAUSED | assistant compatibility/memory only |
| Harness lane | PAUSED | future report/test infrastructure |

## Next vertical sequence

`M09 final → M11/M12 runtime → M13`

## Conditional/optional

- M18: only on quality failure, unresolved head-format uncertainty or explicit attribution request.
- M24: optional internal Q4_0 backend.
- Vision: outside this program.

Program completion is M25, not M23. M08 is accepted; the next practical path is M09→M13. Later qualification
remains explicit follow-up rather than a prerequisite for early experiments.
