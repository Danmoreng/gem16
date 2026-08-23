# Milestone status board — Fast Track R4

Derived from [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Policy: [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md). Last synchronized: 2026-08-23.

## Accepted

`M00 M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M11 M12 M13 M14 M15 M16 M17 M22`

## Active / next

| Milestone/slice | State | Remaining work |
|---|---|---|
| Performance optimization slice | ACTIVE_NEXT | profile-driven, correctness-preserving prefill/decode work toward hard 6,000/150 token/s targets and 6,500 prefill stretch |
| Runner/freeze slice | READY_AFTER_PERFORMANCE | encode the fixed target row, align M20/M21 schemas and provenance, then build one clean immutable candidate |
| M21 | READY_AFTER_FREEZE | repeated real 32K, explicit real 64K pass/fail and measured `base_max_context` |
| M20 | IN_PROGRESS_AFTER_M21 | exact 16K+64 retained medians must reach >=6,000 prompt and >=150 ordinary-decode tok/s; report >=6,500 prompt stretch; latest adjacent development candidate is 5,050.92/139.106, formal medians pending |
| M23 | BLOCKED_BY_M20_M21 | freeze a technical Target with rollback and M19 explicitly pending |
| M25 phase A | PARALLEL_FEASIBILITY_ONLY | assistant compatibility and memory feasibility only |

## Deferred

| Milestone | State | Reason |
|---|---|---|
| M19 | DEFERRED_OWNER | numerical Q4 checks pass; multi-hour task/prose suite is postponed until the end of the current implementation/performance program |

M19 remains mandatory before any shipping or production-quality claim. Its deferral no longer blocks an engineering
M23 Target freeze, but that checkpoint must remain visibly experimental and quality-unqualified.

## Next vertical sequence

`bounded prefill/decode optimization → clean freeze → M21 → M20 → technical M23 → M25 integration → deferred M19 release gate`

## Conditional/optional

- M18: only on quality failure, unresolved head-format uncertainty or explicit attribution request.
- M24: optional internal Q4_0 backend.
- Vision: outside this program.

Program completion still requires M25 and the deferred M19 release gate. The 12B production path remains separately
specialized and regression-protected throughout.
