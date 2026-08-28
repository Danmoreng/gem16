# Milestone status board — Fast Track R4

Derived from [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Policy: [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md). Last synchronized: 2026-08-28.

## Accepted

`M00 M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M11 M12 M13 M14 M15 M16 M17 M20 M21 M22 M23 M25`

## Active / next

| Milestone/slice | State | Remaining work |
|---|---|---|
| Performance optimization slice | ACCEPTED_M20 | retained medians 6,572.809 prompt / 150.615 ordinary decode tok/s; prompt stretch passes |
| M21 | ACCEPTED | 32K/64K/96K pass twice; 100K capacity-rejected; `base_max_context=98,304` |
| M23 | ACCEPTED | exact hash/evidence/capability/rollback freeze; M19 remains visibly pending |
| M25 sampled product | ACCEPTED | exact GPU sampling, CLI/server/Studio D2 chat and bounded 32K continuation pass at `c4ead1d` |
| M25 context | ACCEPTED | repaired fixed-D2 arena repeats at 86,016 with 200 MiB reserve |
| Checkpoint publication | ACCEPTED | separate immutable Target/Assistant repositories and Studio download complete |
| Main promotion | ACCEPTED | current-HEAD release gates passed; Fast Track promoted to `main` |

## Deferred

| Milestone | State | Reason |
|---|---|---|
| M19 | WAIVED_BOUNDED_REPLACEMENT | full GSM8K/AIME plus bounded sampled/product evidence accepted for this checkpoint |

The broad M19 suite is waived for this local checkpoint. Claims remain bounded to the accepted evidence and the
explicit SM120 product contract.

## Next vertical sequence

`qualified checkpoint publication → Studio immutable download → Main promotion → Fast Track complete`

## Conditional/optional

- M18: only on quality failure, unresolved head-format uncertainty or explicit attribution request.
- M24: optional internal Q4_0 backend.
- Vision: outside this program.

M25, publication, Studio download integration and Main promotion are accepted. The 12B production path remains
separately specialized, regression-protected and the default.
