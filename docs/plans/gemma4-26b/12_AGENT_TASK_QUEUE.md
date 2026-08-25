# Agent task queue — derived view

The machine-readable source is [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Do not edit this page as an independent status system.

## Active queue

M00–M17 and M20–M22 are accepted. Technical M23 freeze is the active technical-close step; full M19 is owner-deferred.

| Priority | Slice | State | Notes |
|---:|---|---|---|
| — | M22 product acceptance | accepted | exact M08 product identity, CLI/server behavior, observability and protected 12B regressions |
| — | M20 controlled performance | accepted | 6,572.809 prompt / 150.615 ordinary-decode retained medians; prompt stretch passes |
| — | M21 real context | accepted | 32K/64K/96K pass twice; 100K rejected; `base_max_context=98,304` |
| 1 | M23 technical Target freeze | ready next | carry M19 pending; no shipping/quality claim |
| 2 | M25 phase A assistant feasibility | parallel feasibility | docs/tools/locks only; no base-runtime mutation |
| 3 | M19 task/prose qualification | deferred owner | multi-hour suite at end; required before release claims |

Sub-agent packets follow [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md).
