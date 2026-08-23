# Agent task queue — derived view

The machine-readable source is [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Do not edit this page as an independent status system.

## Active queue

M00–M17 are accepted. M22, bounded prefill, M21 and M20 are the active technical-close sequence; full M19 is owner-deferred.

| Priority | Slice | State | Notes |
|---:|---|---|---|
| 1 | M22 product acceptance | in progress | automate CLI/server behavior and run protected 12B regressions |
| 2 | bounded prefill slice | ready after M22 | profile first; retain only a correctness-preserving win before final evidence |
| 3 | M20/M21 runner contracts and clean freeze | after prefill | make M20 consume native M21 evidence; freeze one hash |
| 4 | M21 real context | after clean freeze | repeat 32K, execute 64K and measure `base_max_context` |
| 5 | M20 controlled performance | after M21 | approved bounded 3-warm-up/10-retained run with telemetry |
| 6 | M23 technical Target freeze | blocked by M20–M22 | carry M19 pending; no shipping/quality claim |
| 7 | M25 phase A assistant feasibility | parallel feasibility | docs/tools/locks only; no base-runtime mutation |
| 8 | M19 task/prose qualification | deferred owner | multi-hour suite at end; required before release claims |

Sub-agent packets follow [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md).
