# Agent task queue — derived view

The machine-readable source is [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Do not edit this page as an independent status system.

## Active queue

M00–M17 and M22 are accepted. Bounded performance optimization, M21 and M20 are the active technical-close sequence; full M19 is owner-deferred.

| Priority | Slice | State | Notes |
|---:|---|---|---|
| — | M22 product acceptance | accepted | exact M08 product identity, CLI/server behavior, observability and protected 12B regressions |
| 1 | bounded prefill/decode optimization | active next | profile first; preserve correctness while pursuing >=6,000 prompt and >=150 ordinary-decode tok/s, with >=6,500 prompt stretch |
| 2 | M20/M21 runner contracts and clean freeze | after performance targets | encode the fixed 16K+64 row, make M20 consume native M21 evidence and freeze one hash |
| 3 | M21 real context | after clean freeze | repeat 32K, execute 64K and measure `base_max_context` |
| 4 | M20 controlled performance | after M21 | bounded 3-warm-up/10-retained row with hard 6,000/150 gates and reported 6,500 prefill stretch |
| 5 | M23 technical Target freeze | blocked by M20/M21 | carry M19 pending; no shipping/quality claim |
| 6 | M25 phase A assistant feasibility | parallel feasibility | docs/tools/locks only; no base-runtime mutation |
| 7 | M19 task/prose qualification | deferred owner | multi-hour suite at end; required before release claims |

Sub-agent packets follow [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md).
