# Agent task queue — derived view

The machine-readable source is [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Do not edit this page as an independent status system.

## Active queue

M00–M17 and M20–M23 are accepted. Bounded external 26B MTP characterization is next; full M19 is owner-deferred.

| Priority | Slice | State | Notes |
|---:|---|---|---|
| — | M22 product acceptance | accepted | exact M08 product identity, CLI/server behavior, observability and protected 12B regressions |
| — | M20 controlled performance | accepted | 6,572.809 prompt / 150.615 ordinary-decode retained medians; prompt stretch passes |
| — | M21 real context | accepted | 32K/64K/96K pass twice; 100K rejected; `base_max_context=98,304` |
| — | M23 technical Target freeze | accepted | exact hashes, capability report, M25 rollback and protected 12B regression |
| 1 | llama.cpp/vLLM 26B MTP baselines | active next | update and pin runtimes; bounded ordinary/MTP runs |
| 2 | M25 assistant and exact target integration | ready | preserve M23 ordinary rollback and separate `mtp_max_context` |
| 3 | M19 task/prose qualification | deferred owner | multi-hour suite at end; required before release claims |

Sub-agent packets follow [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md).
