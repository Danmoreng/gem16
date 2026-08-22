# Agent task queue — derived view

The machine-readable source is [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Do not edit this page as an independent status system.

## Active queue

M00–M12 are accepted. M13 is the next owner-selected implementation slice.

| Priority | Slice | State | Notes |
|---:|---|---|---|
| 1 | M13 full-model reference | ready next | integrate accepted M11/M12 paths and run early quality screen |
| 2 | M12 attention/KV integration | accepted | clean evidence in `artifacts/m12/acceptance.json` |
| 3 | M11 CUDA MoE reference | accepted | clean evidence in `artifacts/m11/acceptance.json` |
| 4 | M10 CPU MoE semantic oracle | accepted | clean evidence in `artifacts/m10/acceptance.json` |
| 5 | evaluation/benchmark harness scaffolding | paused | no production claims |
| 6 | M25 phase A assistant feasibility | paused | docs/tools/locks only |

Sub-agent packets follow [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md).
