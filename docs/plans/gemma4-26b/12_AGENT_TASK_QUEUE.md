# Agent task queue — derived view

The machine-readable source is [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Do not edit this page as an independent status system.

## Active queue

M00–M10 are accepted. M11 is the next owner-selected implementation slice.

| Priority | Slice | State | Notes |
|---:|---|---|---|
| 1 | M11 CUDA MoE reference | ready next | fixed-address correctness path against accepted M10 |
| 2 | M10 CPU MoE semantic oracle | accepted | clean evidence in `artifacts/m10/acceptance.json` |
| 3 | M12 phase A attention/trait fixtures | paused | no engine orchestration edits |
| 4 | M09 residency | accepted | clean evidence in `artifacts/m09/acceptance.json` |
| 5 | evaluation/benchmark harness scaffolding | paused | no production claims |
| 6 | M25 phase A assistant feasibility | paused | docs/tools/locks only |

Sub-agent packets follow [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md).
