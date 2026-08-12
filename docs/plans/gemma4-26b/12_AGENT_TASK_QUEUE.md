# Agent task queue — derived view

The machine-readable source is [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Do not edit this page as an independent status system.

## Active queue

| Priority | Slice | State | Notes |
|---:|---|---|---|
| 1 | M06 native NVFP4 expert compiler | active | critical lane |
| 2 | M10 phase A BF16 MoE oracle | parallel-ready | disjoint numeric/tests area |
| 3 | M12 phase A attention/trait fixtures | parallel-ready | no engine orchestration edits |
| 4 | M09 phase A formulas and one-slot tests | parallel-ready | final reconciliation waits for M08 |
| 5 | evaluation/benchmark harness scaffolding | parallel-ready | no production claims |
| 6 | M25 phase A assistant feasibility | parallel-ready | docs/tools/locks only |
| 7 | M07 provisional NVFP4 head | blocked by M06 | intentionally small |
| 8 | M08 complete artifact/loader | blocked by M06/M07 | next vertical checkpoint |

Sub-agent packets follow [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md).
