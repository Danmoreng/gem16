# Agent task queue — derived view

The machine-readable source is [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Do not edit this page as an independent status system.

## Active queue

M00–M13 are accepted. M14–M17 native-runtime work is unblocked.

| Priority | Slice | State | Notes |
|---:|---|---|---|
| 1 | M14 native MoE decode | ready next | optimize against accepted M11/M13 fixtures |
| 2 | M15 grouped prefill | ready next | bounded workspace and 32K admission |
| 3 | M16 production T=1 head | ready next | optimize or record retained M13 path |
| 4 | M17 rolling integration | ready | integrate accepted native slices incrementally |
| 5 | evaluation/benchmark harness scaffolding | paused | no production claims |
| 6 | M25 phase A assistant feasibility | paused | docs/tools/locks only |

Sub-agent packets follow [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md).
