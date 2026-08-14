# Agent task queue — derived view

The machine-readable source is [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Do not edit this page as an independent status system.

## Active queue

M09 is accepted. M10 is implemented and awaiting clean commit-bound acceptance.

| Priority | Slice | State | Notes |
|---:|---|---|---|
| 1 | M10 CPU MoE semantic oracle | acceptance pending | dirty-worktree BF16 replay and independent NVFP4 diagnostic pass |
| 2 | M11 CUDA MoE reference | blocked | waits for accepted M10 |
| 3 | M12 phase A attention/trait fixtures | paused | no engine orchestration edits |
| 4 | M09 residency | accepted | clean evidence in `artifacts/m09/acceptance.json` |
| 5 | evaluation/benchmark harness scaffolding | paused | no production claims |
| 6 | M25 phase A assistant feasibility | paused | docs/tools/locks only |

Sub-agent packets follow [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md).
