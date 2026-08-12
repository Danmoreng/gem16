# Coding-agent workflow — Fast Track R4

## Start

Read `AGENTS.md`, the active contract, status JSON, assigned milestone and only linked specs. Inspect the actual tree. A drift note is required only for material differences affecting the slice.

## Parallel work

The lead agent may assign isolated worktrees using `PARALLEL_WORKSTREAMS.md`. Every sub-agent has explicit writable paths and returns one reviewable commit/patch. Global status and shared schemas are integration-owned.

## Development loop

1. add/refresh the smallest fixture or failing test;
2. implement the narrow change;
3. run lane-local tests;
4. run relevant 12B regressions for shared code;
5. commit and review;
6. run any expensive full workload only from a clean tree;
7. collect evidence once;
8. merge and update status.

## Performance work

Record hypothesis, changed arithmetic/layout/schedule, correctness, memory, microbenchmark and end-to-end result. Retain rejected results in the existing ledger; do not create a new plan document for each attempt.

## Completion

Report base commit, files, exact commands/results, evidence, deltas, exit status and merge dependency. Do not begin a dependent milestone unless assigned by the lead agent.
