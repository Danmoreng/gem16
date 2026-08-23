# Parallel workstreams and sub-agent orchestration

## Integration model

`feat/gemma4-26b` is the only integration branch. The lead coding agent may create ephemeral local worktrees/branches for sub-agents and merge or cherry-pick reviewed commits. Parallelism is allowed only when file ownership and interfaces are disjoint.

The integration agent alone owns:

- `ACTIVE_CONTRACT.md`;
- `FAST_TRACK_STATUS.json` and status-board synchronization;
- shared model/manifest schema changes;
- final artifact assembly;
- cross-workstream changes to `inference_engine` orchestration;
- final evidence acceptance.

## Current workstreams after accepted M17

The owner restarted the track and accepted M00–M17. The active lanes below share the current performance candidate;
integration and evidence acceptance remain serialized.

| Lane | Immediate task | May start now | Owned area | Merge gate |
|---|---|---:|---|---|
| P — Product | M22 automated CLI/server acceptance and 12B regressions | yes | product driver/tests and compact M22 evidence | shared runtime changes remain integration-owned |
| F — Prefill | profile-driven bounded optimization | after M22 | isolated CUDA/prefill code and tests | completes before final M20/M21 evidence |
| R — Evidence contracts | make M20 consume native M21 output; add provenance/telemetry envelopes | after prefill | M20/M21 runners and manifests | freezes before model execution |
| C — Context | M21 repeated real 32K/64K qualification | after clean candidate freeze | M21 runner and raw/compact evidence | no concurrent GPU model run |
| B — Benchmark | M20 bounded 3/10 telemetry qualification | after M21 | M20 runner and raw/compact evidence | consumes matching M21 evidence; no code changes during measurement |
| Q — Quality | deferred M19 task/prose qualification | no, owner-deferred | quality runner and blind-review package | required before release claims |
| T — MTP feasibility | M25 phase A assistant lock/inventory and compressed-memory model | yes, read-only base | docs/tools/model locks only | base runtime remains untouched |

## Historical M14–M17 parallel window (accepted)

After M13 passes:

| Lane | Task | Shared dependency |
|---|---|---|
| D1 | M14 native MoE decode | accepted M11/M13 fixtures |
| D2 | M15 grouped MoE prefill | accepted M11/M13 fixtures and M09 workspace cap |
| D3 | M16 T=1 production head | accepted M07/M13 head fixtures |
| I | M17 rolling integration | consumes reviewed commits from D1–D3 |
| Q | M18 diagnosis, only if triggered | frozen M13 captures |

This window is complete. M14–M17 are accepted and are retained here only as workstream history.

## File-overlap rules

A sub-agent task packet must list exact writable paths. Typical safe ownership:

- compiler agent: `src/compiler/**`, compiler CLI implementation and compiler tests;
- CPU oracle agent: new `src/numeric/gemma4_26b_moe*` and its tests;
- attention-fixture agent: trait tables and dedicated test files, not shared engine orchestration;
- memory agent: `src/runtime/memory_plan*` and tests, with shared public structs changed only through an integration-owned interface commit;
- harness agent: `tools/validate_*`, `benchmarks/prompts/**`, report schemas;
- MTP feasibility agent: source-lock/inventory tools and planning documents, no `src/cuda/engine/**` edits.

When two lanes need the same public type, the integration agent lands a small interface commit first. Sub-agents then rebase on that commit.

## Task packet

Every sub-agent receives:

```text
workstream and slice
base commit
writable paths
read-only interfaces
prerequisite evidence
required tests
required output/evidence
merge dependency
explicit non-goals
```

A sub-agent returns one reviewable commit or patch plus exact commands/results. It does not update global status or begin a dependent task.

## Merge rules

1. Rebase or refresh onto the current integration commit.
2. Run the lane-local test tier.
3. Review arithmetic/layout changes independently from scheduling changes.
4. Merge the smallest interface-first commit where needed.
5. Run shared 12B regression tests after any common loader/runtime/CUDA change.
6. Update `FAST_TRACK_STATUS.json` only after the merged commit and evidence pass.

## Work that must remain serialized

- M06 native protocol changes and M07 compiler extension in the same compiler files;
- final M08 schema plus loader binding;
- real M09 allocation reconciliation while M08 output is still changing;
- integration-owned engine dispatch/graph changes;
- artifact-format selection and publication lock generation;
- final quality tuning and held-out evaluation;
- MTP base-target freeze, assistant binding and final product release.

## Full-run ownership

Only the integration agent starts full model conversions, held-out evaluations or release benchmarks. Sub-agents may prepare commands and perform small probes. Full runs require a clean committed worktree and a preflight record.
