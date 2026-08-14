# Documentation index — Fast Track R4

## Coding-agent reading order

1. repository `AGENTS.md`;
2. [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md);
3. [`ACTIVE_CONTRACT.md`](ACTIVE_CONTRACT.md), status and [`START_HERE_CODEX.md`](START_HERE_CODEX.md);
4. the assigned file under [`milestones/`](milestones/README.md);
5. only the specs linked by that milestone;
6. historical ledgers only for a concrete question or evidence check.

For multi-agent work also read [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md).

## Active control documents

| Document | Purpose |
|---|---|
| [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md) | short owner-accepted project policy |
| [`ACTIVE_CONTRACT.md`](ACTIVE_CONTRACT.md) | compact normative execution contract and gates |
| [`FAST_TRACK_EXECUTION_PLAN.md`](FAST_TRACK_EXECUTION_PLAN.md) | waves and vertical checkpoints |
| [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json) | single machine-readable status source |
| [`MILESTONE_STATUS_BOARD.md`](MILESTONE_STATUS_BOARD.md) | human status view |
| [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md) | sub-agent isolation and merge rules |
| [`02_AGENT_OPERATING_CONTRACT.md`](02_AGENT_OPERATING_CONTRACT.md) | task behavior |
| [`06_DEPENDENCY_GRAPH.md`](06_DEPENDENCY_GRAPH.md) | dependencies |
| [`07_MEMORY_BUDGET_AND_RESIDENCY.md`](07_MEMORY_BUDGET_AND_RESIDENCY.md) | memory gates |

## Milestones

M00–M09 are accepted plans/cards. M10–M25 are concise milestone cards. M10 is next; M18 is conditional,
M24 optional and M25 is the required MTP final target.

## Specifications

Specifications are loaded on demand. The most frequently active are:

- `NATIVE_CONVERTER_ARCHITECTURE.md` and `NVFP4_QUANTIZATION_SPEC.md` for M06;
- `EMBEDDING_HEAD_SPEC.md` for M07/M16/M25;
- `DERIVED_CHECKPOINT_SCHEMA.md` for M08;
- `MEMORY_ARENA_SPEC.md` for M09/M21/M25;
- MoE, attention and reference-runtime specs for M10–M13;
- quality/benchmark specs for M19–M21;
- API/CLI spec for M22/M25.

## Historical/reference documents

`docs/DECISIONS.md`, `docs/CORRECTNESS.md`, `docs/BENCHMARKING.md`, `docs/PERFORMANCE_LEDGER.md`, source snapshots, appendices and M00–M05 evidence remain available for targeted research. They are intentionally not mandatory full reads before every edit.

## Conflict handling

Use the precedence in `AGENTS.md`, [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md) and [`ACTIVE_CONTRACT.md`](ACTIVE_CONTRACT.md). Current source, tests and accepted evidence decide factual state; a stale generated page cannot override them.
