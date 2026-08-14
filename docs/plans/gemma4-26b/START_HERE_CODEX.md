# Start here — current coding-agent task

Status: M00–M08 accepted; M09 final residency reconciliation is next.
Plan revision: Fast Track R4.

## Read now

1. repository `AGENTS.md`;
2. [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md);
3. [`ACTIVE_CONTRACT.md`](ACTIVE_CONTRACT.md);
4. [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json) and the current milestone;
5. only the specs linked by that milestone.

Do not preload the full decision, correctness, benchmark or performance ledgers. Read historical records only for a concrete question or evidence check.

## Lead-agent orchestration

M08 was accepted on 2026-08-14. The lead agent may assign disjoint sub-agents for:

- M10 phase A BF16 MoE oracle;
- M12 phase A traits and attention/KV fixtures;
- M09 phase A formulas and one-slot admission tests;
- M25 phase A assistant compatibility and memory modeling;
- evaluation/benchmark harness scaffolding.

Use [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md). M08 is accepted; M09 final reconciliation is ready next.

## M08 success

M08 is accepted at implementation commit `f433358b8e2c1250b95801fc898faee4fcedcbe5`. Two clean complete hybrid
builds are byte-identical, the external lock and direct C++ loader validate, the 12B inspect regression passes and
the exact single-arena reference-GPU admission succeeds. See `artifacts/m08/acceptance.json`.

## Full-run rule

For expensive conversions or publication claims, use reviewed, targeted-tested code, a clean worktree and source/output preflight. Small fixtures and bounded diagnostic probes do not need the full release workflow, but their diagnostic status must be recorded.

## Current M09 boundary

Reconcile the real artifact with named CUDA regions and prove one fully resident 32K slot with at least 700 MiB
directly measured free-device margin after initialization. M08's admission probe is exact-arena synthetic evidence,
not model execution. Q4_0 backend work and MTP verifier-head optimization remain outside M09.
