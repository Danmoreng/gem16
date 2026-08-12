# Start here — current coding-agent task

Status: M00–M07 accepted; paused before M08.
Plan revision: Fast Track R4.

## Read now

1. repository `AGENTS.md`;
2. [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md);
3. [`ACTIVE_CONTRACT.md`](ACTIVE_CONTRACT.md);
4. [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json) and the current milestone;
5. only the specs linked by that milestone.

Do not preload the full decision, correctness, benchmark or performance ledgers. Read historical records only for a concrete question or evidence check.

## Lead-agent orchestration

The track is paused after M07 by owner request. Do not start M08 or assign further 26B implementation work without
explicit owner instruction. When resumed, the lead agent may assign disjoint sub-agents for:

- M10 phase A BF16 MoE oracle;
- M12 phase A traits and attention/KV fixtures;
- M09 phase A formulas and one-slot admission tests;
- M25 phase A assistant compatibility and memory modeling;
- evaluation/benchmark harness scaffolding.

Use [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md). M07 is accepted; M08 is next and ready but not started.

## M07 success

M07 is accepted at implementation commit `60f500b7be567fafd483ebd6f5f9b07988197ca1` after a clean Release QAT
conversion, independent lookup/T=1 diagnostic, deterministic host tests, and exact one-payload validation.

## Full-run rule

For expensive conversions or publication claims, use reviewed, targeted-tested code, a clean worktree and source/output preflight. Small fixtures and bounded diagnostic probes do not need the full release workflow, but their diagnostic status must be recorded.

## After M07

M08 complete-artifact and loader integration is next, but is explicitly paused and has not started. Resume only after
owner instruction. Do not start Q4_0 backend work or MTP verifier-head optimization in M07.
