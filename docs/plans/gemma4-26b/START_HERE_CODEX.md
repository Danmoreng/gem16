# Start here — current coding-agent task

Status: M00–M06 accepted; M07 active.
Plan revision: Fast Track R4.

## Read now

1. repository `AGENTS.md`;
2. [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md);
3. [`ACTIVE_CONTRACT.md`](ACTIVE_CONTRACT.md);
4. [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json) and the current milestone;
5. only the specs linked by that milestone.

Do not preload the full decision, correctness, benchmark or performance ledgers. Read historical records only for a concrete question or evidence check.

## Lead-agent orchestration

The lead agent may immediately assign disjoint sub-agents for:

- M10 phase A BF16 MoE oracle;
- M12 phase A traits and attention/KV fixtures;
- M09 phase A formulas and one-slot admission tests;
- M25 phase A assistant compatibility and memory modeling;
- evaluation/benchmark harness scaffolding.

Use [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md). Lane A/M07 remains the priority and owns the tied-head compiler/reference slice.

## M07 success

M07 is complete when the QAT tied embedding/output matrix compiles once to the accepted provisional NVFP4 contract,
lookup and T=1 head behavior match independent dequantized references, softcap/suppression/tie behavior is
deterministic, and exactly one physical tied payload is planned.

## Full-run rule

For expensive conversions or publication claims, use reviewed, targeted-tested code, a clean worktree and source/output preflight. Small fixtures and bounded diagnostic probes do not need the full release workflow, but their diagnostic status must be recorded.

## After M07

Proceed directly to M08 complete-artifact and loader integration. Do not start Q4_0 backend work or MTP verifier-head optimization in M07.
