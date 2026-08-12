# Coding-agent operating contract — Fast Track R4

## Required reading

For a normal task read only:

1. repository `AGENTS.md`;
2. [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md);
3. [`ACTIVE_CONTRACT.md`](ACTIVE_CONTRACT.md), status and the assigned milestone card;
4. only the normative specs linked by that milestone;
5. current source and the narrow tests you will change.

Read the full `docs/DECISIONS.md`, `docs/CORRECTNESS.md`, `docs/BENCHMARKING.md` or historical evidence only for a specific question, regression or evidence check.

## Scope and parallelism

- `feat/gemma4-26b` is the integration branch.
- Ephemeral sub-agent branches/worktrees are allowed under [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md).
- Every task has explicit writable paths and a merge dependency.
- Do not edit global status, shared schemas or integration orchestration unless assigned.
- Do not weaken or genericize the 12B hot path.

## Correctness and fallback

- Missing native capability fails visibly.
- Diagnostic fallback must be explicit and cannot be used for promotion.
- Never mix mathematical tensors from independently sourced checkpoints in the production artifact.
- No CPU expert routing, offload, streaming or hidden layout duplication.
- No allocation, filesystem access, JIT or repack in the token loop.

## Expensive runs

Full conversions, held-out evaluations and publication benchmarks run only from a clean committed worktree after targeted review/tests and preflight. Unit tests and bounded probes may run during development.

## Test policy

Run the smallest applicable tier first and scale effort with risk. Documentation, planning and isolated host changes need focused checks; shared loader/runtime/CUDA changes require relevant 12B regressions; full quality/performance gates apply only to qualification claims. Quality runs follow correctness, timing follows quality sanity, and release evidence uses a frozen artifact hash.

## Stop conditions

Stop the affected lane and report when a locked source or tensor contract changes, conversion is nondeterministic, the 32K margin fails, router/attention semantics cannot be reconciled, 12B output changes unexpectedly, or a native path silently falls back. Other independent lanes may continue.

## Completion response

Return:

```text
slice and base commit
files changed
commands and exact results
evidence paths
memory/performance delta if relevant
exit criteria pass/fail
known risks
merge dependency and newly unblocked work
```
