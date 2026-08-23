# Integration and review sequence — Fast Track R4

## Branch policy

`feat/gemma4-26b` is the integration branch. Sub-agents may use short-lived local worktree branches. The integration agent merges reviewed, scoped commits.

## Recommended commit groups

1. shared interface commit, only when parallel lanes require one;
2. lane-local implementation and tests;
3. evidence/report update;
4. integration commit;
5. status update after acceptance.

Do not combine arithmetic, layout and scheduling changes when they can be reviewed separately.

## Current sequence

- M00–M17 are accepted and remain the protected baseline.
- Close M22 product behavior with automated coverage and 12B regressions.
- Profile prefill and retain only a bounded correctness-preserving improvement before the final evidence freeze.
- Repair the evidence contracts, freeze one clean candidate, run real M21 32K/64K qualification, then run bounded
  formal M20 telemetry against that exact M21 evidence.
- M23 aggregates exact-hash evidence and freezes a technical Target with M19 explicitly pending.
- M25 integrates MTP after that Target is frozen.
- Run the owner-deferred multi-hour M19 task/prose gate before shipping or production-quality status.

A pull request is opened only when the project owner requests publication. Milestone commits and evidence are sufficient for internal progression.
