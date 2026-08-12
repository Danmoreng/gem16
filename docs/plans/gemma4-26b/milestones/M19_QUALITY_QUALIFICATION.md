# M19 — Held-out quality qualification

Status: blocked by frozen M17 artifact
Class: qualification

Normative inputs: [Quality evaluation](../specs/QUALITY_EVALUATION_SPEC.md), [Test matrix](../specs/TEST_MATRIX.md).

## Outcome

Decide whether the frozen production artifact is acceptable on the untouched held-out suite.

## In scope

- teacher-forced NLL/KL/rank metrics;
- task and prose evaluations;
- router/residual diagnostics on failures;
- comparison to QAT BF16 and practical external references;
- final head/profile acceptance.

M18 is triggered only if the result fails or a causal claim is required.

## Exit gate

- [ ] Exact artifact/config hashes are frozen.
- [ ] Held-out thresholds pass without tuning on the test set.
- [ ] Regressions are disclosed by category, not hidden in aggregates.
- [ ] Production quality wording is no stronger than the evidence.
