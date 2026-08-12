# M13 — Complete slow model and early quality gate

Status: blocked
Class: critical gate
Depends on: M07, M08, M09, M11 and M12

Normative inputs: [Reference runtime](../specs/REFERENCE_RUNTIME_SPEC.md), [Quality evaluation](../specs/QUALITY_EVALUATION_SPEC.md), [Test matrix](../specs/TEST_MATRIX.md).

## Outcome

Run a complete deterministic text-only prompt and generation on the 16 GB GPU with correctness-first operators.

## In scope

- tokenizer/template and exact prompt IDs;
- bounded reference prefill and fixed-address decode;
- all 30 attention/MoE layers, final norm and provisional head;
- teacher-forced logits plus selected layer/router captures;
- resident two-turn continuation;
- no token-loop allocation or CPU expert fallback;
- a small development-corpus quality screen.

## Single early go/no-go

Pass when there is no catastrophic or unexplained drift in teacher-forced loss/KL, selected-token ranks, router behavior and a small prose/task subset. This is not held-out final qualification.

- On pass: M14–M17 are unblocked; M18 is optional.
- On fail: freeze captures and run M18 diagnosis before native promotion.

## Exit gate

- [ ] Full prompt and generation complete on the reference GPU.
- [ ] Deterministic repeated execution passes.
- [ ] Numerical captures meet the provisional envelope.
- [ ] Resident continuation/cache-prefix behavior passes.
- [ ] Early quality decision is explicitly `proceed` or `diagnose`.
- [ ] Timing remains labeled reference-only.
