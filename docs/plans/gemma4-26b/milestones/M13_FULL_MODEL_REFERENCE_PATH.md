# M13 — Complete slow model and early quality gate

Status: implementation complete; clean commit-bound acceptance pending
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

## Implementation evidence (2026-08-22)

The isolated `Gemma4Moe26BReferenceEngine` binds the complete immutable M08
arena, all 30 accepted M11/M12 layer views, the tied provisional NVFP4
embedding/head and final norm during initialization. One 440,401,920-byte FP8
K/V allocation is partitioned by the immutable attention traits, and one
3,833,888-byte fixed workspace owns all recurrent, operator, logit and capture
storage. Recurring token execution performs no allocation, filesystem access,
JIT, repack, CPU expert routing or precision fallback.

The exact text-only template encodes `Reply with exactly OK.` to the 18 locked
BF16 prompt IDs. Two same-process runs and one fresh-process run produce
deterministic `OK.` token IDs; the fresh-process full logit payloads are
bit-identical. All logits are finite. A resident second turn advances the
existing prefix from position 20 to 38 and gives the same next-token prediction
in both runs. Warm free memory remains exactly 1,280,507,904 bytes after each
run, above the 700 MiB 32K gate.

Against the locked BF16 capture, the expected first token remains rank 1,
teacher-forced KL is 0.0000121, worst selected-layer relative-L2/cosine is
0.196685/0.982661 and selected router top-8 overlap is at least 5/8 (7 of 8
captures retain 8/8). The second generated token is a period rather than the
BF16/Q4_0 turn terminator; this matches the locked direct Unsloth NVFP4
diagnostic and does not trip the explicit early-screen envelopes. The decision
is `proceed`, not a held-out quality or performance claim. Compact metrics and
raw hashes are in `artifacts/m13/diagnostic-summary.json`; acceptance must bind
them to the clean implementation commit.
