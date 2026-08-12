# M18 — Conditional quantization/source/head diagnosis

Status: conditional
Triggers: M13 failure, M19 failure, head-format uncertainty or explicit causal-attribution request

Normative inputs: [Quality evaluation](../specs/QUALITY_EVALUATION_SPEC.md), [Checkpoint provenance](../specs/CHECKPOINT_PROVENANCE_SPEC.md).

## Outcome

Localize quality loss without blocking the normal native path when M13 is healthy.

## Possible arms

- complete ordinary-BF16 conversion using the same compiler recipe;
- QAT versus ordinary source attribution;
- provisional NVFP4 head versus external official Q4_0 or a small alternative head experiment;
- Unsloth/ModelOpt contextual comparisons;
- per-family/per-layer ablations using frozen captures.

## Rules

- development data only for tuning/diagnosis;
- large conversion/comparison runs use the native data plane;
- do not expand to every arm unless the preceding result needs it;
- produce a concrete corrective recommendation for M06/M07/M16/M24 or a stop decision.

## Exit gate

- [ ] The first material divergence is localized by source, quantizer, tensor family or runtime arithmetic.
- [ ] A bounded corrective action or rejection is recorded.
- [ ] No held-out set was used for tuning.

M18 never automatically blocks M14–M17 after an M13 pass.
