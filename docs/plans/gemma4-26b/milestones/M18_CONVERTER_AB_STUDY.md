# M18 — Converter A/B and causal attribution study

## Objective

Separate quantizer effects from QAT-weight effects by comparing Unsloth direct NVFP4, ordinary BF16 compiled by gem16, and QAT BF16 compiled by the identical gem16 pipeline.

## Why this milestone exists

A direct Unsloth-versus-QAT comparison is scientifically ambiguous. The ordinary-BF16 project conversion is the control needed to determine whether quality differences arise from the QAT master weights or from a different quantizer.

## Prerequisites

- M05–M08 compiler complete
- M13 deterministic correctness runtime path
- Pinned reference captures

## Repository areas to inspect first

- `tools/compare_quantized_checkpoints.py`
- `tools/validate_gemma4_26b_full_model.py`
- `docs/CORRECTNESS.md`
- `docs/PERFORMANCE_LEDGER.md`

## Suggested additions or boundaries

- `tools/analyze_gemma4_26b_conversion.py`
- `docs/GEMMA4_26B_CONVERSION_STUDY.md`
- `benchmarks/results/gemma4_26b/conversion/`

## Implementation sequence

1. Build four immutable artifacts: Unsloth A, own ordinary B, own QAT C, and official Q4_0 D reference.
2. Normalize tensor naming and logical axes without normalizing away real quantization differences.
3. Compare A versus B per tensor and per operator to evaluate the project compiler against Unsloth.
4. Compare B versus C under the same compiler to isolate the changed master weights.
5. Compare C versus D against the same QAT BF16 source to evaluate NVFP4/FP8 versus the QAT-target Q4_0 format.
6. Capture real activation distributions from a disjoint calibration set and run module-output comparisons.
7. Measure router probability, top-8 set/order and selected-weight differences separately from expert-output drift.
8. Run model-wide teacher forcing with layerwise captures and attribute first significant divergence.
9. Do not use task benchmark results to retune the held-out test set. Freeze thresholds and candidate profiles before M19.
10. Publish negative as well as positive findings and state which hypotheses survive.
11. Issue an explicit preliminary proceed/stop decision for native performance work. A catastrophic or unexplained quality loss blocks M14–M17 before expensive kernel optimization begins.

## Required tests

- All compared artifacts have verified locks and tokenizer/template identity.
- Tensor comparison covers 100% of selected production tensors.
- Operator inputs are identical across compared weight formats.
- Router comparison reports set overlap, order agreement and weight drift.
- Teacher-forcing uses identical token IDs and capture positions.
- Repeated analysis is deterministic and output reports are schema-validated.

## Evidence and documentation outputs

- `docs/GEMMA4_26B_CONVERSION_STUDY.md`
- `artifacts/m18/tensor-a-vs-b.json`
- `artifacts/m18/operator-a-vs-b.json`
- `artifacts/m18/model-b-vs-c.json`
- `artifacts/m18/qat-nvfp4-vs-q4.json`
- `artifacts/m18/hypothesis-summary.md`

## Suggested commands

```text
python tools/analyze_gemma4_26b_conversion.py --unsloth "$UNSLOTH_26B" --ordinary "$GEM16_26B_BASE" --qat "$GEM16_26B_QAT" --q4 "$GOOGLE_Q4" --output artifacts/m18
```

## Risks to watch in this milestone

- Different source model revisions can invalidate causal attribution.
- Weight-space metrics may disagree with activation- or task-space quality.
- Router discontinuities can magnify tiny numeric differences.
- Using the held-out M19 suite during quantizer tuning creates leakage.

## Forbidden shortcuts

- Claiming QAT benefit from A versus C alone.
- Comparing different prompt tokens or chat templates.
- Selecting individual tensors from different master checkpoints based only on local MSE.
- Tuning on the held-out quality test set.
- Suppressing regressions because aggregate averages improve.

## Exit criteria

- [ ] A, B, C and D are fully locked and comparable.
- [ ] Project quantizer differences from Unsloth are quantified.
- [ ] QAT master-weight effects are isolated through B versus C.
- [ ] First-divergence and router-drift reports exist.
- [ ] Candidate profiles and M19 thresholds are frozen.
- [ ] No unsupported claim of QAT superiority remains.
- [ ] Preliminary quality result explicitly permits or blocks M14–M17.

## Downstream milestones unblocked

- M14–M17 native performance work only when the preliminary quality gate passes
- M19 held-out quality qualification after M17
- M20 final performance qualification after M17/M19

## Codex execution prompt

```text
You are implementing M18: Converter A/B and causal attribution study in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M18. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M18 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

Expand the causal matrix with candidate G (NVIDIA/ModelOpt NVFP4) and the W4A16 diagnostic arm. Required attribution comparisons become:

```text
Unsloth vs own ordinary-BF16 conversion
own ordinary-BF16 vs own QAT-BF16 conversion
own QAT W4A4 vs own QAT W4A16 diagnostic
each own arm vs ModelOpt control
each head format with identical experts
```

Do not attribute a gap to runtime or checkpoint until the same bytes have been tested through two execution paths or the same path with two checkpoint recipes.
