# M19 — Held-out model-quality qualification and final format selection

## Objective

Evaluate the frozen candidates on a held-out suite, select the production master/format combination, and define the accepted quality envelope for release.

## Why this milestone exists

The project must not equate quantization error, a few matching tokens or faster inference with preserved model quality. This milestone makes the quality decision before headline performance promotion.

## Prerequisites

- M18 study complete
- Frozen candidates and thresholds
- M17 optimized runtime

## Repository areas to inspect first

- `docs/CORRECTNESS.md`
- `tools/validate_sampling.py`
- `benchmarks/prompts/`
- `docs/PERFORMANCE_LEDGER.md`

## Suggested additions or boundaries

- `tools/evaluate_gemma4_26b_quality.py`
- `docs/GEMMA4_26B_QUALITY.md`
- `benchmarks/quality/gemma4_26b/`
- `benchmarks/baselines/gemma4_26b_quality/`

## Implementation sequence

1. Finalize disjoint calibration, development and held-out test manifests with hashes and licenses.
2. Evaluate QAT BF16 reference where hardware allows, official Google Q4_0, Unsloth NVFP4, own ordinary hybrid, QAT hybrid with Q4_0 head and QAT hybrid with NVFP4 head.
3. Measure teacher-forced NLL/cross-entropy, KL divergence, top-1/top-5/top-10 agreement, logit-margin changes, router top-8 overlap and layerwise residual drift.
4. Evaluate deterministic generation on reasoning, factual retrieval, instruction following, code, multilingual and long-form prompts with fixed token budgets.
5. Evaluate sampled output with fixed seeds and the checkpoint's recommended sampling separately from greedy.
6. Use task metrics appropriate to each set; report confidence intervals or bootstrap intervals when feasible.
7. Inspect worst regressions by category and by prompt rather than accepting an aggregate average alone.
8. Decide the master source and tied-head format. Record rejected candidates and reasons.
9. Freeze the production artifact lock and quality thresholds for future regressions.
10. Update capability and model-profile names so experimental candidates cannot be mistaken for production.

## Required tests

- Held-out manifest and all model/tokenizer locks validate before evaluation.
- Every candidate receives identical tokenized inputs, sampling and output limits.
- Evaluator is deterministic where expected and records versions/seeds.
- No benchmark prompt appears in compiler calibration data.
- Worst-case and category-level regressions stay within the accepted envelope.
- Final selected artifact reproduces all reported hashes on a clean run.

## Evidence and documentation outputs

- `docs/GEMMA4_26B_QUALITY.md`
- `artifacts/m19/quality-summary.json`
- `artifacts/m19/teacher-forcing.json`
- `artifacts/m19/task-results.json`
- `artifacts/m19/worst-regressions.md`
- `docs/DECISIONS.md` final source/head-format decision

## Suggested commands

```text
python tools/evaluate_gemma4_26b_quality.py --suite benchmarks/quality/gemma4_26b/test.json --models configs/gemma4_26b_candidates.json --output artifacts/m19
```
```text
python tools/verify_compiled_model.py --model "$GEM16_26B_FINAL" --lock models/gemma4-26b-gem16-hybrid.lock.json
```

## Risks to watch in this milestone

- QAT BF16 reference inference may require a larger GPU and introduce different kernels.
- Task scores can be noisy or contaminated by prompt formatting.
- A candidate may improve average metrics while causing severe category-specific failures.
- Changing the head format after evaluation requires rebuilding and rerunning the entire suite.

## Forbidden shortcuts

- Selecting the fastest candidate before quality results.
- Changing thresholds after seeing held-out results without a documented new test split.
- Cherry-picking prompts or reporting only aggregate averages.
- Calling QAT-derived NVFP4 officially QAT-optimized for NVFP4.
- Mixing weights from ordinary and QAT master checkpoints tensor-by-tensor without a separate research decision.

## Exit criteria

- [ ] A final production source and head format are selected.
- [ ] Selected candidate passes all frozen aggregate, category and worst-case thresholds.
- [ ] Official Q4_0 and Unsloth comparisons are reported honestly.
- [ ] Quality artifacts and evaluator versions are immutable and reproducible.
- [ ] Rejected profiles are documented and not exposed as default.
- [ ] The final artifact lock is frozen for M20–M23.

## Downstream milestones unblocked

- M20 performance qualification
- M21 long-context quality
- M22 product integration

## Codex execution prompt

```text
You are implementing M19: Held-out model-quality qualification and final format selection in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M19. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M19 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

Use plain prose as the primary Gemma 4 PPL/NLL corpus and report code/technical Markdown separately. Verify identical token streams and scored windows before comparison. Add explicit LM-head, KV and W4A4/W4A16 attribution experiments. The ModelOpt control is expected to be informative even if rejected.
