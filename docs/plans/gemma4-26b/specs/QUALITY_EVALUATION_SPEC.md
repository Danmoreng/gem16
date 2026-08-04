# Model quality evaluation specification

## Goal

Decide whether the QAT-derived mixed FP8/NVFP4 candidate preserves or improves useful model quality relative to:

- QAT BF16;
- official Google Q4_0;
- Unsloth published NVFP4;
- own ordinary-BF16 hybrid control.

Quality is evaluated before performance promotion.

## Dataset partitioning

Maintain immutable disjoint manifests:

```text
calibration.json
development.json
held_out_test.json
performance_prompts.json
```

No held-out prompt or expected answer may influence quantizer parameters, tensor-format selection or kernel arithmetic.

## Evaluation layers

### 1. Weight reconstruction

Per tensor:

- relative L2;
- cosine;
- maximum absolute error;
- SQNR;
- saturation;
- code/scale distribution.

Diagnostic only. Not a release gate by itself.

### 2. Operator outputs

With identical real activations:

- NRMSE;
- cosine;
- max error;
- BF16 mismatch count;
- router top-8/weights;
- expert branch output;
- attention output;
- head logits.

### 3. Teacher forcing

Across fixed token sequences:

- mean NLL/cross-entropy;
- perplexity where meaningful;
- KL divergence to QAT BF16;
- top-1/top-5/top-10/top-20 agreement;
- reference-token rank;
- logit margin;
- per-position and per-category distributions;
- layerwise first divergence.

### 4. Deterministic generation

Categories:

- concise factual;
- instruction following;
- multi-step reasoning;
- mathematics;
- code;
- multilingual including German and English;
- long-form coherence;
- structured tools if supported;
- long-context retrieval.

Record token IDs, stop reason, length and deterministic hash.

### 5. Sampled generation

Use fixed seeds and exact sampling parameters. Compare task-level metrics and output validity; do not expect identical tokens.

## Router metrics

For every layer/token:

- full probability KL;
- ordered top-8 agreement;
- top-8 set overlap;
- selected weight L1;
- first changed expert;
- selected contribution drift.

Report routing drift separately because it can cause discontinuous model changes.

## Proposed provisional gates

The coding agent must freeze numeric thresholds before viewing held-out results. Suggested structure, not final values:

```text
no NaN/Inf
teacher-forced NLL regression ≤ frozen relative threshold
top-20 agreement ≥ frozen threshold
router top-8 set overlap ≥ frozen threshold
no category metric regression beyond frozen threshold
no severe worst-case regression on safety/format correctness
deterministic repeatability = 100%
```

Threshold rationale belongs in [`../appendices/ACCEPTANCE_THRESHOLD_RATIONALE.md`](../appendices/ACCEPTANCE_THRESHOLD_RATIONALE.md).

## Statistical reporting

- report sample counts;
- bootstrap confidence intervals for aggregate task scores where appropriate;
- paired comparisons because candidates receive identical prompts;
- retain per-example data;
- do not hide length/stop differences;
- distinguish measurement uncertainty from model drift.

## Task scoring

Use deterministic validators where possible:

- exact/normalized match;
- unit tests for code;
- JSON/schema parse;
- retrieval marker match;
- multiple-choice likelihood;
- reference-token NLL.

LLM-as-judge can supplement but not replace reproducible metrics. Pin judge model/version/prompts and report bias/variance.

## Worst-case analysis

List:

- largest NLL regressions;
- largest router changes;
- changed greedy tokens with smallest/largest reference margins;
- invalid formats/tool calls;
- lost long-context markers;
- repetition/degeneration cases.

Averages do not override severe regressions without an explicit decision.

## Output schema

Every result includes:

- candidate IDs and artifact hashes;
- reference ID/hash;
- evaluator commit;
- suite hash;
- exact tokenization hash;
- hardware/runtime;
- metrics and confidence intervals;
- per-example records;
- pass/fail against frozen thresholds.

## Promotion

M19 selects:

- master source;
- head format;
- production artifact lock;
- future regression thresholds.

Do not use speed in the quality decision except as a tie-break after all candidates pass.

## imp-derived corpus and attribution amendments

- Plain prose is the primary Gemma 4 PPL/NLL signal.
- Code/technical Markdown is a separate domain stratum, not the sole aggregate.
- Candidate G (ModelOpt NVFP4) is a required negative/control arm when available.
- PPL comparisons require identical token IDs, conditioning and scored rows.
- Head, KV, weight and activation quantization contributions must be isolated with controlled toggles.
- Retain per-token NLL so domain-local failures are visible.
