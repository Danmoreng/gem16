# Quality gate checklist

## Freeze

- [ ] Candidate artifact hashes frozen.
- [ ] Evaluator commit frozen.
- [ ] Suite hashes frozen.
- [ ] Thresholds frozen before held-out evaluation.
- [ ] Calibration/dev/test/performance sets disjoint.

## Identity

- [ ] Tokenizer/template identical where required.
- [ ] Exact input token IDs retained.
- [ ] Sampling and seeds identical.
- [ ] Output limits and stop behavior identical.
- [ ] Model/source provenance recorded.

## Numerical

- [ ] No NaN/Inf.
- [ ] Teacher-forced NLL/KL.
- [ ] Top-k agreement and reference rank.
- [ ] Logit-margin analysis.
- [ ] Layerwise residual drift.
- [ ] Attention drift.
- [ ] Router top-8 and weight drift.
- [ ] Head-format comparison.

## Tasks

- [ ] Factual/retrieval.
- [ ] Instruction following.
- [ ] Reasoning/math.
- [ ] Code with executable tests where possible.
- [ ] German/English multilingual.
- [ ] Long-form.
- [ ] Long-context retrieval.
- [ ] Structured tools if supported.
- [ ] Fixed-seed sampling.

## Analysis

- [ ] Aggregate and per-category.
- [ ] Confidence intervals.
- [ ] Worst regressions.
- [ ] Length/stop differences.
- [ ] Invalid-format cases.
- [ ] No cherry-picking.

## Decision

- [ ] Selected master source.
- [ ] Selected head format.
- [ ] Rejected candidates documented.
- [ ] Claims distinguish Q4_0 QAT from NVFP4 conversion.
- [ ] Production lock frozen.
