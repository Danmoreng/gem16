# Acceptance-threshold rationale

## Principle

Thresholds are frozen before held-out evaluation. They should be tight enough to catch meaningful regressions and broad enough to accommodate expected floating-point/quantization differences.

This appendix defines how to choose thresholds, not final numeric values.

## Operator thresholds

Use scale-aware metrics:

- exact bytes where the contract is bit-compatible;
- max absolute plus relative error for small values;
- NRMSE/cosine for large projections;
- BF16 bit mismatch where a BF16 boundary is normative;
- exact IDs for router top-k when reference margins are not tied.

Set thresholds from:

1. scalar host versus trusted reference;
2. CUDA reference versus host;
3. existing accepted 12B FP8/NVFP4 behavior;
4. observed numerical noise across deterministic hardware paths.

Do not set them from the candidate being approved alone.

## Router thresholds

Routing is discontinuous.

Separate:

- probability/logit error;
- top-8 set;
- top-8 order;
- selected weight;
- downstream output.

Suggested release posture:

- deterministic router implementation must match its own reference exactly or within a margin that never changes selected IDs on golden fixtures;
- cross-format candidate quality reports may allow top-8 changes, but held-out aggregate and worst-case thresholds must be frozen.

## Teacher-forcing thresholds

Prefer paired deltas relative to QAT BF16 and official Q4_0:

- NLL relative/absolute delta;
- KL distribution;
- top-k rank agreement;
- reference-token rank;
- low-margin token flip rate.

Calibrate thresholds on development data and validate on held-out data.

## Task thresholds

For each category:

- define minimum sample count;
- define paired score;
- define acceptable regression;
- define severe failure override.

A candidate cannot pass only because another category improves.

## Memory thresholds

These are hard operational thresholds, not statistical:

```text
weights target ≤ 14,100 MiB
hard stop > 14,300 MiB
32K margin ≥ 700 MiB
Base-model 64K margin ≥ 400 MiB if supported; MTP remains ≥ 500 MiB
token-loop allocation = false
fallback/offload = false
```

## Performance thresholds

A release claim requires:

- final candidate faster than official Q4_0 in median prefill and decode;
- controlled 3/10 runs;
- confidence intervals and raw data;
- no quality/memory regression.

For small changes:

- non-overlapping confidence intervals are preferred;
- otherwise characterize as neutral/small/uncertain;
- profile evidence must support causal attribution.

## Determinism

Deterministic mode threshold:

```text
100% repeated output checksum identity
100% selected expert checksum identity where captured
```

Any failure is a bug, not a tolerated percentage.

## Threshold changes

Changing a frozen threshold after held-out results requires:

- decision record;
- explanation;
- new held-out split or explicit reclassification as development;
- rerun of all candidates.
