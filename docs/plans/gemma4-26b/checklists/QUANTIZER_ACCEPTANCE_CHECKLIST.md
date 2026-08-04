# Quantizer acceptance checklist

Applies separately to FP8, NVFP4 and Q4_0.

## Contract

- [ ] Mathematical reconstruction equation written.
- [ ] Block/group size fixed.
- [ ] Scale dtype and direction (multiplier/divisor) fixed.
- [ ] Value codec and nibble order fixed.
- [ ] Rounding and ties fixed.
- [ ] Zero block/row behavior fixed.
- [ ] NaN/Inf behavior fixed.
- [ ] Version/profile name assigned.

## Reference codec

- [ ] Independent host encoder.
- [ ] Independent host decoder.
- [ ] Boundary/tie/saturation fixtures.
- [ ] Endianness fixtures.
- [ ] Exact known bytes.
- [ ] No GPU intrinsic as normative encoder.

## Compiler

- [ ] Source shape/dtype validated.
- [ ] Axis split/transpose explicit.
- [ ] Streaming/bounded memory.
- [ ] Deterministic repeated hashes.
- [ ] Resume verifies completed work.
- [ ] Per-tensor provenance.
- [ ] Full byte accounting.

## Statistics

- [ ] Relative L2.
- [ ] Cosine.
- [ ] Max absolute error.
- [ ] SQNR.
- [ ] Saturation.
- [ ] Scale distribution.
- [ ] Code histogram.
- [ ] Zero blocks/rows.
- [ ] Statistics cover all tensors or are labeled sampled.

## Operator evidence

- [ ] Synthetic output comparison.
- [ ] Real-activation output comparison.
- [ ] Real-shape CUDA reference comparison.
- [ ] Native path comparison.
- [ ] Required BF16 boundaries.
- [ ] No hidden fallback.

## Model evidence

- [ ] Teacher-forced logits.
- [ ] Router drift when relevant.
- [ ] Full-model deterministic generation.
- [ ] Held-out quality suite.
- [ ] Worst regressions inspected.

## External comparison

- [ ] Ordinary own conversion compared with Unsloth for FP8/NVFP4.
- [ ] QAT own conversion compared with official Q4_0 where relevant.
- [ ] Master-weight and quantizer effects not conflated.

## Acceptance

- [ ] All correctness thresholds frozen and passed.
- [ ] Artifact lock/reproducibility passed.
- [ ] Decision record names retained/rejected algorithm.
