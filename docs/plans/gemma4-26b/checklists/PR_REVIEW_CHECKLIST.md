# Milestone change-set and final pull-request review checklist

## Scope and design

- [ ] Review record names the milestone and implementation slice.
- [ ] No unrelated refactor/UI/format churn.
- [ ] Drift from the anchored plan is documented.
- [ ] Public/internal API changes are intentional.
- [ ] 12B and 26B paths remain clearly separated.
- [ ] New model assumptions are validated, not guessed.

## Correctness

- [ ] Host oracle or trusted reference exists.
- [ ] Quantization/rounding/scale semantics documented.
- [ ] Tensor axes/shapes/names validated.
- [ ] Router tie and reduction order deterministic.
- [ ] K=V projection does not alias final K/V cache.
- [ ] Output softcap/suppression/tie behavior preserved.
- [ ] No NaN/Inf is hidden or repaired silently.

## Memory and ownership

- [ ] Checked arithmetic and alignment.
- [ ] One persistent device layout per weight.
- [ ] Tied head has one allocation.
- [ ] No token-loop allocation.
- [ ] Workspace bounded by chunk, not context×experts.
- [ ] Weight sharing and session isolation correct.
- [ ] Failure cleanup tested.

## Native path

- [ ] Capability and runtime dispatch are visible.
- [ ] No silent BF16/reference fallback.
- [ ] Required SASS/instruction evidence exists where claimed.
- [ ] CUDA Graph addresses/lifetimes are safe.
- [ ] No local-memory spill or race overlooked.

## Tests

- [ ] New unit tests cover success/failure.
- [ ] Real checkpoint fixture/probe included.
- [ ] CUDA tests and sanitizers run where relevant.
- [ ] Determinism tested.
- [ ] 12B regressions run.
- [ ] Commands and exact results attached.

## Quality/performance

- [ ] Quality gate precedes performance promotion.
- [ ] Benchmark inputs and timing boundaries fair.
- [ ] Warmups/repetitions/raw runs retained.
- [ ] Memory/power/clock/thermal telemetry included.
- [ ] Negative/rejected results documented.
- [ ] Claims match evidence and confidence intervals.

## Documentation

- [ ] Decision record updated if policy/arithmetic changed.
- [ ] Correctness/memory/benchmark docs updated.
- [ ] Performance ledger updated for optimization.
- [ ] Capability/user docs accurate.
- [ ] Exit criteria marked pass/fail.

## Security/provenance

- [ ] Immutable source/artifact hashes.
- [ ] No model repository code execution.
- [ ] No tokens/secrets/private paths in artifacts.
- [ ] License/attribution reviewed for copied/reference code.
