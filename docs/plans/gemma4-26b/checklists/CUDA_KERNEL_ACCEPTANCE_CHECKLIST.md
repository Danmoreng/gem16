# CUDA kernel acceptance checklist

## Semantics

- [ ] Inputs/outputs/shapes documented.
- [ ] Quantization and cast boundaries documented.
- [ ] Accumulation/reduction order documented.
- [ ] Reference implementation independent.
- [ ] Dynamic control and fixed-address ownership clear.

## Correctness

- [ ] Synthetic boundary fixtures.
- [ ] Real-shape fixtures.
- [ ] Real checkpoint activation comparison.
- [ ] Determinism repeated.
- [ ] NaN/Inf handling.
- [ ] Partial/tail shapes.
- [ ] All layer types/shapes covered.

## Safety

- [ ] `compute-sanitizer memcheck`.
- [ ] `racecheck`.
- [ ] `initcheck`.
- [ ] No out-of-range expert/token IDs.
- [ ] No uninitialized padding.
- [ ] Graph capture/replay safe.
- [ ] Failure path checks launch errors.

## Resources

- [ ] Registers recorded.
- [ ] Local memory/spills recorded.
- [ ] Shared memory recorded.
- [ ] Occupancy recorded.
- [ ] Workspace accounted.
- [ ] No token-loop allocation.
- [ ] No persistent duplicate weights.

## Native dispatch

- [ ] Built for SM120/SM120a as required.
- [ ] SASS contains intended instruction.
- [ ] Runtime trace proves kernel executed.
- [ ] No fallback count.
- [ ] Capability output accurate.

## Performance

- [ ] Parent and candidate adjacent.
- [ ] Microbenchmark 3/10 or justified screen.
- [ ] End-to-end benchmark.
- [ ] Memory delta.
- [ ] Power/clocks/thermals controlled.
- [ ] Confidence intervals.
- [ ] Rejected candidates recorded.

## Integration

- [ ] Whole-layer/full-model comparison.
- [ ] Greedy and sampling where affected.
- [ ] Resident session/reset/cancellation.
- [ ] 12B regression.
- [ ] Documentation/ledger.
