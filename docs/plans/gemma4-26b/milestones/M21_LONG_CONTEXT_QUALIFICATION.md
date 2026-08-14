# M21 — 32K, 64K and maximum safe base context

Status: blocked by frozen M17 artifact
Class: qualification; may run alongside M19/M20

Normative inputs: [Memory arena](../specs/MEMORY_ARENA_SPEC.md), [Benchmark matrix](../specs/BENCHMARK_MATRIX.md).

## Outcome

Qualify real execution at 32K, attempt 64K and determine the largest safely supported base context.

## Sequence

1. confirm 32K with at least 700 MiB direct free margin;
2. run base-model 64K with at least 400 MiB margin;
3. if 64K passes, probe upward; if it fails, search between 32K and 64K;
4. choose the largest tested base-model context satisfying correctness, allocation and 400 MiB margin;
5. record the limiting region and selected prompt chunk.

Use real prefill plus decode, ring wrap/global extent tests and continuous telemetry. Allocation-only success is not support.

## Exit gate

- [ ] 32K is release-qualified.
- [ ] 64K has an explicit pass/fail result.
- [ ] `base_max_context` is measured and reproducible.
- [ ] No context claim relies on reduced hidden safety margin.
