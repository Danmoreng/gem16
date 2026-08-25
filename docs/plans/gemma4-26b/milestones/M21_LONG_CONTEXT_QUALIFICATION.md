# M21 — 32K, 64K and maximum safe base context

Status: accepted 2026-08-25; `base_max_context=98,304`, first reproducible rejection at 102,400
Class: qualification; follows the joint clean candidate freeze and precedes M20 execution

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
The compact reconciler binds every result to the compiled artifact lock, toolchain lock, clean source revision, M20
benchmark binary and separately hashed context-driver binary. M20 rejects an M21 envelope if the artifact,
toolchain, native `src`/`include`/CMake trees or benchmark binary changes; a later evidence-harness-only commit is
reported but does not invalidate an otherwise byte-identical native candidate.

## Exit gate

- [x] 32K is execution-qualified for the technical M23 target with 1,122,828,288 bytes free.
- [x] 64K passes twice with 778,895,360 bytes free.
- [x] `base_max_context=98,304` passes twice with 434,962,432 bytes free; 102,400 is rejected twice.
- [x] No context claim relies on reduced hidden safety margin; the 400 MiB rule is unchanged.

Compact evidence is `artifacts/m21/acceptance.json`.
