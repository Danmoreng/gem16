# M20 — Controlled performance qualification

Status: in progress; 120.398 tok/s three-run candidate available, formal 3/10 telemetry run remains
Class: qualification; GPU execution follows M21 and consumes its matching context evidence

Normative inputs: [Benchmark matrix](../specs/BENCHMARK_MATRIX.md), [Telemetry artifact](../specs/TELEMETRY_ARTIFACT_SPEC.md).

## Outcome

Measure batch-one prefill, TTFT, ITL and decode on the exact candidate hash under fair boundaries.

## Rules

- correctness sanity before timing;
- three warm-ups and ten retained runs for promotion;
- fixed prompts, output counts, sampling and KV precision;
- record raw runs, clocks, power, thermals, VRAM and actual dispatch;
- compare external Q4_0/Unsloth with explicit format/timing caveats.
- require the native M21 acceptance envelope from the identical artifact, binary and source revision.

## Exit gate

- [ ] No fallback/offload/prompt-cache asymmetry exists.
- [ ] Native paths and instruction evidence are recorded.
- [ ] Headline claims use retained distributions and honest boundaries.
- [ ] Performance does not invalidate M19 quality or M21 memory.
