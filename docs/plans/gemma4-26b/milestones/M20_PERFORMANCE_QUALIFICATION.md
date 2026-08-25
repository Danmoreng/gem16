# M20 — Controlled performance qualification

Status: in progress; current adjacent 16K+64 candidate averages 6,574.164 prompt and 139.054 ordinary-decode tok/s; fixed owner decode target and formal 3/10 telemetry run remain
Class: qualification; GPU execution follows M21 and consumes its matching context evidence

Normative inputs: [Benchmark matrix](../specs/BENCHMARK_MATRIX.md), [Telemetry artifact](../specs/TELEMETRY_ARTIFACT_SPEC.md).

## Outcome

Measure batch-one prefill, TTFT, ITL and ordinary decode on the exact candidate hash under fair boundaries, and pass
the fixed owner targets without changing model semantics.

## Promotion row and targets

- scenario: `wikipedia-real-16k64-greedy`;
- prompt: exactly 16,384 tokens from manifest SHA-256
  `9a5859b979d91fccf71bcbb61aade6372cf2cc3c708e6c47b8b6cfd99f7abd2d`;
- generation: 64 output forwards and 63 timed post-first-token intervals in a 16,448-token context;
- hard retained medians: **prompt throughput >= 6,000 token/s** and **ordinary decode throughput >= 150 token/s**;
- non-blocking competitive stretch: **prompt throughput >= 6,500 token/s**.

The current 6,574.164/139.054 development row passes the prompt target and non-blocking stretch but remains an
adjacent two-run development result rather than formal M20 acceptance. The external
vLLM 0.27.1 community-W4A16 row motivates the targets but differs in checkpoint and prefill timing boundary and is
not the correctness oracle.

## Rules

- correctness sanity before timing;
- three warm-ups and ten retained runs for promotion;
- fixed prompts, output counts, sampling and KV precision;
- native base Target, FP8 KV, CUDA Graph replay, batch one and deterministic greedy;
- MTP/speculative decode, prompt cache, CPU offload, fallback and recurring allocation disabled;
- retain `prompt_ms=first_prefill_launch_to_prefill_sync`,
  `ttft_ms=request_ready_to_first_token_ready`,
  `decode_ms=sum_of_post_first_token_synchronized_intervals`, with model load excluded;
- record raw runs, clocks, power, thermals, VRAM and actual dispatch;
- compare external Q4_0/Unsloth with explicit format/timing caveats.
- require the native M21 acceptance envelope from the identical artifact, binary and source revision.
- align the M20 qualifier with this single bounded promotion row and the owner-deferred M19 policy before execution;
  its older mandatory multi-scenario/M19 gates cannot accept this milestone unchanged.

## Exit gate

- [ ] No fallback/offload/prompt-cache asymmetry exists.
- [ ] Native paths and instruction evidence are recorded.
- [ ] Headline claims use retained distributions and honest boundaries.
- [ ] Retained median prompt throughput is at least 6,000 token/s.
- [ ] Retained median ordinary decode throughput is at least 150 token/s with MTP/speculative decode disabled.
- [ ] The 6,500 prompt-token/s stretch outcome is reported separately as pass or miss.
- [ ] Performance does not invalidate M19 quality or M21 memory.
