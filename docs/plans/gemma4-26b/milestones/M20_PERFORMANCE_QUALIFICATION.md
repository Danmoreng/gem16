# M20 — Controlled performance qualification

Status: accepted 2026-08-25; retained medians 6,572.809 prompt and 150.615 ordinary-decode tok/s
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

The formal three-warm-up/ten-retained row passes both hard targets and the non-blocking stretch with deterministic
`c750d0…` output. Compact evidence is `artifacts/m20/acceptance.json`. The external
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
- require the native M21 acceptance envelope from the identical artifact, toolchain lock, unchanged native
  `src`/`include`/CMake source trees and M20 benchmark-binary hash; the distinct M21 context-driver binary is also
  hashed independently. Evidence-harness-only follow-up commits are allowed only while those native trees and the
  binary remain identical;
- use only the bounded promotion row above for this qualification. Full M19 remains owner-deferred and is reported
  but does not block this technical performance milestone.

## Exit gate

- [x] No fallback/offload/prompt-cache asymmetry exists.
- [x] Native paths and instruction evidence are recorded.
- [x] Headline claims use retained distributions and honest boundaries.
- [x] Retained median prompt throughput is at least 6,000 token/s.
- [x] Retained median ordinary decode throughput is at least 150 token/s with MTP/speculative decode disabled.
- [x] The 6,500 prompt-token/s stretch outcome is reported separately as pass.
- [x] Performance does not invalidate deferred M19 policy or accepted M21 memory.
