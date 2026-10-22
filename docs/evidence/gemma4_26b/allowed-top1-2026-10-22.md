# Allowed-top1 selection shortcut — 2026-10-22

Accepted bounded Linux candidate: ordinary decode improves reproducibly by **4.45%**
in both AB/BA comparison pairs, with exact output-token identity, unchanged sampled
peak VRAM, and no material D2 change. This implements the first candidate identified
by the [decode audit](trellis35-decode-cost-audit-2026-10-22.md).

Record: `artifacts/trellis35/allowed-top1-2026-10-22.json`.
Raw root: `benchmarks/results/2026-10-22/c0a807af/allowed-top1/`.
The record preserves all six cohorts, ten samples per cohort, full token identities,
telemetry summaries, source/binary hashes, exact commands, diagnostic scripts and raw hashes.

## Change and numerical boundary

The non-sampling `Gemma4Moe26BReferenceEngine::SelectToken()` still calls `Prediction()`
first, preserving its router/finite checks, result and self-feed update. It then searches
only the active host suppression span. If the global post-softcap winner is allowed,
it returns that winner directly; otherwise the existing GPU masked-argmax/readback
fallback runs unchanged. The fixed host array's zero padding is outside the span.

Removing other candidates cannot change an allowed global winner under the same FP32
post-softcap values and lowest-ID tie rule. No precision, GPU numerical kernel, logits,
sampling/penalty branch, arena, KV, context or inference timing boundary changes.

The additional host unit test compares all 31 non-total suppression subsets against an
independent masked reduction for six vectors, including ties, signed zero and softcap
saturation. Token 0 remains allowed when absent from the active prefix of padded storage.
The new model-qualified selection driver checks six actual masks, including the original
mask, blocked top-1, duplicate IDs and 16 suppressed maxima. It checks full raw-logit
identity, independent masked argmax, repeated selection, `Prediction()` before/after,
next-token self-feed/fallback behavior and sampled/penalty reset determinism.

## Primary comparison

Parent source is `c0a807afcbcd2528f304bada513afd11d1e9c979`; the exact saved audit binary
is reused. The candidate record captures the uncommitted source delta and binary hash.
Both binaries remain in the raw roots for independent reproduction.

RTX 5080 Laptop, driver 615.71.09 / CUDA 13.4.92; same verified Trellis target and
assistant, max-power and active Dynamic Boost, firmware-managed clocks. Each cohort
starts at <=50 C. Workload: the same 16,384 prompt IDs, 1,135 outputs / 1,134 measured
decode intervals, context 17,519, FP8 KV, greedy batch one, suppression
`[1,50,106,258882,258883]`, three warmups and ten retained runs. Native process lifecycle
and harness timing boundaries are unchanged. Profiling/sanitizer/fault-injection runs
are separate from primary measurements.

| Cohort in execution order | Median tok/s | Mean 95% CI tok/s | Peak sampled VRAM MiB |
|---|---:|---:|---:|
| Parent A | 128.989 | 128.966–129.004 | 12455 |
| Candidate B | 134.730 | 134.674–134.894 | 12455 |
| Candidate B2 | 134.711 | 134.640–134.859 | 12455 |
| Parent A2 | 128.967 | 128.961–128.990 | 12455 |
| Parent D2 | 172.344 | 172.293–172.482 | 12725 |
| Candidate D2 | 172.331 | 172.198–172.460 | 12725 |

Ordinary gains are **4.4505%** and **4.4539%**. D2 differs by -0.0074% at the medians
with overlapping intervals; this is an adjacent non-regression check, not a D2 speed claim.
All 60 retained runs produce exactly the same 1,135 native output IDs, zero fallbacks
and no reported token-loop allocations.

High-utilization telemetry clock medians are 2655 / 2625 / 2617 / 2655 MHz in ordinary
AB/BA order and 2625 MHz for both D2 cohorts. These telemetry samples include prefill
and warmup, not isolated decode clocks. No higher-clock explanation is claimed for the
candidate gain. Device-wide 200 ms VRAM sampling is not an instantaneous allocator peak.
The pinned llama ordinary audit reference remains about 138.31 tok/s; this patch alone
does not reach it and introduces no fresh cross-engine quality equivalence claim.

## Dispatch, errors and regression evidence

- In the eight-output decode-only trace, all seven ordinary selection ranges contain
  one 16-byte status DtoH and zero extra `ArgmaxKernel` calls. The parent audit had
  two transfers and an extra masked-argmax launch in each corresponding range.
- A separate changed-mask one-output trace blocks 61684, selects 119308, and retains
  one masked argmax plus the original 16-byte and 4-byte DtoH transfers.
- Explicit host-copy fault injection clears either finite or router-finite status
  after the existing GPU transfer. Both runs fail visibly with the expected non-finite
  error before the allowed-winner shortcut. No GPU operation is added by the injector.
- `ctest --test-dir build/Linux/blackwell-release --output-on-failure -R
  '^gem16-(unit|responses-allocation)$'` passes, including the new exhaustive host cases.
- Both Trellis35 and internal NVFP4 runs of `gem16-26b-selection-driver --model DIR`
  pass; absent fixture environment is a CTest skip, not qualified evidence.
- The four focused conditional-graph/Trellis/attention/NVFP4-consumption CTests and
  `^gem16-cuda$` pass. Explicit-fixture `^gem16-12b-m22-product$` passes.
- Native ordinary/D2 8/64 probes match the frozen fixture. Protected 12B and internal
  NVFP4 ordinary/D2 64-token vectors match exactly, with no fallback/loop allocation.
- Memcheck conditional graph, streaming-ring control and full-model Trellis D2
  (original 16K prompt, full context, eight outputs) each report zero errors.
  Eighteen benchmark-harness Python tests pass.

No new Windows, media, broad-quality or everyday-context release qualification is claimed.
The earlier D2 graph/ring fixes remain unchanged. This evidence is additive and preserves
all previous baselines. Later kernel/replay experiments are separate candidates.
