# Ordinary local Attention and FP8 O

The owner authorizes two bounded experiments on the qualified N1 baseline
`d1b6a3c66e1f299fa02e74c25bc30cf0300c4db2`. Start with the Ordinary local
Attention split and merge. Finish that decision before measuring an O
candidate against the resulting parent. A matched-token Ordinary replay
identified these two operators as candidates for measurement.

1. Capture exact post-RoPE Q, physical FP8 K/V, BF16 scales, ring controls
   and consumed outputs from real local layers. Match effective windows
   and measure split plus merge together. Test one asynchronous V staging
   variant. It may overlap V loads with QK and softmax while keeping the
   original per-dimension accumulation, reduction and merge order.
2. Capture actual FP8 O activations, scales and output from both local
   and global layers. Verify the original resident weights and compare one
   fresh geometry or scheduling candidate that keeps each Split-K2 half
   and final addition exact. Earlier O weight-layout and generic prefetch
   experiments remain closed.

Require exact final outputs and relevant intermediate state, bounds and
sanitizer checks, SM120a/RDC dispatch and resource evidence and a positive
operator lower 95% bound. The operator screens target at least 40 µs
per Ordinary token for local Attention and 30 µs for O. Weight local and
global samples by their layer counts. Preserve early, middle, late and
wrap cases. Confirm cache state and disclose any omitted surrounding work.
For local Attention, apply the threshold to both warm and evicted samples.
For O, use evicted weights as the primary screen because the 30 O matrices
exceed L2. Record warm timings as a separate diagnostic.

A passing operator screen permits one narrow runtime trial. Require the
original 16K 1,135-output workload, actual dispatch, full token parity
and an unprofiled ABBA result with positive lower 95% bounds in both
orders and at least +0.5% pooled Ordinary throughput. The D2 margin is
−0.20% at the lower 95% bound in each order. Protect 12B Unified, 26B
Compact Vision and internal 26B NVFP4, preserve prefill and TTFT, and
check cache/session boundaries and peak VRAM. Keep only a fully qualified
candidate. Record all results and the README locally.
For prefill throughput and inverse TTFT, the lower 95% bound must exceed
−1.0% in both orders.

Both experiments are complete. The local attention split was retained
after the operator, Ordinary, D2 and protected-profile checks. The
two-tile FP8 O geometry failed its operator gate and was excluded from
production. See the [completed evidence](../../evidence/gemma4_26b/local-attention-o-2026-11-05.md).
