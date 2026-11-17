# 12B D3 exact/default and experimental fast math — 2026-11-17

The owner explicitly authorizes two paths: retain the current exact-Ordinary
D3 path and add a visibly experimental faster D3 path whose rounding/math and
output IDs may differ, with the first objective approaching vLLM performance.
This supersedes the diagnosis-only/no-numerical-change boundary for the new
opt-in experimental path only. Default exact behavior remains unchanged.

Start with measured global-attention gap (3.256 vs0.827ms/group). Use an offline,
project-compiled SM120 FP8 Tensor Core attention derivative: FP8-rounded queries
and softmax tiles, FP32 online accumulation, original FP8 K/V and BF16 scales,
causal four-query/16-head/global-H512 geometry. No weight-format/cache layout,
sampling, output length, model or host-offload change. Keep exact variant as
startup default and fail visibly for unsupported fast geometry; report changed
operators and actual output differences. New compiled artifact embeds ready
SM120 cubin bytes with source/compiler/hash provenance; never JIT or load files
in recurring inference. Load module/resources before graph capture; use existing
reserved workspace, full conditional D3 graph and no per-token allocations.

Validate operator math/finite/causal/memory boundaries and sanitizers, graph
replay/dispatch, unchanged exact outputs and fast determinism, VRAM and model
quality limitations. Compare with one warmup plus three primary requests for
each needed final cohort; profiler rates stay diagnostic. No26B changes, no
commit/push is authorized for this experiment yet.

Implemented first slice: an offline embedded TMA/FP8 global-attention kernel,
fixed to capacity17519 on SM120, plus cached four-row norm/quant boundaries
ported from the accepted Ordinary math. These boundaries preserve their
reduction/rounding/packing exactly; only the global attention changes math.
Exact remains default; `GEM16_12B_D3_ATTENTION=fp8` selects the experiment,
`exact` or no variable selects the original path. Unsupported geometry fails.

One warmup plus three final measurements yields100.31tok/s exact and107.34tok/s
experimental (+7.01%); vLLM's fresh D3 reference is117.13tok/s. All1135 exact
IDs match the previous native Ordinary/D3 stream; experimental is deterministic
but first differs at index46. This first slice is complete locally, with no
claim of quality parity or attainment of117tok/s. Full evidence:
[`d3-fast-attention-2026-11-17.md`](../../evidence/gemma4_12b/d3-fast-attention-2026-11-17.md).
