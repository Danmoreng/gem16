# Experimental 12B D3 follow-up — 2026-11-17

The owner requests continuing from107.34tok/s toward vLLM117.13, with changed
arithmetic allowed in the explicit experimental path and exact default retained.
This supersedes the completed-first-slice boundary only for measured follow-up
candidates. No commit/push or other-model tuning is authorized. General answer
quality is a separate subsequent evaluation, not established by token equality
or by another engine using a similar precision.

First bounded candidate: four-row post-attention normalization/residual/pre-MLP
normalization/FP4 packing. Actual final profile costs0.754ms/group versus vLLM's
smaller fused norm/quant kernels. Test128-thread paired values/packed conversions
and a warp-first RMS tree; this explicitly changes FP32 reduction order, keeping
BF16 intermediate boundaries, block16 FP4 scales, weights and layout. Use actual
model norm weights plus zero/general/large/alias cases, finite/error/guard and
scoped sanitizers, then one warmup plus three full-model samples if valid.
Reject non-winning candidates and preserve their evidence; never overwrite the
107.34 baseline. No new default exact changes, resident weight copies, runtime
allocations/JIT or timing-boundary changes.

The paired candidate reaches110.89tok/s. Reusing the existing global cubin at
one query row for Assistant reaches111.58tok/s; the latter preserves the paired
candidate's1135 output IDs but changes proposals. Both remain explicit fast
mode only. Next bounded candidate is local Target FP8 TensorCore attention:
1024-slot native circular history plus4 speculative rows, QH16/KVH8/H256,
unchanged row-specific causal/window masks and cache commit/rollback. Check
window fill/crossing/wrap/end, finite/uniform/error/guard/graph and sanitizers,
then one warmup plus three full requests only if valid. Changed local Q/P
rounding is disclosed; default exact local kernel stays intact.

After the speed candidate closes, run a small quality smoke screen using the
existing pinned sgl-eval harness: first32 cached GSM8K test questions, one greedy
answer/question,512-token cap, no thinking, same model/server/context17519/D3,
exact mode versus final experimental mode. This is not a throughput cohort or
proof of general quality equivalence; retain every answer/score and disclose
truncations. Keep the default exact path regardless of this small screen.

Complete locally: selected32-token local tiles improve the final point estimate
to112.100752tok/s (+4.438557% over107.336558);4.488408% more would reach vLLM.
Exact1135 output/1311 draft IDs match. The quality screen scores29/32 exact and
31/32 experimental, with all unscored answers limited by512tokens; no general
quality equivalence is established. Runtime sources, original measurements,
qualifications and compressed traces remain preserved; no commit/push made.
See [final evidence](../../evidence/gemma4_12b/d3-fast-followup-2026-11-17.md).
No further automatic kernel search or larger quality campaign is pending.
