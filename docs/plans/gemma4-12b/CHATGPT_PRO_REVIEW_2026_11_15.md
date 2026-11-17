> Historical task contract, closed or superseded. Current disposition is in
> [the archive index](../../evidence/benchmark-archive-2026-11-17.md).
> This file does not authorize a new experiment.

# Review request: Gemma 4 12B Ordinary, D2 and Prefill

Please review the attached current-worktree snapshot and measured evidence.
Primary objective: make GemSixteen's **12B Ordinary and fixed-D2 Decode**
competitive with or faster than llama.cpp and vLLM, while preserving outputs
and the existing numerical/runtime contracts. Prefill is a secondary objective.
Propose a new, evidence-based optimization plan; do not implement changes.

This archive was requested after the owner cleared the earlier experiment
Goal. It is a review/export task, not authorization to resume optimization,
commit, push, change quantization or run a new GPU campaign.

## Read first

1. `review/CURRENT_STATE.md`: authoritative snapshot status, measured table,
   rejected versus unfinished experiments, timing and version distinctions.
2. `review/BENCHMARK_INDEX.json`: exact samples, source paths, output hashes,
   revisions, commands and memory evidence. Follow its links to raw text.
3. `review/SOURCE_GUIDE.md` and `reference/REFERENCE_PROVENANCE.json`:
   our hot paths, two measured llama.cpp source revisions, installed vLLM
   Python/Triton code, matching upstream CUDA kernels, FlashInfer/CUTLASS.
4. `docs/evidence/gemma4_12b/` and the original raw experiment records.
   Older in-progress prose is not proof that a later stage was completed.

## Measured questions to answer

- Ordinary's native GPU budget is dominated by NVFP4 projections (~34%), FP8
  QKV/O (~25.5%) and the tied BF16 vocabulary head (~12.3%). D2 shifts cost
  toward NVFP4 short-batch Gate/Up and Down (~17.6% each), FP8 T3 (~14.6%),
  the target head (~10.7%), attention and the Assistant. CPU gaps are small
  in the available native traces. Confirm the attribution before prioritizing.
- Compare the **actually observed dispatch** with reference source. vLLM
  logs identify `FlashInferCutlassNvFp4LinearKernel` and `TRITON_ATTN`;
  compiler and CUDA Graphs remain enabled. Available alternative source does
  not establish its execution. Inspect small-M tiling, weight reuse, split-K,
  asynchronous loads, FP8 packing/scales, normalization boundaries, attention,
  target/Assistant head and D2 state/graph handling.
- Which reference techniques can transfer to our exact 12B shapes without
  changing the reduction order, BF16 rounding, quantization scales, token
  selection, cache semantics or model behavior? Classify every proposed
  numerical/layout change explicitly instead of assuming equivalence.
- Explain how much of the reference gap could come from format/KV/backend
  differences. llama.cpp's converted Q8/NVFP4 GGUF is not identical to the
  native/vLLM mixed FP8/NVFP4 source; native and reference output streams can
  differ. Do not attribute the whole gap either to quantization or to kernels
  without evidence. Exact output parity is required between our candidate and
  its measured native parent, not falsely asserted across engines.
- Check Assistant cost separately from acceptance. Existing native and vLLM
  D2 acceptance rates are close and do not alone explain their throughput gap.
  The actual Assistant config has `use_ordered_embeddings=false`; the mere
  presence of a sparse/centroid head implementation is not evidence it ran.
- For Prefill, use the fully confirmed chunk curve and independent logical
  batch/microbatch follow-up. Native improves toward 8192; llama.cpp peaks
  around 1024; vLLM around 4096. Large logical batch does not rescue oversized
  physical microbatches. Explain possible operator/traffic causes as hypotheses
  unless profiling proves them. Do not revive the old unequal-chunk fairness
  objection as if the control measurements were still missing.

## Previously tested ideas: do not simply repeat them

The five-step campaign did not retain any optimization runtime patch:

| Step | What was tested | Actual state |
|---|---|---|
| 1 | One-M16 NVFP4 D2 Gate/Up tile | Operator +15.30% warm / +22.00% evicted; full D2 +0.188335%, rejected below the predeclared 0.5% minimum; runtime restored |
| 2 | Packed FP8 T3 QKV/O | +30.37% warm / -5.46% evicted, bit-equal actual cases; rejected before model integration |
| 3 | Two-way ordered K-loop unroll for Ordinary QKV | Operator about +2.31% warm / +2.23% evicted; full Ordinary +0.1436075%, rejected below 0.5%; runtime restored |
| 4 | Round/post-attention norm/residual/pre-MLP quantization fusion | Offline only: bit-equal cases, +34.925% warm, +0.417% fully evicted with a 95% interval crossing zero; no production/E2E qualification |
| 5 | Assistant BF16 GEMV exact-tree reduction | Derivative/observer/replay prepared; **not GPU-tested**, no E2E evidence |

Step 4's eviction condition may not represent a freshly produced O-projection
input. Reviewing that assumption is permitted in the plan, but do not silently
reclassify its old result as a passing screen. Raw unrounded O-projection
boundaries still need alignment with the normalization fixtures. Earlier 12B
T3 WMMA target-head and warp/overlap failures are preserved in historical
evidence; require a materially different premise before repeating them.

## Required deliverable

Provide a ranked plan of **five concrete experiments**, prioritizing Decode
and D2. For each, give:

- Exact files/functions/shape-specific dispatch condition in our code, and the
  reference file/function that motivates the idea. Cite archive paths/lines.
- Measured cost share, proposed mechanism and an Amdahl-style plausible E2E
  bound. Separate demonstrated facts, source deductions and hypotheses.
- The numerical/format/memory risks, what must remain exact, and the minimal
  actual-input/operator/dispatch/sanitizer and unprofiled model measurements
  that could accept or reject it. Keep cached and evicted screens separate.
- One bounded first implementation, explicit stop criteria, and which existing
  rejected or untested idea it differs from. Avoid another broad parameter
  sweep or infrastructure rewrite before a concrete performance hypothesis.

Say if the best next step is completing the unfinished Step 4/5 rather than a
new idea, and explain why. Do not invent timings or missing profile evidence.
If a useful source or capture is omitted, identify it precisely. Please put
the complete plan in the response text as well as any optional download, so
the work order remains usable if generated file links expire.

## Non-negotiable constraints

Keep the exact model/checkpoint/precision, token IDs, output length, greedy
sampling/suppression/template, context, batch/sequence controls and cache
semantics for candidate/parent qualification. No CPU/expert weight offload,
second persistent weight copy, recurring allocation, filesystem/JIT/repack in
native inference, tolerance relaxation or hidden fallback. Preserve 12B media,
26B Compact Vision and internal NVFP4 regression paths; keep specialized hot
paths specialized. Primary claims need actual GPU dispatch/resource facts,
correctness, raw unprofiled samples, peak memory and code/toolchain identities.

Previously declared model gates were at least 0.5% pooled primary throughput
gain with positive 95% lower bounds in both orders, secondary non-inferiority
of -0.20%, and Prefill/inverse-TTFT margin -1%. If proposing a different gate,
state why; it would need a new owner decision before implementation. A smaller
positive result is still a measured gain, but failed the earlier promotion rule.
