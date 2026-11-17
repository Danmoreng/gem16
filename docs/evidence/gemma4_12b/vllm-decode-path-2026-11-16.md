# Installed vLLM 0.33.0 Gemma 4 12B decode path

Read-only audit, 2026-11-16, completed before the daily shutdown deadline.
No GPU job, profiler, implementation change or new benchmark was started.
Scope: the installed code and the just-completed MTP D3/D4 runs, not a generic
claim about every vLLM model/backend. Existing source hashes are recorded in
`artifacts/dflash/2026-11-16/vllm-source-manifest.json` (3,231 Python files,
aggregate `4584b0367bdebfec43afbef57dc720478ed85e9e142232275072d250b35be609`).

## What 117 tokens/s means

The D3 cohort reports 117.41 output tok/s. A representative retained request
has 1,335 proposals, 689 accepted drafts and 445 verification groups.
The 1,134 measured output intervals equal 689 + 445: **2.5483 outputs/group**.
Its measured decode time is 9,652.024 ms, or **21.690 ms/group**, including
drafting, target work, sampling and runtime overhead. This is not 117 full
target passes per second. D4 averages 2.7301 outputs/group but its groups
cost more: its final throughput is essentially tied with D3 (117.29).

The historical vLLM D2 request has 501 groups, 633 accepted drafts and
10,052.294 ms decode time: 2.2635 outputs/group and 20.065 ms/group.
The historical gem16 D2 request has 507 verification groups and 12,105.5 ms
decode time. Its transaction/tail accounting differs slightly, so group
division is descriptive rather than a matched operator-cost comparison.
Outputs differ across these engine/depth cohorts; this is not exact parity.

## One steady-state speculative decode cycle

1. The target verifier evaluates the committed input plus the previous draft
   block in one causal multi-row forward. D3 normally needs four query rows;
   D4 normally needs five. Tail/padding handling can change physical rows.
2. Each target layer performs the Gemma-specific residual/norm sequence,
   Q/K/V projections (or Q-only in KV-shared layers), Q/K normalization and
   RoPE, attention over the resident FP8 cache, output projection, and the
   dense gated MLP with its required normalization/rounding boundaries.
   GPU weight residency is maintained; no CPU weight streaming is requested.
3. The target head computes logits for verification positions. The sampler
   accepts the matching draft prefix and returns target-selected replacement
   or bonus outputs. Rejected speculative positions are corrected in request
   state/position/cache metadata before the next cycle. Greedy temperature 0
   is our measured policy; do not generalize its acceptance semantics to
   arbitrary probabilistic sampling.
4. The lightweight official Assistant receives the selected target hidden
   states and token embeddings. Its pre-projection maps to width 1,024;
   all four Q-only Assistant layers execute for **each** autoregressive draft
   token. It reads target-layer 46 local KV and target-layer 47 global KV.
   It has no separate K/V projection or growing long-context KV cache.
   Its own tied head predicts a draft; post-projection provides feedback.
5. At D3 the Assistant is iterated three times, not one parallel diffusion
   forward. The next target batch verifies that block, amortizing expensive
   target weights across multiple useful outputs.

This is the steady-state cycle; target verification of the previous proposals
and generation of the next proposals form a pipeline in the runner. It must
not be described as a single indivisible graph containing the entire server.

## Actual graph and attention configuration

The D3 log records `VLLM_COMPILE`, Inductor, `FULL_AND_PIECEWISE`, capture
sizes 1/2/4/8, separate compiled `backbone` and `eagle_head`, and full draft
decode graph captures. The generic design uses full graphs for supported
uniform decode and piecewise graphs for other batches:
[vLLM 0.33.0 graph design](https://docs.vllm.ai/en/v0.33.0/design/cuda_graphs/).

Installed `TargetDependentARSpeculator` maintains separate draft-prefill and
draft-decode graph managers. Remaining draft steps have two source routes:
`_fused_multi_step_decode` can replay a multi-step full graph when the backend
supports metadata updates; `_multi_step_decode` loops over steps with graph
replay per step. For Gemma, positions/sequence lengths stay fixed while
drafting because the Assistant only reads target KV. Do not claim which
multi-step branch executed from the startup capture lines alone; an exact
current D3 trace/observer is still needed to establish that boundary.

**Our target and Assistant use TRITON_ATTN.** The startup explicitly forces it
because local/global head dimensions differ (256/512). This is not evidence
for FlashAttention/FA4/XQA as the MTP attention backend. FlashInfer appears
in the matrix-multiplication stack; that is a separate dispatch decision.
Target FP8 KV scales are copied correctly to the Assistant's shared-KV
attention modules. The general Assistant design is also documented in the
[official model API](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/gemma4_mtp/).

## Kernel families worth inspecting

- Mixed checkpoint FP8 attention projections and NVFP4 MLP remain distinct.
  `CompressedTensorsW4A4Fp4` uses a selected NVFP4 linear kernel. The measured
  startup logs identify FlashInfer's **CutlassFp4GemmRunner** and cache hits.
  Its CUTLASS route accepts a prequantized activation or performs scaled FP4
  activation quantization, then block-scaled FP4 GEMM. Weight padding/scale
  swizzling is performed in `process_weights_after_loading`, outside inference.
- A preserved **November 15 Ordinary** torch profile actually contains
  `kernel_unified_attention`, SM120 block-scaled CUTLASS kernels,
  `triton_red_fused_add_rms_norm_scaled_fp4_quant_1`, and
  `triton_poi_fused_gelu_mul_scaled_fp4_quant_slice_2`. This is real dispatch
  evidence for the installed stack, but **not** a D3/D4 per-kernel breakdown.
- Startup pass flags such as `fuse_norm_quant=False` do not establish that
  no fusion occurred: Inductor-generated fused kernels are present in that
  older profile. Conversely, do not assume each observed Ordinary fusion also
  executes for every D3/D4 shape without capturing it.

## Source entry points in the installed package

Package root:
`third_party/cache/vllm-0.33.0-env/lib/python3.13/site-packages/vllm/`.

- `v1/worker/gpu/model_runner.py`: `sample` at 1633, rejection sampler at
  1688, `sample_tokens` at 2149 and next-block `speculator.propose` at 2308.
- `v1/worker/gpu/spec_decode/gemma4/speculator.py`: fixed-position Q-only
  policy, forced attention backend, target-KV aliasing and calibrated scales.
- `v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py`: graph
  managers at 89, `propose` at 165, multi-step branch at 354, per-step loop
  at 493 and fused multi-step route immediately afterward.
- `model_executor/models/gemma4_mtp.py`: Assistant layer math, width-1,024
  embedding/head, pre-/post-projections and target-hidden-state feedback.
- `model_executor/models/gemma4.py`: target attention and dense MLP; layer
  forward at 711 preserves Gemma's residual and normalization order.
- `model_executor/layers/quantization/compressed_tensors/schemes/`:
  `compressed_tensors_w4a4_nvfp4.py` and FP8 schemes select linear kernels.
- `model_executor/kernels/linear/nvfp4/flashinfer.py`: CUTLASS preparation
  and quantized GEMM application (class at 186, apply at 223).

Raw current logs/results: `artifacts/dflash/2026-11-16/vllm-mtp-{3,4}.*`.
Earlier actual kernel capture and its deliberately narrow scope:
`benchmarks/results/2026-11-15/58febf4/history-and-12b/actual-dispatch.json`.

## Concrete next steps for November 17

1. Measure the already implemented native 12B D4 path against native D2 at
   the owner's 1 warmup + 3 measured requests. No implementation needed just
   to enable D4. Preserve formats, output IDs/length controls and residency.
2. Capture **one bounded diagnostic D3/D4 decode window** in vLLM, outside
   primary throughput timing. Attribute Assistant, target attention, FP8
   projections, NVFP4 MLP, head/sampler and CPU/GPU gaps separately. Establish
   actual graph boundaries and exact matrix shapes/tactics/instructions.
3. Replay the largest *measured* gap on actual native inputs at the same
   row counts. Candidate areas are multi-row attention, FP8 QKV/O, NVFP4
   layout/tactics, and existing norm/activation/quant fusion boundaries.
   These are hypotheses, not an authorized blanket tuning campaign or
   evidence that any one kernel explains the 117 tok/s.
4. Preserve native BF16 rounding, FP8/NVFP4 scales, KV transaction/rollback,
   tied-weight aliasing and no token-loop allocation/repack. Some analogous
   fusions were already tested/rejected: read their evidence before reopening.
   Native D4 graph chaining is a separate conditional candidate; native D2
   already has a more device-driven chain, so CUDA Graphs alone cannot explain
   vLLM's advantage. Do not promise 117 tok/s from copying a graph setting.

Continuation: no owned jobs remain. Source/log audit only; exact current
D3/D4 kernel timing, instruction-level bottlenecks and the fused-versus-loop
multi-step choice remain unmeasured. No code change or benchmark is pending
in the background. No commit/push/release or timer change.
