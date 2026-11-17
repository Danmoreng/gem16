# 12B MTP arithmetic: optimized default and exact reference

Owner decision2026-11-17: prefer the measured optimized12B D3 arithmetic by
default, and retain the existing exact-reference path as an explicit choice.
This replaces the earlier rule that exact arithmetic must remain the default.
Target acceptance, causal masks, KV commit/rollback, model weights, sampling
controls and output-count/timing contracts are preserved.

## Selecting a mode

Selection happens once at execution-slot initialization, before CUDA capture.

| `GEM16_12B_D3_ATTENTION` | Behavior |
|---|---|
| Unset or `auto` | Prefer optimized for supported12B D3 geometry; otherwise select the established native reference path and report the selection reason. |
| `optimized` | Require optimized; unsupported configurations fail visibly. |
| `fp8` | Compatibility alias for `optimized`. |
| `exact` | Require reference arithmetic, including in optimized-supported geometry. |

Fixed D3 on SM120 with checkpoint FP8 KV supports both greedy and sampled
generation. The optimized slot accepts any integer capacity from 5 to 262144.
Five positions accommodate one prompt token and four verifier rows. Smaller
slots use native arithmetic in automatic mode. The global Target and Assistant
read the same physical cache. Their TMA descriptors use the allocated capacity,
and each query masks positions after its causal endpoint. Local Target attention
retains the 1024-token sliding window. The math selection applies to complete
D3 groups. Short requests can use native partial groups or Ordinary tail steps.

For sampled text chat, start `gem16-chat` or `gem16-server` with the pinned
Target and Assistant, `--mtp-draft-tokens 3 --kv-cache fp8 --max-context 262144`.
Choose any supported capacity that fits the execution slot. Both applications
load the checkpoint's temperature 1.0, top-k 64 and top-p 0.95 by default.
For the raw token-ID runner, supply `--sample --temperature 1 --top-k 64
--top-p 0.95` explicitly. Use `--greedy` to select greedy decoding.
Native Studio offers **D3 (fixed)** for 12B. Its default profile uses D2.

Set `GEM16_12B_D3_ATTENTION=exact` before startup for reference arithmetic.
The sampler applies the requested controls to the selected Target logits for
each verification row. Commit advances the RNG step and history by the emitted
token count. For Ordinary same-seed token identity checks, select reference
arithmetic explicitly.

Inference JSON and `/health` disclose `mtp_math_mode`, `mtp_math_policy`,
`mtp_math_selection_reason` and `mtp_math_quality_state`. The optimized attention
label is `optimized_d3_target_and_assistant_fp8_tensor_core_attention`.
Server startup logs the selected mode/reason. Existing model-profile qualification
fields describe the locked model profile; the separate math-quality field
reports `initial_quality_screen_only` for optimized mode.

## Correctness boundary

Reference preserves native arithmetic and the qualified greedy Ordinary/MTP
token-identity regressions. The latest full check retains1135 output IDs and
1311 proposed draft IDs. “Exact” does not prove universal bit identity for every
prompt, hardware, sampling configuration or seed.

Optimized changes Target/Assistant FP8 Q/probability rounding, Tensor Core
attention/reduction order and paired norm reductions. Greedy outputs may differ
from Ordinary/reference. These numerical changes do not permit accepting a
draft rejected by the selected Target verifier. No delayed-rejection, relaxed
acceptance or other deliberately lossy verification algorithm is introduced.
Legacy telemetry `batched_exact_target` names the verification algorithm, not
Ordinary-token identity for optimized math.

Changed Target numerics mean **distribution identity to reference is not
guaranteed**. An unchanged acceptance algorithm and good task scores do not
prove that guarantee. This is a measured performance choice with disclosed
numerical/quality limits; it does not depend on a claim that all external
inference engines make the same tradeoff.

## Evidence and limits

The [sampled context validation](../artifacts/gemma4_12b/d3-sampled-context-2026-11-17.json)
records one Linux SM120 run with 262080 synthetic prompt IDs and 64 sampled
output IDs at capacity 262144. Optimized fixed D3 executed 26 conditional CUDA
graph groups and reached a sampled whole-device peak of 14421 MiB at 200 ms
sampling intervals. The record also covers 19 shorter model runs, scoped
operator and CUDA sanitizer checks, and four sampled HTTP chat turns.
Two optimized runs with the same seed and output limit reproduced all IDs,
while the resident D3 audit found that a two-token limit using the native tail
can produce different IDs from the first two tokens of a longer run with the
same seed.

On the pinned12B16K workload, reference measures100.31tok/s and optimized
112.10tok/s; fresh vLLM D3 is117.13tok/s. Primary cohorts use one warmup and
three retained requests. Dynamic Boost and changed output/acceptance are
disclosed; this is not a universal speedup claim.

The first32 pinned GSM8K questions score29/32 reference and31/32 optimized.
All unscored answers reach the shared512-token cap, and31 answer texts differ.
This small screen does not establish general quality equivalence, coding/tool
quality, long-context factuality, sampled-distribution identity or Windows/media
qualification. Operator numerical/causal/guard/graph and scoped CUDA sanitizer
checks pass; no full-model sanitizer campaign is implied.

Detailed [diagnosis](evidence/gemma4_12b/d3-path-diagnosis-2026-11-17.md),
[first slice](evidence/gemma4_12b/d3-fast-attention-2026-11-17.md) and
[follow-up/quality evidence](evidence/gemma4_12b/d3-fast-followup-2026-11-17.md)
preserve original samples and source/binary/compiler/model identities. Historical
records describe their original opt-in/default policy and remain evidence.
New runs record actual math mode; explicitly set `exact` when reproducing the
former exact-default configuration.

The owner authorizes committing and pushing implementation and evidence.
No release or additional kernel optimization is part of this policy change.
The [default-selection validation](evidence/gemma4_12b/d3-default-policy-2026-11-17.md)
reaches112.06tok/s with unchanged optimized IDs and confirms the reference and
visible selection/error behavior.
