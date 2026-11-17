# Experimental 12B D3 follow-up — 2026-11-17

The experimental path now reaches **112.100752tok/s**, up4.438557% from the
previous107.336558. This is95.7044% of the fresh vLLM117.132291 reference;
another4.488408% speedup would match that reference. The exact default remains
unchanged. This is one workload on one GPU, not a general throughput/quality
qualification. No commit or push was made.

## Primary evidence

| Cumulative stage | Three retained samples, tok/s | Median |
|---|---|---:|
| Previous experimental baseline |107.387381,107.336558,107.168171|107.336558|
| Paired post-attention norm/FP4 |111.027346,110.891631,110.738943|110.891631|
| Plus Assistant global FP8 |111.574639,111.584520,111.581226|111.581226|
| Plus local Target FP8 |112.100752,112.178377,112.086348|112.100752|
| Earlier exact default |100.385075,100.307823,100.239550|100.307823|
| Fresh vLLM D3 reference |117.146656,117.132291,117.094626|117.132291|

Each primary cohort has **one warmup plus three retained requests**. The exact
and vLLM rows reuse the earlier same-day primary evidence; they were not rerun
as performance controls. The final native binary additionally passes a separate
full1135-output correctness request: all1135 output IDs and all1311 draft IDs
equal the exact baseline. Its diagnostic rate is excluded from this table.

All primary requests retain batch1,16384 input IDs,1135 outputs,1134 timed
decode intervals, capacity17519, greedy/suppression and FP8 KV. Model/Assistant
identities, native resident weight layout and the post-first-token timing
boundary are unchanged; see the [initial diagnosis](d3-path-diagnosis-2026-11-17.md).
Default prefill remains8192 versus vLLM max-batch1024 under that protocol.
The final112.10 result is11.756739% above the earlier exact100.31 result.

Dynamic Boost is not locked and this is not an AB/BA confidence campaign.
Whole-process200ms NVML peaks are11775MiB for all three new stages. Median SM
clocks when utilization≥90% are2362,2332 and2287MHz respectively; the previous
experimental stage has2362MHz and vLLM2190MHz. These windows include prefill and
are not decode-only clock measurements. Weight/KV/workspace arena bytes stay
unchanged; the final graph-device allocation is31457280bytes, unchanged from
the previous experimental path.

## What changes

`GEM16_12B_D3_ATTENTION=fp8` selects the cumulative experiment; unset or `exact`
keeps the original. Experimental support remains fixed to12B D3/checkpoint FP8
KV/SM120/capacity17519. Unsupported geometry fails visibly. The new result label
is `experimental_d3_target_and_assistant_fp8_tensor_core_attention`.

1. The four-row post-attention boundary uses128 threads and paired values,
   warp-first RMS reductions and packed FP4 conversion. BF16 intermediate
   boundaries, FP4 block16 scales, weights and layouts stay intact, while FP32
   reduction order changes. Actual dispatched SASS contains
   `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`;54registers,16shared bytes,
   zero captured local bytes/thread. Operator medians improve roughly8%.
2. Assistant global attention reuses the already embedded global cubin at
   **one query row**. The slot lends its existing Target attention workspace;
   proposals and verification use it sequentially on the same stream. No new
   allocation or second weight copy is introduced. Assistant FP8 Q/P rounding
   is now explicitly approximate; this port preserves all1135 paired-stage
   output IDs but changes proposals.
3. Forty local Target layers use an offline embedded FP8 Tensor Core QK/PV
   kernel: QH16/KVH8/H256, four rows grouped by KV head, four segments,32-token
   tiles, FP8 Q and P*256 with FP32 online softmax/accumulation/merge. It reads
   the original1024-slot ring plus four speculative rows. Each row retains its
   own causal1024-token window; cache append/commit/rollback are untouched.

The first local prototype spilled heavily; pointer selection/int32 indexing
reduced resources. A valid64-token-tile screen still had360stack bytes and
24.31µs/operator. The selected32-token tile has236registers, **zero stack/local**,
15360dynamic+1024static shared bytes, and18.29µs versus36.85µs native in the
same diagnostic operator screen. Only the selected32-tile variant entered a
full-model cohort. These operator timings are not primary throughput claims.

The ready local SM120a cubin is169KiB, with source/actual Triton3.8.0 and
ptxas-blackwell13.3.33 hashes in its generated provenance. A cold offline build
reproduces both cubin and embedded header byte-exact. The global artifact is
unchanged. Module/resource preparation happens before capture; runtime never
imports Triton, JITs, repacks weights or reads artifact files while decoding.

## Acceptance and final actual dispatch

| Stage | Groups | Accepted drafts | Target batches |
|---|---:|---:|---:|
| Previous experimental |447|684|450|
| Paired boundary |433|698|436|
| Plus Assistant |434|697|437|
| Plus local Attention |441|690|444|

The paired-stage gain includes changed outputs and a better acceptance ratio;
it must not be attributed entirely to kernel latency. Local FP8 lowers operator
cost but worsens acceptance relative to the Assistant stage, leaving a smaller
0.466% cumulative end-to-end increment. Final output first differs from the
previous experimental stream at zero-based index2. Every cohort is deterministic
across its three requests and reports zero fallbacks/token-loop allocations.

The final bounded trace has13 groups in verifier context band16414..16445.
Actual calls/group:40 `gem16_d3_local_fp8` plus40 merges, eight four-row global
FP8 calls plus three one-row Assistant global calls,48 paired post-attention
boundaries. All selected graph nodes share one host-launch correlation, keeping
the full conditional GPU-driven D3 chain. No hot allocation/module-load runtime
API appears in the selected recurring interval.

| Diagnostic milliseconds/group | Before local FP8 | Final | vLLM reference |
|---|---:|---:|---:|
| Kernel interval union |22.899|22.396|21.457|
| Target local attention |1.831|1.132|1.538|
| Target global attention |0.681|0.690|0.828|
| Target norm/quant/rotary/elementwise |1.751|1.756|1.160|
| Target Down |2.787|2.806|2.362|
| Target head/sampling/control |2.957|2.979|2.753|
| Intragroup time without a kernel |0.389|0.394|0.138|

Assistant global attention drops from0.549ms in the previous107.34 capture to
0.132ms final. Kernels/group remain1125; local attention replaces split/merge
with split/merge. Costs come from separate captures with different values,
clocks and output streams, not additive guaranteed savings. Next measured
targets are **Down, remaining norm/quant boundaries and graph-node gaps**.
The current Target attention paths are already faster than this vLLM reference
in the diagnostic band; another attention rewrite is less well supported.

## Numerical, memory and exact-default checks

The paired operator passes12 zero/general/large cases with actual layer0/15/31/47
norm weights, aliasing and optional outputs. Those sampled results happen to be
byte-exact; full-model output equality is not promised for the changed tree.
Predeclared hidden/decoded-FP4 L2 bounds are0.05%/1%; neither is loosened.

Single-row global attention passes14 cases, finite/causal/guard/graph checks;
maximum relative L2 against native is7.72745%. Four-row global14-case regression
also passes. Local attention passes24 cases including early/empty segments,
127/128, window fill/crossing1020..1024, wrap2047, context16383 and capacity end.
Uniform queries match a separate CPU ring/speculative/causal reference within
1e-6. Moderate-query maximum relative L2 is9.60795%, maximum absolute error
0.00967897, below the predeclared experimental15% vector bound. These vector
errors are not a model-quality score. All operators replay graphs deterministically.
Scoped memcheck/synccheck/racecheck report zero errors/warnings. Full-model
sanitizers, Windows, media and26B qualification were not run.

Products build passes with
`cmake --build build/Linux/blackwell-release --target gem16-run gem16-chat gem16-server gem16-unit-tests -j4`.
`build/Linux/blackwell-release/bin/gem16-unit-tests` reports all host unit tests
passed. Four accounting tests pass via
`python -m unittest discover -s tests/python -p test_d3_path_accounting.py -v`.

## Initial quality screen

The existing pinned sgl-eval harness scores the first32 cached GSM8K test
questions on the same final server binary, greedy, no thinking, one answer per
question, max512tokens, D3/capacity17519. This is a **small arithmetic smoke
screen**, not general quality qualification or a performance cohort.

| Quality result | Exact | Experimental |
|---|---:|---:|
| Scored correct |29/32 (90.625%)|31/32 (96.875%)|
| Truncated at512tokens |3|1|
| API errors |0|0|

Thirty-one answer texts differ. There are zero exact-correct/fast-wrong pairs
and two exact-unscored/fast-correct pairs. Every unscored response is truncated;
all completed answers are scored correct. The higher fast score is therefore
associated with fitting more final answers inside the common output budget,
not proof of better mathematical reasoning. The dataset is small and does not
cover code, long-context factuality, tools or multimodal quality. There is no
evidence of degradation in this screen, and no general equivalence claim.

One orchestration retry used a different loopback port because the terminated
exact server's port remained in TCP TIME_WAIT. Its32 saved answers were reused;
no samples were rerolled. Both owned servers stopped cleanly.

## Portable evidence and storage

[Evidence manifest](../../../artifacts/gemma4_12b/d3-fast-followup-2026-11-17.json)
records samples, exact commands/environments, binary/source hashes, memory and
clock telemetry, dispatch, qualification and all paired quality scores.
[Source snapshot](../../../artifacts/gemma4_12b/d3-fast-followup-2026-11-17-source.json.gz)
and [raw receipts](../../../artifacts/gemma4_12b/d3-fast-followup-2026-11-17-raw.json.gz)
preserve this uncommitted state and every quality response.
Full local results: `benchmarks/results/2026-11-17/d3-fast-followup/`.
SQL/compiler temporaries used checked RAM scratch and are removed. Two bounded
original Nsight reports are losslessly compressed with verified round-trip
SHA256;88.9MB replaces188.5MB uncompressed. Compact kernel/API gzip records and
ordered CSVs remain available directly. No own GPU jobs remain.
