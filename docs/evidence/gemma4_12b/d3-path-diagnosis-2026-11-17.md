# 12B D3 kernel diagnosis — 2026-11-17

The largest measured native/vLLM gap was Target global attention, not draft
acceptance or insufficient graph capture. Native used scalar FP32 score/value
arithmetic on dequantized FP8 K/V; vLLM used FP8 queries and FP8 softmax tiles
with Tensor Core QK/PV. The subsequent explicitly authorized experimental path
is recorded [separately](d3-fast-attention-2026-11-17.md).

## Scope and timing

RTX5080 Laptop, 16GB, driver620.70.06; 12B NVFP4 Target snapshot
`b1f649734b34aa5575b03d186abd1b9be3d0d5c4`, BF16 Assistant snapshot
`364bd03c9952e5b7da73665ee30c9eccfc408345`, native base `73cf742`.
Target weights SHA256 `7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b`;
Assistant weights SHA256 `3279c173daddd7186e79d652ad94022415736d3a1370625696c898429b06d6df`.
Native CUDA13.4.92/GCC16.2.1; vLLM0.33.0, Torch2.14.0/CUDA13.0,
Triton3.8.0. Exact inspected vLLM files are hashed in the evidence record.

Batch1, greedy, 16384 input IDs, 1135 outputs, capacity17519, FP8 KV;
1134 timed decode intervals after first token. Suppression, prompt and output
length stay fixed. Native prefill chunk8192 and vLLM max-batch1024 follow the
existing protocol; decode timing excludes prefill. Streams differ across engines.
Fresh vLLM D3: one warmup plus three samples
117.146656 /117.132291 /117.094626tok/s, median117.132291.

Separate bounded diagnostic captures compare12 groups per engine in verifier
context band16414..16445. The first/last two of16 vLLM profiler iterations are
excluded by position, not speed. Native group acceptance is reconstructed and
checked against counters from the same capture. GPU launch correlations resolve
all vLLM events to CPU phase annotations, including asynchronous graph nodes
without an External ID. Costs below are milliseconds/group, not primary rates.

| GPU work | Native exact | vLLM |
|---|---:|---:|
| Kernel interval union |25.321|21.457|
| Assistant, summed |4.557|4.318|
| Target backbone, summed |17.841|14.470|
| Target head/sampling/control, summed |2.924|2.753|
| Target global attention |3.256|0.828|
| Target local attention |1.710|1.538|
| Target FP8 QKV/O |3.530|3.933|
| Target NVFP4 Gate/Up |4.163|4.650|
| Target NVFP4 Down |2.683|2.362|
| Target norm/quant/rotary/elementwise |2.213|1.160|
| Intragroup interval without a kernel |0.371|0.138|

There are1173 native kernels/group versus959.17 vLLM. Family/phase sums may
overlap and cannot be added as independent savings. Captures use the same
context band but different values, outputs, profiler implementations and clocks.

## Actual order, shapes and graphs

Native steady group: three sequential one-row Assistant proposals; build four
verification rows;48 Target layers; four-row output head; accept prefix and
update KV/control; conditional continuation. Selected groups share **one host
cudaGraphLaunch** for the GPU-driven conditional D3 chain.

vLLM steady worker iteration verifies the preceding four-row block, samples and
accepts, then proposes the next block. It makes **three graph launches**:
four-row Target backbone, four-row first Assistant pass (rejected positions
remain padded), and one fused graph containing the remaining two one-row draft
steps. All16 captured iterations enter `_fused_multi_step_decode`. Target
head/sampling is outside its backbone graph. More graph capture alone does not
explain the native gap; native already chains groups on the GPU.

Native global dispatch: grid35×4, block256,64 registers,41056 shared bytes;
eight split and eight merge kernels/group. Disassembly of the **actual dispatched
mangled symbol** has82 static FFMA sites, zero QMMA/HMMA. Native BF16-rounded Q
and FP8-dequantized K/V use FP32 score/probability/value accumulation.

vLLM global: `kernel_unified_attention`, grid5×1×16,128 threads,254 registers,
41728 shared bytes; eight calls plus eight `reduce_segments` calls/group.
Installed attention code enables quantized Q for FP8 KV on this backend;
the unified kernel retains FP8 K/V, uses `tl.dot(Q,K)`, rounds `P*256` to FP8
and computes FP8 PV/256. Matching cached cubins use E4M3 QK/PV MMA and32 static
QMMA sites. **The exact runtime vLLM CUDA module identity was not captured**;
cache name/resource matching is disclosed rather than treated as exact identity.

## Output semantics and next work

All1135 native D3 IDs equal the existing native Ordinary stream. Fresh vLLM D3
matches its earlier D3 stream, but differs from its own Ordinary at zero-based
index516 (9593 versus8582). Native/vLLM D3 first differ at index21. The vLLM
advantage is not better acceptance here: native437 groups/694 accepted versus
vLLM445/689 for1135 outputs.

Algorithmically lossless speculation does not imply bitwise equality between
different finite-precision matrix/reduction shapes; see the official
[vLLM speculative-decoding documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/).
These traces do not isolate the first differing logit to one operator.
That would require fixed-prefix layer/logit comparisons, not another throughput
run. Neither cross-engine equality nor general quality equivalence is claimed.

With exact output retained, sensible remaining targets are measured
norm/quant boundaries, Down and launch/control gaps; changes need byte-exact
checks. With changed math explicitly allowed, global attention was the first
candidate and is now implemented experimentally. Its successful result shifts
the next investigation to remaining Target boundaries/Down and Assistant costs.

## Evidence

[Machine-readable record](../../../artifacts/gemma4_12b/d3-path-diagnosis-2026-11-17.json)
contains costs, output comparisons, graph boundaries, instruction evidence and
capture hashes. The linked374KB raw archive retains primary JSON/CSV, exact
commands, ordered representative kernel CSVs and scoped qualification logs.
Large bounded reports/events remain under
`benchmarks/results/2026-11-17/d3-path-diagnosis/`; SQL intermediates were on
checked RAM scratch and are removed. Four host accounting tests pass via
`python -m unittest discover -s tests/python -p test_d3_path_accounting.py -v`.
