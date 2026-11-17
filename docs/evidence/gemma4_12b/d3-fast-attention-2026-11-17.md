# Exact default and experimental fast 12B D3 — 2026-11-17

This records the first107.34tok/s slice. The subsequent
[measured follow-up](d3-fast-followup-2026-11-17.md) reaches112.10tok/s and adds
Assistant/global, local Target and paired-norm changes plus a small quality screen.

The first experimental implementation raises D3 decode from100.31 to107.34tok/s
(+7.01%) on the existing16K workload. vLLM remains faster at117.13tok/s.
The original exact path remains the default; the experiment deliberately changes
global-attention math and does not promise identical IDs or general quality.

| Final primary cohort | Three retained samples, tok/s | Median |
|---|---|---:|
| Exact default |100.385075,100.307823,100.239550|100.307823|
| Experimental |107.387381,107.336558,107.168171|107.336558|
| Fresh vLLM D3 reference |117.146656,117.132291,117.094626|117.132291|

Each cohort has **one warmup plus three measurements**, batch1,1135 outputs and
1134 timed decode intervals. Prompt, greedy selection, suppression, context,
checkpoint/Assistant identities and KV layout are unchanged; see the
[diagnosis](d3-path-diagnosis-2026-11-17.md) for the protocol/toolchains.
The final exact and experimental cohorts use the **same immutable executable**,
SHA256 `b0dbfafb246d61302e65e415af2a0807d4771edfb4a82adbf0959bcc2338fb47`,
base `73cf742` plus the source snapshot in the evidence record.
These are point estimates, not a locked-clock AB/BA confidence campaign.
Whole-process200ms NVML peaks are11773/11775MiB. Median SM clocks at≥90% GPU
utilization are2497/2362MHz respectively; this is not a decode-only clock window.
Weight, KV and workspace arena byte counts are identical. Graph-device memory
is29360128 versus31457280 bytes (+2MiB).

## Selection and implementation

Set `GEM16_12B_D3_ATTENTION=fp8` before startup for the experiment; unset it or
set `exact` for the original path. Supported experimental geometry is **12B,
D3, checkpoint FP8 KV, SM120, capacity17519 only**. Other geometry fails visibly;
an actual17520 startup rejection passed. The result's `decode_attention_path`
explicitly reports `experimental_d3_global_fp8_tensor_core_q_and_probabilities`.
The existing `batched_exact_target` verification label describes the acceptance
algorithm; it does not assert equality to the original Ordinary math in this mode.
This is an opt-in diagnostic configuration, not a new qualified Studio profile.

Only eight four-row Target global-attention layers change math: Q is clamped/
rounded to E4M3 with scale1; FP8 Tensor Core QK/PV uses original FP8 K/V and BF16
scales; online softmax tiles round P*256 to E4M3;16 segments are merged in FP32.
There is no extra attention-scale factor beyond the native Q contract. TMA
loads contiguous K/V tiles from the existing cache. Assistant, local attention,
matrix weights/projection kernels, prefill and Ordinary tail retain their math.

Cached four-row input/post-attention/post-MLP norm boundaries are also ported
from the accepted Ordinary kernels, including BF16 physical Down input. They
preserve the native reduction tree, intermediate rounding and FP8/NVFP4 packing.
Attention BF16 rounding and FP8 quantization fuse into one kernel, removing48
launches/group. These ports introduce **no additional output/proposal changes**
relative to the measured TMA-attention-only variant.

The Triton source is **offline build input only**. An embedded SM120a cubin is
loaded/prepared once before capture; recurring inference never invokes Python,
JIT, filesystem loading or repacking. Compiler: Triton3.8.0 and actual
`ptxas-blackwell13.3.33`, separately from native NVCC13.4.92. Both generated
attention header and norm CUDA reproduce byte-exact from cold offline builds.
Source/compiler/output hashes are checked into their provenance JSONs.

Actual final attention dispatch is grid16×4,128 threads,145 registers,
74272 dynamic+1024 static shared bytes, zero stack/local bytes. The embedded
artifact has32 static QMMA sites. Its16KiB TMA descriptor scratch and partials
reuse the already reserved MTP arena. The full conditional D3 graph is retained.

## Correctness and measured limits

All1135 exact IDs equal the existing native Ordinary/D3 stream. Experimental
IDs are deterministic across the three requests but first differ at index46
(47th output; exact3818, experimental5581). Accepted drafts fall694→684 and
groups rise437→447; the gain is not an acceptance-ratio improvement.

Attention checks cover14 position/query cases, including early/empty segments,
the511/512 boundary and capacity end, causal uniform-reference checks, finite
values, guarded buffers and three graph replays. The predeclared experimental
moderate-query relative-L2 bound is15%; observed maximum8.20329%, maximum
absolute error0.00989445. These are synthetic attention-vector errors, **not an
answer-quality score**.124 actual-weight norm-boundary comparisons pass byte
exactly at layers0/15/31/47. Scoped memcheck/synccheck/racecheck pass with zero
reported errors/warnings for both operators. This is not a full-model sanitizer,
quality, Windows,26B or media qualification.

Final products build succeeds:
`cmake --build build/Linux/blackwell-release --target gem16-chat gem16-server gem16-unit-tests -j4`;
host unit binary reports `all host unit tests passed`. The four Python accounting
tests pass. Every full primary run reports zero fallbacks/token-loop allocations.
Final Nsight capture independently finds no allocation/module-load APIs inside
the selected recurring GPU interval and confirms all six new dispatch families.

## Final profile and remaining gap

In13 groups in the same16414..16445 context band, final global attention costs
**0.671ms/group**, compared with3.256ms exact and0.828ms vLLM in the separate
diagnostic captures. Kernel count falls1173→1125/group; whole kernel union is
23.133ms, versus25.321 exact and21.457 vLLM. Final Target backbone15.396ms,
Assistant4.798ms, head/sampling/control2.940ms; intragroup no-kernel interval
0.384ms. Separate captures and changed token streams/clocks prevent interpreting
these differences as additive causal throughput savings.

The global-attention gap is closed in this diagnostic band. Further work should
profile remaining norm/quant boundaries, Down and Assistant against actual
inputs before choosing another kernel change. Another9.13% speedup from the
experimental result would reach the current117.13tok/s reference; that is a
remaining objective, not a guaranteed gain.

[Evidence JSON](../../../artifacts/gemma4_12b/d3-fast-attention-2026-11-17.json)
contains raw final samples/IDs, commands/environment, memory/clock telemetry,
source/binary/artifact hashes, scoped checks and final actual dispatch costs.
[Portable source snapshot](../../../artifacts/gemma4_12b/d3-fast-attention-2026-11-17-source.json.gz)
and [raw evidence archive](../../../artifacts/gemma4_12b/d3-paths-2026-11-17-raw.json.gz)
preserve this uncommitted experiment. Full local receipts are under
`benchmarks/results/2026-11-17/d3-fast-attention/`; large SQL/compiler temporary
files used checked RAM scratch, now removed. No commit or push was made.
