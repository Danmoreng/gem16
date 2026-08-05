# Performance ledger

## 2026-08-05 Windows short-context llama-bench characterization

The standard llama.cpp b10240 `llama-bench` `pp512`/`tg128` matrix and the corresponding gem16
`prefill --context 512` / `decode --context 1 --tokens 128` paths were rebuilt and measured on the Windows RTX 5080
Laptop GPU. The final engine runs started idle at 50/51 C and both reached 66 C. Llama.cpp executed its built-in
warm-up plus 13 repetitions; repetitions 1–3 were discarded and 4–13 were reported. Gem16 executed three warm-ups
and ten measured repetitions.

| Engine | pp512 median tok/s | Prompt time | tg128 median tok/s | Aggregate ms/token | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| gem16 `cc01a05` | **6,877.29** | **74.448 ms** | 52.43 | 19.072 ms | 10,634 MiB |
| llama.cpp b10240 | 5,400.91 | 94.799 ms | **62.08** | **16.109 ms** | **9,824 MiB** |

Gem16 leads pp512 by 27.34%, while llama.cpp leads tg128 by 18.39%. The result is retained as a standardized-shape
development characterization, not exact token/format parity: llama-bench uses random synthetic tokens, a zero-depth
generation start, F16 K/V, and aggregate timing; gem16 uses deterministic benchmark/generated tokens, the smallest
legal one-token initial context, checkpoint-FP8 K/V, and records every inter-token interval. The attention formats
also differ between the direct checkpoint and patched GGUF. Full distributions and telemetry are in
`benchmarks/baselines/llama_cpp/windows-short-context-cc01a05.json`; raw evidence is under
`benchmarks/results/2026-08-05/cc01a05/blackwell16gb-windows-short-context-llama-bench-b10240-3x10/`.

## 2026-08-05 Final corrected Linux 16K qualification

At exact commit `a819d14c`, the alternating Wikipedia 16K qualification ran three warm-up and ten measured
ordinary/fixed-D2 pairs. All 26 executions retained Target hash
`08ddc8178b2c9ac3caefa046da1c521318b913f32f275f18892ad98d21c25ea1`. Ordinary decode reaches 47.760 tok/s
median with 95% CI `[47.758,47.767]`; corrected fixed D2 reaches 87.423 tok/s with CI `[87.412,87.546]`, a 1.830x
speedup. Restoring required verifier BF16 boundaries costs approximately 2.7% against the invalid S00-era 89.86
characterization. Correctness takes priority and that old number is not retained.

The controlled same-machine cross-engine 3/10 run gives gem16 5,866.86 prefill tok/s, 2,792.64 ms TTFT, 87.66 D2
tok/s, and 11.408 ms ITL. vLLM 0.26.0 reaches 6,257.37/2,618.35/82.25/12.158; llama.cpp b10240 reaches
3,941.23/4,157.08/83.89/11.921. Gem16 remains 6.24% behind vLLM prefill and 48.86% ahead of llama.cpp, while D2
remains 6.57% ahead of vLLM and 4.49% ahead of llama.cpp. Gem16 peaks at 11,746 MiB in command-wide 200 ms
telemetry. vLLM retains disclosed autotuning OOM fallbacks and an untuned 8K FP4-shape warning. Machine-readable
summaries are under `benchmarks/baselines/cross_engine_mtp/`; raw ignored evidence is under
`benchmarks/results/2026-08-05/a819d14c/blackwell16gb-linux-maxpower-12b-sprint/S08-final/`.

The Linux performance qualification is complete. Sprint closure still requires the externally blocked Windows gate
and the direct all-regions sampling/media memory-reserve record; no final stop category is asserted before those
facts are recorded.

## 2026-08-05 Restore exact sampled-MTP verifier and graph suppression semantics

The S08 same-seed sampled-MTP gate failed at the sprint parent. Bisect and direct boundary comparison localized the
numerical regression to `31c8519`, which removed standalone BF16 rounds after V and O projections when production
prefill projections began writing BF16 directly. The direct short-batch verifier continued writing FP32 and
therefore lost two ordinary-decode boundaries. Restoring those rounds only for MTP verification made its target
logits bit-identical to ordinary decode at the investigated divergence.

That exact comparison exposed a second issue: ordinary CUDA-Graph sampling captured suppression count zero during
initialization, while direct MTP observed the later configured count. Sampling now reads the runtime count from
`DecodeControl`, and fixed-D2 row controls carry it as well. Host and CUDA CTest pass. D1/D2/D4 same-seed matrices
pass for seeds 0/1/42 in checkpoint-FP8 and BF16 cache modes and at repetition penalties 1.0 and 1.1; resident
sampled D2 chat remains identical and GPU-chained. Sampling memcheck reports zero errors and racecheck zero
hazards. This is a correctness repair, not a speed claim. It adds back the required verifier round launches, so
S08 must remeasure fixed-D2 performance before retaining prior final numbers.

## 2026-08-04 Reject two-CTA cluster sharing for global D512 prefill

Nsight Compute admitted S07B after one representative second-chunk global prefill launch produced 823,607,909 L2
sectors (about 26.36 GB), 312.531 MB DRAM reads, and a 98.675% L2 hit rate. The SM120 device supports clusters but
provides 102,400 shared bytes/SM; the parent already allocates 99,328 bytes/block and 255 registers/thread, so it
runs one CTA/SM at 16.67% occupancy. A bounded `(1,2,1)` cluster prototype kept four query heads per CTA. Rank 0
loaded and converted raw FP8 K/V; rank 1 copied the exact converted BF16 tile through distributed shared memory
into its local `ldmatrix` operand view. This avoided assuming remote `ldmatrix` support and retained all arithmetic
and tile geometry.

The CUDA operator suite passed. The candidate used 254 registers/thread, unchanged 99,328-byte shared allocation,
and no reported spills. It reduced L2 sectors by 46.00% to 444,751,632, while DRAM reads fell only 2.34% to
305.224 MB. Distributed-shared copies and two cluster synchronizations per shared operand raised representative
kernel time from 162.219 to 312.865 ms (+92.87%). A short 8K prefill screen regressed from 7,272.01 to 6,246.32
tok/s (-14.10%). Decision: reject before exact-16K escalation, remove the prototype completely, restore production,
and pass Host/CUDA CTest. The experiment confirms that duplicate K/V traffic is almost entirely cheap L2 reuse;
DSM sharing is worse. Raw ignored evidence is under
`benchmarks/results/2026-08-04/3e8a1e3/blackwell16gb-linux-maxpower-12b-sprint/S07B-cluster-kv/`.

## 2026-08-04 Reject global GQA publication-barrier removal after counter admission

After Nsight Compute 2026.2.1 became available, a representative ordinary 16K global GQA split launch measured
385.984 us, 68 registers/thread, 42,112 allocated shared bytes/block, and a 33-block grid covering only 0.28 waves
on 60 SMs. Its 4.60 warp cycles per issued instruction were dominated by short scoreboard (19.0%), wait (16.7%),
MIO throttle (14.8%), long scoreboard (9.0%), and barrier (3.88%). This admitted bounded S06C Step 1 as the second
S05 candidate.

The candidate removed only the explicit `inverse_sum_shared` publication barrier for heads 0-14; the first barrier
inside the next head's `DecodeBlockMaximum` safely published the prior write and protected reduction reuse. The
final head retained its output-publication barrier. Racecheck reported zero hazards, exact checksums/hashes were
unchanged, and resources remained 68 registers/thread and 42,112 shared bytes. The profiled kernel fell to
374.432 us (-2.99%). Ordinary 16K/64-token decode improved in both adjacent orders: 47.8754 to 48.0027 tok/s and
47.9021 to 48.0006 tok/s (+0.21-0.27%). Exact fixed-D2 did not: the first order improved 89.7415 to 90.0450 tok/s,
but reversed ordering regressed 89.8866 to 89.7869 tok/s. Decision: reject S06C because its required D2 win was not
stable, remove it completely, restore production, and pass Host/CUDA CTest again.

The new prefill counters separately measured about 26.32 GB of L2 sector traffic, 440 MB of DRAM traffic, and a
98.675% L2 hit rate in one second-chunk global D512 launch. This validates S07B's repeated cache-resident K/V traffic
premise, while the decode kernel's 9.0% long-scoreboard share supports a future S06D discussion. Neither starts now:
S06A and S06C consumed S05's bounded two-candidate set. Raw ignored evidence is under
`benchmarks/results/2026-08-04/c8be1d6/blackwell16gb-linux-maxpower-12b-sprint/S06-ncu-admission/` and
`S06C-global-gqa-barrier/`.

## 2026-08-04 Reject combined prompt Q/K/V CUTLASS projection

S06A reordered Q/K/V weights and scales contiguously inside the sole Target arena and replaced separate prompt
Q/K/V GEMMs with one local `N=8192` or global Q/K `N=8704` CUTLASS GEMM. Combined row-major outputs were consumed
through exact strided normalization views; decode/MTP retained direct interior weight bindings. There was no second
persistent layout, byte increase, fallback, or token-loop allocation. A 128-token full-model comparison retained
byte-identical logits, and all exact Wikipedia runs retained hash
`b21d3676985d06ad23525f9f4b52dd5134fee95d97d8aaca89e61736d74c3afb`.

The first one-warm-up/two-run screen measured candidate/parent at 5,944.02/5,888.09 prefill tok/s, but reversed
one-warm-up/three-run ordering measured parent/candidate at 5,895.26/5,888.89 tok/s with substantially overlapping
confidence intervals. The candidate was 0.11% slower in the reversed order. Profiling explains the neutral result:
FP8 CUTLASS launches fell from 368 to 192, but their total time fell only from 543.812 to 542.321 ms (1.491 ms,
0.27% of the family and 0.054% end to end). Output tiles still load A independently, while strided Q/K and V
normalization consumed most of the small reduction. Decision: reject S06A for no stable end-to-end win and remove
the implementation completely. Raw ignored evidence is under
`benchmarks/results/2026-08-04/75efb08/blackwell16gb-linux-maxpower-12b-sprint/S06A-combined-qkv/`.

## 2026-08-04 Reprofile the promoted 12B stack and admit combined Q/K/V only

A fresh exact 16K Nsight Systems profile at `c50dd4d` measured the `gem16.prefill` range at 2,769.309 ms. The
2,005 contained kernel launches sum to 2,768.461 ms, leaving 0.848 ms (0.031%) of host/API gap. The ranked families
are global D512 attention 1,005.440 ms (36.307%), NVFP4 GEMM 605.276 ms (21.857%), FP8 Q/K/V/O projection
pipeline 591.191 ms (21.348%), local attention 321.451 ms (11.608%), Q/K/V normalization plus K/V quantization
134.617 ms (4.861%), residual/norm boundaries 63.402 ms (2.289%), NVFP4 activation quantization/interleave
23.356 ms (0.843%), and NVFP4 weight preparation 20.538 ms (0.742%). Final output work is 2.753 ms (0.099%).

The profile executes 368 separate FP8 CUTLASS projection kernels. Together with the source-confirmed repeated use of
the same quantized activation for separate Q/K/V launches, this admits S06A's bounded combined-Q/K/V prototype.
No second candidate is admitted yet: S06B is ordered after S06A resolution; S06C/S06D require matching stall
counters; and S07B requires attributed duplicate K/V traffic rather than timing plus source inspection alone.
Nsight Compute is not installed, so per-family DRAM bytes, L2 hit rate, and dominant stalls remain explicitly
unavailable and no traffic or latency-hiding claim is made. Raw ignored evidence and the complete phase table are
under `benchmarks/results/2026-08-04/c50dd4d/blackwell16gb-linux-maxpower-12b-sprint/S05-reprofile/`.

## 2026-08-04 Retain the 8K prefill chunk after 8K/12K/16K sweep

Hypothesis: S02's arena reduction may admit a larger checkpoint-FP8 prefill chunk that removes repeated per-chunk
MLP preparation and scale-interleave work from the fixed 16K prompt. Separate temporary builds screened 8,192,
12,288, and 16,384 tokens; all selectors and memory diagnostics were removed after the experiment.

Dry-run prefill arenas were 1,585,254,144, 2,331,561,728, and 3,077,869,312 bytes. With Target, assistant,
sampling, media regions, ordinary graphs, fixed-D2 graph, and a 17,519-position plan resident, direct
`cudaMemGetInfo` probes retained 4,323,082,240, 3,576,496,128, and 2,829,910,016 free bytes, all above the 700 MiB
gate. Sampled exact-workload peak process use was 11,746, 12,458, and 13,170 MiB.

Short synthetic screens were mixed, so all candidates ran the exact Wikipedia 16K workload with one warm-up and
three measurements. The 8K/12K/16K medians were 5,917.30/5,845.77/5,916.23 prompt tok/s and
2,768.83/2,802.71/2,769.33 ms TTFT. The 8K and 16K confidence intervals overlap, and 16K is 0.02% slower; 12K is
materially slower. All runs retained hash `b21d3676985d06ad23525f9f4b52dd5134fee95d97d8aaca89e61736d74c3afb`.
The 16K build halved NVFP4 preparations from 288 to 144 and interleaves from 96 to 48 but did not improve the final
distribution.

Both candidate builds produced byte-identical full logits against 8K at `chunk - 1`, `chunk`, and `chunk + 1`.
Repeated image/audio prompts also emitted identical output bytes, with no fallback or token-loop allocation.
Decision: retain 8,192 because neither larger chunk wins the required exact workload. Raw ignored evidence is under
`benchmarks/results/2026-08-04/6a8892d/blackwell16gb-linux-maxpower-12b-sprint/S04-chunk-sweep/`.

## 2026-08-04 Reject direct CUTLASS MLP activation-scale generation

Hypothesis: the fused RMSNorm/NVFP4 quantizer can write E4M3 scales directly in CUTLASS's 128-row tiled layout,
remove 96 interleave launches from a two-chunk 16K prompt, and shrink compact scale storage to the five-row MTP
verifier extent. Two implementations were screened: direct byte stores from each token CTA and a coalesced variant
that sequentially preserved the exact 256-thread reduction for four rows separated by 32 and emitted one 16-byte
store per scale tile.

Both variants produced exact E2M1 payload and interleaved E4M3 identity for FP32 and physical-BF16 inputs at 1, 5,
127, 128, 129, 257, and 8,192 rows. Full CUDA CTest passed, and parent/candidate full-logit dumps were byte-identical
at 1, 129, 257, 8,192, and 8,193 prompt tokens. The candidate removed 1,964,800 workspace bytes, all interleave
launches, and no recurring allocation or fallback was added.

The short 16K rejection screens were decisive. Direct stores reached 5,432.25 and 5,402.87 tok/s versus adjacent
parent results of 6,013.21 and 5,899.37 tok/s. Coalesced stores reached 5,465.17 tok/s versus 5,956.87 tok/s
(-8.25%). At 8K, Nsight measured parent quantization plus 96 interleaves at 22.483 + 1.073 ms and the direct kernel
at 22.619 ms, but the following regular Gate/Down CUTLASS kernels rose from 371.417 to 598.712 ms and fused Up from
236.388 to 280.868 ms. The removed interleave is a cheap immediately preceding layout/cache warm-up for the much
larger GEMMs. The direct kernel retained 40 registers and zero stack/local memory, adding 960 dynamic bytes to the
parent's 1,024 static shared bytes.

Decision: reject and remove S03. The 1.87 MiB arena reduction does not justify a repeatable 8-10% prefill loss; keep
compact generation plus the separate interleave. Raw ignored evidence is under
`benchmarks/results/2026-08-04/6a8892d/blackwell16gb-linux-maxpower-12b-sprint/S03-direct-scales/`.

## 2026-08-04 Promote final-row-only prompt RMSNorm

Hypothesis: ordinary prompt processing only needs final RMSNorm for the last row of the final chunk, while the
separate MTP verification path needs at most five rows. The promoted path skips final RMSNorm on non-final chunks,
normalizes the final prompt row into workspace row zero, and retains five FP32 rows in the arena.

The named allocation falls from 125,829,120 to 76,800 bytes at the 8,192-token chunk, an exact 125,752,320-byte
reduction. Parent/candidate runtime reports show the same whole-workspace reduction at 1, 8,192, 8,193, and 16,384
prompt tokens; weights, KV, graph-private bytes, and recurring allocation behavior are unchanged.

Host and CUDA CTest and `validate_inference.py` pass. Parent and candidate produce byte-identical full-logit dumps
and identical output IDs at 1, 8,192, 8,193, and 16,384 prompt tokens, with zero fallbacks and no token-loop
allocation. D1/D2/D4 continuation screens also retain identical 16-token outputs. The regenerated pinned vLLM
0.26.0 boundary fixture is numerically unchanged from 0.25.1: 129 tokens pass, while the inherited 257-token
Top-1 mismatch remains identical in parent and candidate. The project owner explicitly directed the sprint to
ignore that pre-existing external-reference mismatch rather than attribute it to this change.

Short one-warm-up/three-measurement 16K prefill screens were order-dependent: the first ordering measured
5,969.94 versus 5,917.29 tok/s parent/candidate, while the reversed ordering measured 5,927.10 versus 5,954.59
tok/s. The D2 screen measured 88.403 versus 88.258 tok/s with overlapping confidence intervals and identical output
hashes and acceptance counters. No long escalation was run at the owner's request.

Decision: promote as the S02 memory foundation. It removes 119.93 MiB from the named arena with exact internal
behavior and no established performance regression. Raw ignored evidence is under
`benchmarks/results/2026-08-04/ed0ed689/blackwell16gb-linux-maxpower-12b-sprint/S02-final-rmsnorm/`.

## 2026-08-04 Reject cached attention-merge split weights

Hypothesis: the ordinary, global-T3, and local-D2 attention merge kernels can compute each split exponential once
for the denominator, retain it in a separate shared array, and reuse it across output dimensions without changing
the reduction or accumulation order. The candidate did so for all three source families with a 512-FP32-entry
array covering the 262,144-position contract.

Focused tests covered one valid split, the local maximum, 16K, 64K, global T3, local D2, repeated bit identity, and
the 512-split maximum. Full CUDA CTest passed; Compute Sanitizer reported zero memcheck errors and zero racecheck
hazards. SASS reduced static `MUFU.EX2` instructions from 22 to 5 in every merge specialization. Shared memory rose
from 1,056 to 3,104 bytes; D256/D512/global-T3/local-D2 registers changed from 43/40/56/54 to 45/47/54/54, with
zero stack or local memory throughout.

Short one-warm-up/three-measurement screens did not establish an end-to-end win. Ordinary 16K was neutral at
47.88567 versus 47.88526 tok/s. Two 64K screens favored the candidate by 0.270% and 0.175%, but the reversed-order
screen's confidence intervals overlap. Fixed-D2 screens changed sign: +0.176% over 128 outputs and -0.134% over
64 outputs, both with overlapping intervals and identical parent/candidate output hashes. The project owner asked
to avoid long benchmarks, so the optional 30-run escalation was not performed.

Decision: reject and remove the implementation because no representative end-to-end metric improved with the
required statistical support. Persistent, workspace, graph, and KV bytes were unchanged; the temporary 2 KiB
per-CTA shared increase is absent from production. Raw ignored evidence is under
`benchmarks/results/2026-08-04/ed0ed689/blackwell16gb-linux-maxpower-12b-sprint/S01-merge-weights/`.

## 2026-08-04 IMP versus llama.cpp on Gemma 4 26B A4B QAT Q4_0

Hypothesis: IMP's consumer-Blackwell-specific Gemma-4 MoE implementation may improve prompt processing and ordinary
decode over llama.cpp for the exact official QAT Q4_0 GGUF on the 16 GB reference GPU. IMP is pinned to upstream
commit `a392904d4216388828d0d56317de046f4ca49627`. Docker is unavailable on this host, so a clean host Release build
uses GCC 15.3, CUDA 13.3.73, and CUTLASS 4.6.1; an initial GCC 16 build exposed an upstream missing direct
`<cstdint>` include, and no source patch is retained.

The model loads and emits coherent text, but IMP cannot retain all experts on this card. It uploads 21 of 30 MoE
expert layers and leaves 3.59 GiB on the host, disabling CUDA Graphs. The measured ten-repetition run records
75,875 expert-cache hits and 42,226 misses (64.2% hit rate). Its default 3,900 MiB library-reserve estimate leaves
only 256 K/V tokens and rejects pp512, so the characterized run uses the documented
`vram.library_reserve_mb=256` override, a 1,024-token minimum pool, and FP8 E4M3 K/V. The directly adjacent
llama.cpp b10240 run uses the same GGUF with all model tensors on CUDA0, Q8_0 K/V, and Flash Attention.

| Engine | Prefill pp512 | Prefill time | Decode tg256 @ ctx512 | Aggregate ITL | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| IMP `a392904d` | 1,533.75 tok/s | 333.82 ms | 51.64 tok/s | 19.365 ms | 14,804 MiB |
| llama.cpp b10240 | 5,087.77 tok/s mean | 101.18 ms | 169.762 tok/s mean | 5.891 ms | 15,316 MiB |

llama.cpp is 3.32x faster in prefill and 3.29x faster in decode. IMP's lower VRAM is not an efficiency win: it is
the consequence of forbidden CPU expert offload. An explicit INT8-K/V probe is invalid because IMP rejects
Gemma-4's global head dimension 512. The experimental graph-under-offload option also fails capture and falls back
at 52.03 tok/s.

Decision: do not use this IMP/Q4_0/16 GB path as a baseline or as evidence for gem16 speed targets. The result is
specific to a checkpoint and memory class outside IMP's published hero configuration: IMP documents Q4_K_M/NVFP4
on a 32 GB RTX 5090, where all experts fit. No quality comparison or profiler-level kernel audit was performed.
Full provenance and caveats are in
`benchmarks/baselines/imp/gemma4-26b-a4b-qat-q4_0-characterization.json`; raw local evidence is under
`benchmarks/results/2026-08-04/1ffabc4/gemma4-26b-a4b-qat-q4_0-imp-vs-llama-adjacent/`.

## 2026-08-04 llama.cpp Gemma 4 26B A4B QAT Q4_0 exploration

Hypothesis: Google's official QAT Q4_0 GGUF for the 26B A4B MoE will establish whether the model can execute
usefully on the 16 GB reference Blackwell GPU and provide an initial performance target for a future native gem16
backend. The exact revision is `d1c082be9cf3c8a514acf63b8761f4b41935842e`; its 14,439,363,584-byte GGUF has
SHA-256 `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`.

llama.cpp b10240 identifies 25.23B total parameters, 128 experts with 8 active, and offloads all 31 layer groups.
The measured configuration forces the tied `token_embd.weight` to CUDA0 because the default otherwise leaves a
577.50 MiB CPU-mapped model buffer. It uses Flash Attention, Q8_0 K/V, batch one, firmware `max-power`, and active
Dynamic Boost. After discarding three conditioning repetitions, ten measured runs give:

| Mode | Tokens/context | Median tok/s | Median elapsed/ITL |
|---|---:|---:|---:|
| Prefill | 128 | 2,334.86 | 54.824 ms elapsed |
| Prefill | 512 | 5,167.55 | 99.091 ms elapsed |
| Prefill | 2,048 | 5,025.43 | 407.527 ms elapsed |
| Prefill | 8,192 | 4,746.58 | 1,725.877 ms elapsed |
| Decode, 256 tokens | 128 | 174.564 | 5.729 ms/token |
| Decode, 256 tokens | 2,048 | 167.057 | 5.986 ms/token |
| Decode, 256 tokens | 8,192 | 158.267 | 6.318 ms/token |

Peak sampled VRAM is 15,442 MiB, leaving about 438 MiB against llama.cpp's 15,880 MiB CUDA report and missing the
700 MiB project safety margin. Context creation plus one token succeeds at 16K, 32K, and 64K, but 64K reaches
15,800 MiB; these probes do not prove full-context prefill workspace or stability. Maximum observed power is
174.15 W and temperature is 74 C.

Decision: retain this as an exploratory performance/fit characterization, not a quality-accepted baseline. Native
Q4_0 dispatch is not profiled, quality is unevaluated, and llama-bench provides aggregate rather than per-token
latency distributions. Full statistics and provenance are in
`benchmarks/baselines/llama_cpp/gemma4-26b-a4b-qat-q4_0-characterization.json`; raw local evidence is under
`benchmarks/results/2026-08-04/1ffabc4/gemma4-26b-a4b-qat-q4_0-llama-max-power/`.

## 2026-08-03 Windows: adjacent gem16 versus llama.cpp b10240 refresh

Gem16 commit `1ffabc4` and llama.cpp b10240 (`0b14b87d7`) were rebuilt on Windows with CUDA 13.3 for SM120a and
run serially on the RTX 5080 Laptop GPU in Lenovo Max Power mode. Both received the exact 16,384-token Wikipedia
prompt, generated 1,135 fixed greedy target positions with D2 MTP, and used three warm-ups plus ten measured runs.
The GPU cooled to 50 C before each engine; 200 ms telemetry confirms both reached approximately 175 W.

Gem16 reaches median 6,047.04 prompt tok/s, 2,709.43 ms TTFT, 90.949 effective D2 tok/s, and 10.995 ms ITL.
Llama.cpp reaches 3,940.28 prompt tok/s, 4,158.08 ms TTFT, 86.775 effective D2 tok/s, and 11.524 ms ITL. Gem16 is
therefore 53.47% faster in prefill and 4.81% faster in decode, with 34.84% lower TTFT and 4.59% lower ITL. Mean
95% intervals are `[6,042.78, 6,055.66]` versus `[3,935.87, 3,946.22]` prompt tok/s and
`[90.941, 90.961]` versus `[86.730, 86.818]` D2 tok/s.

All ten outputs are deterministic within each engine. Gem16 records 1,016 proposed / 625 accepted / 391 rejected
drafts and 509 Target batches; llama.cpp records 1,035 / 616 / 419 and 519 batches. Their respective output hashes
match the current Linux characterization. Active telemetry measures 154.91/145.91 W mean power,
2,534/2,111 MHz mean SM clock, and 11,820/10,586 MiB sampled peak VRAM for gem16/llama.cpp.

This is a controlled performance comparison, not exact tensor-format parity: gem16 uses the direct FP8/NVFP4
checkpoint and FP8 KV, while llama.cpp maps attention weights and KV to Q8_0. The new deterministic Windows GGUFs
are recorded separately from their platform-specific Linux whole-file hashes. Full distributions and telemetry
summaries are in `benchmarks/baselines/cross_engine_mtp/windows-characterization.json`; raw ignored evidence is
under `benchmarks/results/2026-08-03/1ffabc4/blackwell16gb-windows-maxpower-cross-engine-mtp-b10240-3x10/`.

## 2026-08-03 Linux max-power vLLM 0.26.0 / llama.cpp b10240 refresh

Hypothesis: Rebuilding the two external runtimes at their latest stable pins and rerunning the exact public 16K
fixed-D2 workload will establish whether gem16's current decode lead survives an adjacent same-machine comparison.
The refresh uses vLLM 0.26.0 with Torch 2.11.0, Transformers 5.14.1, compressed-tensors 0.17.0, and the audited
Gemma 4 graph-suppression patch; llama.cpp is release b10240 at
`0b14b87d7c20cb753b94b96854dd7b45306fc696` with regenerated target/assistant GGUFs.

On the Linux RTX 5080 Laptop in firmware `max-power` mode with `nvidia-powerd` active, the exact 16,384-token
Wikipedia prompt plus 1,135 fixed output positions completes three warm-ups and ten measurements per engine:

| Engine | Prefill tok/s | TTFT | Effective D2 tok/s | ITL | Sampled peak VRAM |
|---|---:|---:|---:|---:|---:|
| vLLM 0.26.0 | **6,247.55** | **2,622.47 ms** | 81.95 | 12.202 ms | 15,465 MiB |
| **gem16 `8e86cb38`** | 5,863.59 | 2,794.19 ms | **89.58** | **11.163 ms** | 11,867 MiB |
| llama.cpp b10240 | 3,922.61 | 4,176.81 ms | 82.88 | 12.065 ms | 10,631 MiB |

Gem16 decode is 9.31% faster than vLLM and 8.08% faster than llama.cpp; median ITL is 8.51% and 7.48% lower.
Gem16 prefill is 6.15% below vLLM and 49.48% above llama.cpp. Every engine produces one stable internal output
hash across all ten measurements. Gem16 reports 1,016 proposed, 625 accepted, and 391 rejected drafts over 509
Target batches; vLLM reports 1,083/590/493 over 542 batches; llama.cpp reports 1,035/616/419 over 519 batches.
Proposed drafts are not counted as output throughput.

A vLLM cold start initially exhausted the 64 GiB no-swap host by launching many concurrent memory-heavy `cicc`
processes. The retained harness caps startup to four compiler jobs and one internal NVCC thread per job; a 48 GiB
systemd scope prevents a third-party JIT failure from evicting the desktop. `gpu_memory_utilization=0.98` lacked
warm-up reserve, while 0.92 safely provisions 19,069 FP8-KV tokens for the 17,519-position workload. FlashInfer
still reports autotuning OOM fallbacks and an untuned 8K NVFP4 shape; both remain disclosed in raw logs.

Decision: publish this as a controlled same-machine performance comparison, not exact output/semantic parity.
Gem16/vLLM use direct FP8/NVFP4 plus FP8 KV; llama.cpp maps attention to Q8_0 and uses Q8_0 KV. Prefill boundaries
and output hashes differ, and external MTP does not preserve each external runtime's ordinary sequence. Raw data is
under `benchmarks/results/2026-08-03/8e86cb38/blackwell16gb-linux-maxpower-cross-engine-mtp-v026-b10240-3x10/`.
The engine binary is exact commit `8e86cb38`; the run records three dirty benchmark pin/script entries.

## 2026-08-03 Prefill: store recurrent hidden and residual streams as physical BF16

Hypothesis: checkpoint-FP8 prefill stores the recurrent `hidden_a`/`hidden_b` streams as BF16-valued FP32 even
though both residual boundaries round every value to BF16. Keeping those streams as physical BF16 should halve
their recurring traffic, let the fused FP8/NVFP4 RMSNorm quantizers consume the final representation directly,
and remove the unused full-chunk `post_norm` allocation without changing arithmetic or decode.

On the Windows RTX 5080 Laptop in Max Power mode, the serial 16K synthetic qualification with three warm-ups and
ten measured runs improves from 5,980.87 to 6,045.67 prompt tok/s (+1.08%) and reduces median TTFT from 2,739.40
to 2,710.04 ms (-1.07%). Mean throughput and 95% intervals are 5,984.21 `[5,975.65,5,992.76]` for the parent and
6,046.11 `[6,036.98,6,055.24]` for the candidate. An earlier adjacent one-warm-up/three-run screen independently
measures 6,069.32 versus 6,102.76 tok/s (+0.55%) with non-overlapping intervals. Every measured run emits token
`1896`.

The exact 16,384-token Wikipedia prompt still emits token `61684`; its one-output characterization completes in
2,658.84 ms. The complete CUDA suite proves bit identity between BF16-valued FP32 and physical-BF16 inputs at the
fused FP8 RMSNorm quantizer, fused NVFP4 RMSNorm quantizer, and residual/RMSNorm boundary with and without the
layer scalar. A 16K/64 decode screen retains checksum `7409890874386593231` and reaches 47.799 tok/s, confirming
that decode and fixed-address graph semantics are unchanged.

The 16K workspace falls from 1,963,502,848 to 1,711,844,608 bytes: exactly 240 MiB removed. Two recurrent 8K
buffers each fall from FP32 to BF16, and the unused 120 MiB prefill `post_norm` region is removed. In adjacent
128-token Nsight traces, the 192 residual calls fall from 1.975 to 1.851 ms (-6.28%), the 96 NVFP4 RMSNorm
quantizers fall from 0.685 to 0.579 ms (-15.56%), and the BF16 embedding pair falls from 11.46 to 10.72 us; the
FP8 RMSNorm quantizer is neutral at 1.094 versus 1.102 ms. The new residual kernel uses 24 registers, 2 KiB shared
memory, and zero stack/local memory. Decision: promote physical-BF16 recurrent prefill storage for checkpoint-FP8;
MTP verification and BF16 correctness retain FP32 storage. Raw ignored evidence is under `build/step7-*`.

## 2026-08-03 Prefill: fuse the Up epilogue into gated GELU and NVFP4 quantization

Hypothesis: the prefill MLP materializes the complete 240 MiB BF16 Up projection, then launches a separate
GELU-times-Gate plus NVFP4 quantizer and a scale-interleave kernel before the Down projection. A custom SM120
CUTLASS epilogue can instead read the stored BF16 Gate projection as C, round the Up accumulator at the existing
BF16 boundary, apply the required BF16 GELU-tanh and product boundaries, and emit the Down projection's packed
E2M1 activations plus E4M3 scales directly in CUTLASS interleaved order. The device-resident Down input divisor is
passed to CUTLASS's block-scale store, avoiding any host scalar transfer.

On the Windows RTX 5080 Laptop in Max Power mode, the 16K synthetic qualification with three warm-ups and ten
measured runs improves from 5,821.19 to 5,962.64 prompt tok/s (+2.43%) and reduces median TTFT from 2,814.54 to
2,747.78 ms (-2.37%). Mean throughput and 95% intervals are 5,822.28 `[5,807.14,5,837.43]` for the parent and
5,962.81 `[5,952.73,5,972.89]` for the candidate. Two alternating one-warm-up/three-run screens independently
measure 6,035.32 versus 5,898.13 tok/s (+2.33%) and 5,990.55 versus 5,842.77 tok/s (+2.53%). Every run emits the
same first token, `1896`.

The exact 16,384-token Wikipedia one-output qualification reaches 6,041.33 tok/s and 2,711.99 ms median TTFT over
three warm-ups and ten measured runs. Every run emits token `61684` and SHA-256
`584cbf379f6308a52a4e7790140edace9072e661dcd02af44cd5b9369afa4182`. A dedicated CUDA fixture proves that the
new epilogue's packed E2M1 payload and interleaved E4M3 scales are bit-identical to the former separate BF16
Gate/Up projections, gated-GELU quantizer, and scale interleave.

The prefill workspace falls from 2,207,296,768 to 1,963,502,848 bytes, removing 232.5 MiB net. A 128-token Nsight
trace observes 96 custom fused CUTLASS calls over warm-up plus measurement and no separate gated-GELU quantizer or
Down-input scale-interleave launch. The fused kernel averages 43.97 us; resource inspection reports 160 registers,
zero stack, and zero local memory. The remaining 96 scale-interleave calls belong only to MLP-input activations and
are the next isolated target. Complete CUDA tests pass. Decision: promote the fused epilogue and remove the full
BF16 Up arena; no legacy production switch remains. Raw ignored evidence is under `build/step4-*`.

## 2026-08-03 Prefill: store attention output directly as physical BF16

Hypothesis: online local/global attention already closes at Gemma's required BF16 boundary before the O projection,
so writing FP32, launching a separate BF16-rounding kernel, and rereading FP32 wastes traffic and arena space. The
promoted kernels write paired physical BF16 values directly and the existing per-token O-input quantizer now has a
physical-BF16 input specialization. MTP verification and BF16 correctness mode retain their FP32 contracts.

On the Windows RTX 5080 Laptop in Max Power mode, the current-head 16K parent reaches 5,741.36 prompt tok/s and
2,853.68 ms median TTFT over three warm-ups and ten measured runs. The candidate reaches 5,825.66 tok/s (+1.47%)
and 2,812.39 ms (-1.45%); throughput 95% intervals are `[5,728.00,5,762.80]` and
`[5,810.80,5,841.73]`. Two alternating one-warm-up/three-run screens independently put the candidate at
5,882.20 and 5,866.99 tok/s versus adjacent parent results of 5,813.87 and 5,750.49 tok/s, excluding the observed
downward thermal drift as the explanation.

The exact 16,384-token Wikipedia one-output qualification reaches 5,868.77 tok/s and 2,791.73 ms median TTFT over
three warm-ups and ten measured runs. Every run emits token `61684` and the retained one-token SHA-256
`584cbf379f6308a52a4e7790140edace9072e661dcd02af44cd5b9369afa4182`. A dedicated CUDA fixture proves the
physical-BF16 and former BF16-valued-FP32 quantizers produce bit-identical E4M3 bytes and FP32 scales.

Workspace falls from 2,341,514,496 to 2,207,296,768 bytes: exactly 128 MiB removed. A 128-token Nsight trace
observes 80 local and 16 global prefill-attention kernels writing `unsigned short`, followed by 96
`QuantizeTokenReferenceKernel<unsigned short>` calls and no `RoundBf16Kernel`. Complete Host/CUDA CTest passes.
Decision: promote the physical-BF16 attention boundary as the sole checkpoint-FP8 prefill path; no legacy switch
remains. Raw ignored evidence is under `build/point1-*`.

## 2026-08-03 Five-phase performance sprint closure

The sprint closes at commit `2be75d7` with a Windows Max Power 16K/1,135-token qualification using three warm-ups
and ten measured runs. Current gem16 reaches 5,677.61 prefill tok/s, 2,885.72 ms TTFT, 91.462 effective fixed-D2
tok/s, and 10.933 ms median ITL. All ten runs retain the ordinary-Target hash
`374a7e9a564421be4f7d19cb125a651f73505077983b77b1149bfa82e3c81e8a`, exact 1,006/629/377
proposed/accepted/rejected counters, 505 Target batches, no fallback, and no token-loop allocation. Workspace and
KV remain 2,348,879,872 and 311,287,808 bytes.

For orientation against the latest pinned same-laptop Max Power competitor characterizations:

| Engine / retained environment | Prefill tok/s | TTFT | Effective D2 tok/s | ITL |
|---|---:|---:|---:|---:|
| **gem16 `2be75d7`, Windows** | 5,677.61 | 2,885.72 ms | **91.462** | **10.933 ms** |
| vLLM 0.25.1, Linux | **6,308.53** | **2,597.12 ms** | 81.964 | 12.201 ms |
| llama.cpp 10210, Linux | 3,947.45 | 4,150.53 ms | 84.179 | 11.879 ms |

Gem16 decode is 11.59% above vLLM and 8.65% above llama.cpp; its median ITL is 10.39% and 7.96% lower. Gem16
prefill is 43.83% above llama.cpp but remains 10.00% below vLLM, with TTFT 11.11% higher than vLLM. The ideal goal
is therefore achieved for decode and against llama.cpp in both disciplines, but not yet for prefill against vLLM.
These ratios are a pinned cross-day, cross-OS orientation rather than a new simultaneous parity run; checkpoint,
KV precision, output semantics, and timing-boundary disclosures remain those of the public cross-engine
characterization below.

Relative to that earlier public gem16 row at 5,315.11/85.261 tok/s, this sprint improves prefill by 6.82%, decode
by 7.27%, TTFT by 6.38%, and ITL by 6.78%. Promoted work comprises the CUTLASS FP8 scaling epilogue, physical BF16
prefill output, and vector fixed-T3 FP8 staging/scale/store path. The D512 CUTLASS FMHA/2SM route is unsupported by
the pinned backend, the narrow shape-specific CUTLASS plan regresses, and paired assistant-output loads are neutral;
all three were fully removed and retained only as negative evidence. The next performance campaign should target
the remaining approximately 631 prefill tok/s / 289 ms TTFT gap to vLLM from a fresh current-head profile.

## 2026-08-03 Decode phase: vectorize the fixed-T3 FP8 staging and output epilogue

Hypothesis: the exact fixed-three-row Target verifier already reuses each FP8 weight fragment across all rows, but
its CTA-local activation copy moves only four bytes per loop iteration and its output epilogue reloads two BF16
channel scales and writes two scalar FP32 values for every token. All production activation, scale, and output
regions are 256-byte aligned; all supported projection widths and output rows are compatible with 16-byte staging,
paired BF16 scales, and aligned `float2` output stores.

The promoted kernel copies the three E4M3 activation rows to the existing dynamic shared-memory region as `uint4`,
loads the two invariant channel scales once as `__nv_bfloat162`, and writes each adjacent output pair as `float2`.
It retains every E4M3 input byte, FP32 MMA accumulator, per-token activation scale, left-to-right two-multiply
scaling order, output value, grid shape, MMA chain, and synchronization boundary. Shared-memory and workspace
sizes are unchanged.

The 16K/64-token one-warm-up/three-run screen improves from 94.524 to 100.075 effective tok/s (+5.87%). Parent
and candidate both produce SHA-256 `6c60019e7631df51734518bda3b56fcb694e69f217a04ebdf896b10f16914aac`,
48 proposed, 37 accepted, 11 rejected, and 26 Target batches. An intermediate `uint4`-only screen reaches 94.826
tok/s and reduces the profiled T3 family by just 0.93%, showing that the scale/store epilogue supplies the material
gain.

In adjacent same-day 16K/32-token Nsight traces, 1,152 fixed-T3 calls fall from 62.652 ms total / 54.385 us mean to
40.567 ms / 35.214 us (-35.25%, approximately 1.84 ms per D2 group). Unchanged assistant GEMV, assistant head,
and Target head timings remain within approximately 1%, excluding a clock-state explanation. Registers rise from
42 to 44, dynamic shared memory remains shape-dependent at 11,520/12,288/24,576 bytes, and local memory remains
zero.

The final Windows Max Power 16K/1,135-token qualification uses three warm-ups and ten measured runs. Median decode
rises from the physical-BF16 parent's 86.393 to 91.462 tok/s (+5.87%); the non-overlapping 95% intervals are
`[86.349,86.414]` and `[91.433,91.498]`. Median ITL falls from 11.575 to 10.933 ms (-5.54%). Prefill is unchanged
within noise at 5,677.61 tok/s. All ten runs produce the same 1,135 Target IDs and SHA-256
`374a7e9a564421be4f7d19cb125a651f73505077983b77b1149bfa82e3c81e8a`, with exactly 1,006 proposed, 629
accepted, 377 rejected, and 505 Target batches. The benchmark runner rejects any fallback or token-loop allocation.
Workspace remains 2,348,879,872 bytes and KV remains 311,287,808 bytes. Full Host/CUDA CTest passes.

Decision: promote vector staging plus paired-scale/paired-store epilogue as the sole fixed-T3 FP8 path. Runtime JSON
reports `decode_order_fp8_t3_vector_stage_scale_store_qkv_o_nvfp4_down8`. Raw ignored screens, qualification, and
profiles are under `benchmarks/results/2026-08-03/aedb033-worktree/blackwell16gb-windows-fixed-t3-fp8-*`.

## 2026-08-03 Decode phase: reject paired loads in the assistant output head

Hypothesis: the official assistant's 1,024-wide tied-BF16 output head can consume adjacent coefficients as
`__nv_bfloat162` and adjacent FP32 hidden values as `float2`. The candidate executes 16 paired loop iterations per
lane instead of 32 scalar-load iterations while retaining 32 scalar FP32 FMAs, the warp candidate reduction,
suppression, tie break, and GPU argmax.

The 16K/64-token one-warm-up/three-run D2 screen is exact: parent and candidate both produce SHA-256
`6c60019e7631df51734518bda3b56fcb694e69f217a04ebdf896b10f16914aac`, 48 proposed, 37 accepted, 11 rejected,
and 26 Target batches. Median throughput moves only from 94.524 to 94.717 tok/s (+0.20%). Full Host/CUDA CTest
passes for the candidate.

The required 16K/1,135-token three-warm-up/ten-run qualification reaches 86.384 tok/s with 95% interval
`[86.355,86.438]`. The unchanged physical-BF16 parent is 86.393 tok/s with interval `[86.349,86.414]`, so the
candidate is -0.01% with fully overlapping intervals. All ten runs retain the complete Target hash
`374a7e9a564421be4f7d19cb125a651f73505077983b77b1149bfa82e3c81e8a`, 1,006/629/377 proposed/accepted/rejected
drafts, and 505 Target batches. Workspace, KV bytes, recurring allocations, and fallbacks are unchanged.

An initial cross-day profile appeared to reduce the assistant head by 35%, but unchanged BF16 GEMV and Target-head
kernels accelerated by a similar amount and exposed a power/clock confound. The final adjacent same-day Nsight
pair measures 24 head calls at 14.680 ms total / 611.669 us mean for the scalar parent and 14.790 ms / 616.259 us
for the paired candidate (+0.75%). Both use 39 registers, 64 bytes static shared memory, and zero local memory.
Decision: remove the vector candidate and retain the scalar warp-row head. Raw ignored evidence is under
`benchmarks/results/2026-08-03/82139e9-worktree/blackwell16gb-windows-assistant-output-bf16x2-*`.

## 2026-08-02 Projection-plan phase: reject a separate FP8 N64 plan for narrow K/V

Hypothesis: Gemma's narrow FP8 K/V output shapes (N=512 or 2,048) could benefit from CUTLASS's shipped
M128xN64xK64 SM120 geometry while the wider Q/O shapes retained the promoted M128xN128xK64 plan. A temporary
static selector instantiated N64 only for N<=2,048; all other projection shapes and all decode kernels were
unchanged.

The complete Host/CUDA CTest suite passes, including exact physical-BF16 FP8 epilogue bits. The 16K Windows Max
Power fixed-D2 screen (one warm-up, three measured runs) reaches 5,722.85 prompt tok/s with samples
5,746.35/5,722.85/5,698.11 and a 95% interval of `[5,662.50,5,782.37]`. Median TTFT is 2,862.91 ms and decode is
86.384 tok/s. The result retains 1,135 generated IDs and SHA-256
`374a7e9a564421be4f7d19cb125a651f73505077983b77b1149bfa82e3c81e8a`.

This does not establish a speedup: the candidate lies inside the same-session N128 observation range of roughly
5,675 to 5,752 prompt tok/s, its interval overlaps the retained parent qualifications, and its three samples show
the same downward thermal drift. Adding another large CUTLASS instantiation without measurable end-to-end value
would increase binary/build complexity. The selector and N64 instantiation are therefore removed; M128xN128xK64
remains the sole FP8 prefill plan. The ignored raw screen is
`benchmarks/results/2026-08-02/e16f84a-worktree/blackwell16gb-windows-cutlass-plan-narrow-n64-screen.json`.

## 2026-08-02 Global-attention phase 1: reject direct CUTLASS D512 FMHA substitution

Hypothesis: CUTLASS's Blackwell warp-specialized FMHA examples could replace the manual Gemma global-attention
kernel and provide a bounded path to 2SM scheduling. The production kernel is worth attacking: the retained 16K
prefill profile attributes approximately 1,969.67 ms across two prompt executions to global attention, while the
retained 16K decode trace attributes approximately 3.017 s across 128 global-attention instances, about 33% of
that trace. The current D512 kernel uses four query heads per CTA, 256 threads, 96 KiB static shared memory, and
254 registers without local-memory spills.

A standalone SM120a compile probe instantiated regular CUTLASS FMHA with a `256x128x512` QK tile. Compilation
stops at the library's UMMA shape gate: the corresponding PV operation is `M128,N512,K128`, but its N dimension
must be at most 256. The regular example is a 1SM design and its shipped head-dimension instantiations stop at
128, so it cannot express Gemma's D512 output by changing a dispatcher constant. The temporary probe was removed
after recording the failure.

CUTLASS's available 2SM D512 implementation is the MLA inference path, not standard FMHA. It assumes one query,
128 heads, a paged latent-cache contract, separate latent and RoPE dimensions, and (for the FP8 route) FP8 Q.
Gemma instead supplies 16 BF16 query heads and scaled FP8 K/V under the engine's contiguous-cache contract. The
prefill MLA example only covers a 128-wide latent dimension. Adapting that path would therefore add quantization
or cache expansion and large head-count waste before any attention work; it is not a correctness-preserving
kernel substitution.

No executable candidate passed the compile and semantic gates, so no runtime speedup is claimed and no favorable
timing was manufactured. The production kernel remains unchanged. A viable future D512 implementation requires a
purpose-built 2SM design with two N256 PV output passes (or an equivalent tiled reduction) that preserves BF16 Q,
checkpoint-scaled FP8 K/V, online-softmax order, and the existing cache semantics.

## 2026-08-02 Prefill phase 8: store FP8 projection boundaries as physical BF16

Hypothesis: the fused Q/K/V/O CUTLASS epilogue already produces the exact model-required BF16 value, so retaining
that value in an FP32 arena wastes half the projection traffic and storage. The promoted path stores physical BF16
directly, teaches V RMSNorm, fused Q/K RMSNorm/RoPE, and O residual/RMSNorm to consume the 16-bit boundary, and
leaves the fixed-T3 MTP verifier on its existing FP32 projection contract. Decode kernels and arithmetic are
unchanged.

Against the immediately preceding fused-epilogue parent, the exact Windows Max Power 16K fixed-output D2
qualification moves from 5,665.28 to 5,703.40 prompt tok/s (+0.67%) and from 2,892.00 to 2,872.67 ms median TTFT
(-0.67%). The parent throughput 95% CI is `[5,659.30,5,672.38]`; the physical-BF16 candidate interval is
`[5,699.43,5,710.69]`. Decode is effectively unchanged at 86.325 versus 86.377 tok/s. All ten candidate runs retain
1,135 IDs, SHA-256 `374a7e9a564421be4f7d19cb125a651f73505077983b77b1149bfa82e3c81e8a`, 629 accepted of 1,006 proposed
drafts, and 505 Target batches.

A final exact-source repeat after the test/metadata additions reaches 5,675.51 tok/s (+0.18%) and 2,886.79 ms
(-0.18%), with throughput CI `[5,665.76,5,690.84]` overlapping the parent. Its samples decline monotonically from
5,708.21 to 5,658.22 tok/s after the extended qualification session, so the timing claim is conservatively a small
positive effect rather than an unconditional +0.67%. It retains the same IDs, hash, MTP counters, and memory plan.
The promotion is justified by the material deterministic memory reduction plus non-regressing repeated throughput,
not by selecting only the faster run.

The 16K workspace falls from 2,613,121,024 to 2,348,879,872 bytes: 264,241,152 bytes (252 MiB, 10.11%) removed,
with the same 311,287,808-byte KV allocation. Runtime JSON now distinguishes the numerical
`fp8_prefill_output=scaled_bf16` boundary from `fp8_prefill_storage=physical_bf16`. A bounded Nsight Systems trace
over one warm-up and one measured 128-token prompt observes the physical consumers directly: 96
`ProjectionRmsNormRotaryBf16BatchKernel<uint16_t>`, 96 `RmsNormKernel<uint16_t,true>`, and 192
`RmsNormResidualBf16Kernel<uint16_t>` instances across the two prompt executions. No BF16-to-FP32 expansion kernel
exists; the 96 remaining round kernels close attention outputs. The attempted 16K Windows trace completed GPU
work but hung during Nsight report finalization and produced no artifact, so it is not cited as evidence.

Host/CUDA CTest passes with exact physical-BF16 CUTLASS output bits and direct physical-input RMSNorm and Q/K
RMSNorm/RoPE comparisons. Direct 129/257 vLLM boundary validation remains Top-1 at engine rank one with selected
logprob deltas 0.5643/0.4235. Decision: promote physical BF16 Q/K/V/O prefill storage. Raw timing and trace evidence
is under `benchmarks/results/2026-08-02/8ed71e6-worktree/blackwell16gb-windows-physical-bf16-*`.

## 2026-08-02 Prefill phase 7: fuse FP8 scaling and BF16 rounding into CUTLASS

Hypothesis: every production FP8 Q/K/V/O prompt projection already ends at the exact scaled-BF16 boundary, so a
CUTLASS EVT can broadcast the per-token FP32 activation scale and per-channel BF16 weight scale, preserve the
left-to-right FP32 multiplication order, round the final fragment to BF16, and eliminate the separate scale/cast
kernel. The BF16 value remains in the existing FP32 workspace in this phase; physical BF16 storage is evaluated
separately so its consumer rewrites cannot obscure the epilogue result.

On the Windows RTX 5080 Laptop in the firmware Max Power state, the same-commit 16K fixed-output D2 parent reaches
5,417.10 prompt tok/s (mean 5,419.28; 95% CI `[5,405.15,5,433.42]`) and 3,024.495 ms median TTFT. The adjacent
3-warm-up/10-run EVT candidate reaches 5,665.28 tok/s (+4.58%; mean 5,665.84; CI `[5,659.30,5,672.38]`) and
2,892.00 ms TTFT (-4.38%; CI `[2,888.39,2,895.06]`). Decode is unchanged within noise at 86.311 versus
86.325 tok/s. All ten candidate runs retain 1,135 IDs, SHA-256
`374a7e9a564421be4f7d19cb125a651f73505077983b77b1149bfa82e3c81e8a`, 629 accepted of 1,006 proposed drafts,
and 505 Target batches.

Nsight Systems over one warm-up plus one measured 16K prefill records 736 fused FP8 GEMMs and no standalone FP8
scale kernel. Only the 192 attention-boundary `RoundBf16Kernel` calls remain. The fused FP8 family consumes
1,077.50 ms of the two prompts; the profile remains dominated by global attention (1,969.67 ms), NVFP4 CUTLASS
(1,113.43 ms), and local attention (642.27 ms). Host/CUDA CTest passes, including exact fused-BF16 comparisons
for the small and real 128x4,096x3,840 FP8 geometries. Direct 129/257 vLLM boundary validation keeps both
reference Top-1 tokens at engine rank one (selected-logprob absolute deltas 0.5643/0.4235). Workspace and KV
allocations remain exactly
2,613,121,024 and 311,287,808 bytes, and the token loop is unchanged.

Decision: promote the fused scaled-BF16 CUTLASS epilogue as the sole production FP8 prompt projection. Remove the
separate production scale kernel and its unrounded selector. Raw Windows evidence is under
`benchmarks/results/2026-08-02/e331687-worktree/blackwell16gb-windows-fp8-fused-epilogue-*`. These Windows timings
are a development characterization; Linux remains the publication/reference qualification environment.

## 2026-08-02 Prefill phase 6: reject prefill CUDA Graphs

Hypothesis: after fixing the 8K projection and attention geometry, CUDA Graph replay can remove enough recurring
host launch work to improve the exact 16K end-to-end boundary. Three exact, fixed-address candidates were tested:
48 layer-specific suffix graphs covering O/MLP/residual work; paired prefix/suffix graphs; and two complete 48-layer
chunk graphs specialized for absolute starts 0 and 8,192. Arbitrary shapes and positions retained the direct path.
No candidate changed kernel arguments, arithmetic, cache semantics, output IDs, or arena addresses.

The short screens reached 5,437.39/5,428.69/5,424.82 tok/s for suffix, prefix-plus-suffix, and full-chunk capture.
Only the suffix candidate warranted repetition. In the reproducible adjacent rerun, detached `74972ac` reaches a
5,385.37 tok/s median (mean 5,386.48; 95% CI `[5,382.20,5,390.76]`) and 3,042.315 ms TTFT. The suffix candidate
reaches 5,390.25 tok/s (mean 5,390.16; CI `[5,384.45,5,395.87]`) and 3,039.56 ms. This is only +0.09% throughput
and -0.09% TTFT with overlapping intervals. An earlier low parent run at 5,075.59 tok/s did not reproduce and is
not used as evidence. All measured outputs retain one hash.

The suffix design instantiates 48 additional executables, raises graph-private device memory by 8,388,608 bytes,
and issues 96 graph replays for the two 8K chunks. Nsight still reports effectively unchanged dominant work:
1,119.67 ms CUTLASS, 1,023.42 ms global attention, and 331.12 ms local attention. The engine is GPU-kernel-bound;
removing host submissions does not produce a statistically supported end-to-end win. Prefix and complete-chunk
capture are slower even in the screen and add broader shape/position specialization.

Decision: reject all three candidates and remove their capture, replay, metadata, and graph-storage changes. Keep
the direct 8K prefill plan as the sole production path. Raw evidence is under
`benchmarks/results/2026-08-02/74972ac*/blackwell16gb-linux-prefill-phase6-*`.

## 2026-08-02 Prefill phase 5: pipeline local-attention FP8 staging

Hypothesis: the 40 sliding layers can use the global kernel's successful raw-FP8 ping-pong pattern without changing
the qualified 32-key online-softmax order. The promoted kernel overlays one BF16 K/V operand tile and two raw-FP8
tiles in the existing 64 KiB shared allocation. It overlaps current-V transfer with QK and next-K transfer with PV,
then performs the same E4M3x4 conversion, paired BF16 stores, MMA sequence, masking, and FP32 accumulation.

Against the scaled-BF16 parent's 5,271.29 tok/s and 3,108.15 ms, the 3-warm-up/10-run candidate reaches
5,379.58 tok/s (+2.05%) and 3,045.59 ms (-2.01%). Its throughput 95% CI is `[5,374.21,5,389.56]`, TTFT CI is
`[3,039.96,3,048.64]`, and all runs retain SHA-256
`584cbf379f6308a52a4e7790140edace9072e661dcd02af44cd5b9369afa4182`. Nsight reduces the 80 local-attention
kernels from 416.12 to 331.47 ms (-20.34%); the selected kernel uses 254 registers, 65,536 static shared bytes, and
zero local memory. Host/CUDA CTest, exact-blue, and direct 129/257 vLLM boundary gates pass.

Two global-attention candidates were rejected and removed. A synchronous 32-key tile was performance-neutral and
exceeded the CUDA operator error budget (max absolute 0.0337). Replacing accurate `expf` with `__expf` reached a
5,399.31 tok/s short screen but changed both direct boundary Top-1 predictions and was rejected without relaxing
tolerances. Global attention remains the largest profile family at approximately 1,026 ms.

Decision: promote asynchronous local FP8 staging as the sole production local-prefill path. Preserve the accurate
global softmax and 16-key order. Raw evidence is under
`benchmarks/results/2026-08-02/31c8519-worktree/blackwell16gb-linux-prefill-phase5-*`.

## 2026-08-02 Prefill phases 3/4: reject layer-major prep reuse; close FP8 BF16 outputs

Phase 3 hypothesis: a layer-major text plan can transform Gate/Up/Down into transient CUTLASS scratch once per
layer and reuse those bytes across both 8,192-token chunks, avoiding half of the prompt-time weight preparation
without a second persistent model layout. The candidate retained one full prompt hidden-state arena and three
prepared current-layer matrices, preserved the output hash, and reduced preparation calls as designed, but its
1-warm-up/3-run median was 4,951.66 tok/s versus the promoted parent's approximately 5,173 tok/s. The changed
whole-model traversal and larger working set outweighed preparation reuse. The complete candidate was removed and
Phase 3 produces no commit.

Phase 4 hypothesis: every CUTLASS FP8 Q/K/V/O scale output reaches an existing BF16 boundary, so the scale kernel
can perform that exact cast and eliminate standalone V and O round passes. The operator keeps an explicit FP32
mode for its independent exact test; production selects `scaled_bf16`. Against the 8K-chunk parent's 5,172.75 tok/s
and 3,167.37 ms, the adjacent 3-warm-up/10-run candidate reaches 5,271.29 tok/s (+1.90%) and 3,108.15 ms (-1.87%).
Its throughput 95% CI is `[5,263.42,5,279.11]`, TTFT CI is `[3,103.56,3,112.81]`, and all outputs retain SHA-256
`584cbf379f6308a52a4e7790140edace9072e661dcd02af44cd5b9369afa4182`.

Nsight confirms only 96 standalone `RoundBf16Kernel` calls remain at 16K instead of the 288 implied by the
8K-chunk parent; the remaining calls close the attention output. The scaled/BF16 output kernels take 109.25 ms,
while total profiled prompt execution remains dominated by global attention (1,019.82 ms), all CUTLASS GEMMs
(1,113.01 ms), and local attention (416.12 ms). Host/CUDA CTest, exact-blue, and direct 129/257 vLLM boundary gates
pass. Workspace, persistent weights, cache, and decode kernels are unchanged.

Decision: promote the exact FP8 scaled-BF16 output boundary and remove the redundant V/O casts. Keep the rejected
layer-major candidate absent. Next rebuild the profile-dominant global prefill attention rather than adding another
layout-preparation mechanism. Raw evidence is under
`benchmarks/results/2026-08-02/22b3727-worktree/blackwell16gb-linux-prefill-phase{3,4}-*`.

## 2026-08-02 Prefill phase 2: promote an 8,192-token checkpoint-FP8 chunk

Hypothesis: the current 2,048-token prompt plan repeats all 48 layers eight times for the 16K workload, while vLLM
uses an 8,192-token scheduling budget. The measured max-power parent at `0c0ec10` reaches 5,034.80 tok/s median and
3,254.15 ms TTFT over three warm-ups and ten measured runs. A 4,096-token screen reaches 5,139.29 tok/s. The final
8,192-token candidate reaches 5,172.75 tok/s (+2.74%) and 3,167.37 ms (-2.67%) with 95% throughput CI
`[5,165.02,5,183.88]` and TTFT CI `[3,160.59,3,172.11]`; neither interval overlaps the parent. All measured runs
retain the same one-token SHA-256 `584cbf379f6308a52a4e7790140edace9072e661dcd02af44cd5b9369afa4182`.

The wider plan increases total MTP-resident workspace from 726,836,224 to 2,611,102,720 bytes. External 200 ms
telemetry measures a 12,704 MiB process peak with target and assistant resident, below the 15.3 GB limit. Host and
CUDA CTest pass, and the 129/257 direct-vLLM prefill boundary cases retain Top-1 rank one after updating the runtime
plan invariant. An independently tested M256xN128xK128 CUTLASS NVFP4 plan falls to 2,823.61 tok/s and is removed. A
single N=30,720 Gate/Up CUTLASS GEMM is performance-neutral at 5,239.68 versus 5,239.19 tok/s in adjacent short
screens and is also removed rather than increasing scratch memory.

Decision: make 8,192 the sole checkpoint-FP8 prefill chunk. Keep BF16 correctness mode unchanged. Next remove
recurring prompt-time NVFP4 weight preparation without retaining a second persistent model layout. Raw evidence is
under `benchmarks/results/2026-08-02/0c0ec10/blackwell16gb-linux-prefill-phase2-*`.

## 2026-08-02 Linux max-power cross-engine D2 MTP characterization

Hypothesis: the large retained Windows/Linux performance discrepancy is primarily a laptop power-policy problem,
not an operating-system kernel advantage. On the reference Lenovo Legion, Linux exposed `max-power` but kept the
GPU at its 80 W default because the installed `nvidia-powerd` Dynamic Boost service was disabled. Enabling that
service raised the firmware-managed loaded ceiling to 175 W; the same current-head 16K/64 development screen rose
from 57.425 to 85.769 effective tok/s (+49.4%). Active sampled SM clock rose from about 1,572 to 2,550 MHz while
maximum temperature remained 64 C in the bounded confirmation.

The post-prefill-optimization refresh uses commit `4b237b16366b1a9ee2cd339f0549e06a7cfc69aa`, the exact
16,384-token Wikipedia prompt with little-endian-uint32 SHA-256
`d07ad4d805944f0b87869da0c5bb44d99e8c43c0eb57d05a108ad80a6abb51a8`, 1,135 fixed greedy output positions,
batch one, fixed D2 MTP, three warm-ups, and ten measured runs per engine. Proposed tokens are not counted.

| Engine | Prefill median | TTFT median | Effective D2 median | ITL median | Accepted/proposed | Target batches |
|---|---:|---:|---:|---:|---:|---:|
| gem16 | 5,315.11 tok/s | 3,082.53 ms | **85.261 tok/s** | **11.729 ms** | 629/1,006 | 505 |
| vLLM 0.25.1 | **6,308.53 tok/s** | **2,597.12 ms** | 81.964 tok/s | 12.201 ms | 590/1,083 | 542 |
| llama.cpp 10210 | 3,947.45 tok/s | 4,150.53 ms | 84.179 tok/s | 11.879 ms | 616/1,035 | 519 |

Gem16 decode is 4.02% above vLLM and 1.29% above llama.cpp in this workload; gem16 prefill is 15.75% below vLLM
and 34.65% above llama.cpp. Relative to the prior public gem16 row, prefill rises 7.06% and TTFT falls 6.60%; the
external rows remain effectively stable. All ten outputs are deterministic within each runtime. Current gem16
retains one fixed 1,135-token ordinary-Target SHA-256
`374a7e9a564421be4f7d19cb125a651f73505077983b77b1149bfa82e3c81e8a`; external hashes differ, and prior gates
show their MTP paths differ from their own ordinary Target output.

Format and timing disclosure: gem16 and vLLM consume the direct mixed FP8/NVFP4 checkpoint with FP8 KV. llama.cpp
uses the patched GGUF with 144 native NVFP4 MLP tensors, 184 Q8_0 attention tensors, and Q8_0 KV. llama.cpp's
prefill boundary is narrower. Whole-command 200 ms telemetry samples maximum VRAM at 12,720/15,142/10,630 MiB
and maximum power at 177.67/179.60/176.23 W for gem16/vLLM/llama.cpp respectively; startup/autotuning is included
in those maxima.

Decision: retain this as a prominent reproducible development characterization, not a format- or quality-parity
headline. Require `nvidia-powerd` plus the disclosed platform profile for direct reproduction on this laptop.
`README.md`, `scripts/benchmark-cross-engine-mtp.sh`, the checked-in exact-token workload, and
`benchmarks/baselines/cross_engine_mtp/characterization.json` form the public reproduction boundary. Raw JSON,
console logs, and telemetry remain under
`benchmarks/results/2026-08-02/4b237b1/blackwell16gb-linux-maxpower-cross-engine-mtp-prefill-refresh/`.

## 2026-08-01 Six-phase Windows final qualification

The ordered six-phase program completed with one promoted production change: Phase 5's assistant BF16x2 GEMV.
Phases 1, 2, 3, 4, and 6 were fully reverted after their correctness or performance gates failed, and each negative
result is retained below. The final committed source was rebuilt before measurement; the complete host/CUDA CTest
suite passed. Immediately before the run, the RTX 5080 Laptop reported 0% GPU utilization, 0 MiB allocated VRAM,
44 C, and 14.28 W.

The final exact Wikipedia 16K/1,135-token qualification used three alternating warm-up pairs and ten alternating
measured Ordinary/MTP pairs. MTP reached 63.892 tok/s median and 63.889 tok/s mean with 0.140 tok/s standard
deviation and a 95% CI of `[63.789,63.989]`. Samples were
64.168/64.030/63.894/63.890/63.909/63.950/63.723/63.703/63.812/63.811 tok/s. Ordinary reached 37.995 tok/s
median, so fixed D2 delivered a 1.682x effective-output speedup. Relative to the retained pre-program 63.091 tok/s
median, the final result is +1.27%.

Every one of the ten measured MTP runs produced all 1,135 output tokens and exactly matched Ordinary SHA-256
`43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`. Each retained the same 501 groups,
1,002 proposed, 633 accepted, and 369 rejected tokens, with no fallback or token-loop allocation. Workspace and KV
allocations were constant at 741,486,080 and 369,098,752 bytes.

Decision: the six-phase program improves the retained engine but does not meet the hard 64.82 tok/s gate. The final
gap is 0.928 tok/s, or 1.43% of the target, and `target_not_met` is retained rather than selecting the favorable
64.168 tok/s sample. The next measured assistant target is its approximately 22.8 ms output-head candidate kernel;
new D512 work should wait for Linux Nsight Compute evidence. Raw ignored final evidence is
`benchmarks/results/2026-08-01/318ed53/blackwell16gb-windows-six-phase/final-wikipedia-16k-3x10.json`.

## 2026-08-01 Phase 6: D512 decode layout and MMA decision

Profile gate: the current child-node Nsight Systems 16K/32-token trace assigns 40.990 ms to 96 fixed-D2 scalar
global-GQA split kernels, or approximately 3.416 ms per D2 group across the eight global layers. Merge adds only
0.629 ms total. Nsight Compute 2026.2 could not collect SM/DRAM counters from this Windows CUDA Graph: both basic
kernel replay and a minimal three-metric application replay terminated the instrumented process with Windows status
`0xC00000FD`, while normal execution and Nsight Systems child-node tracing remained stable. No incomplete NCU result
is treated as evidence.

MMA decision: do not repeat the existing BF16/TF32 D512 Tensor-Core prototypes unchanged. They were previously up
to 13.65% slower than retained scalar split-GQA, and the current kernel already stages one physical K/V tile for
all 16 query heads. Without compute-saturation counters, a new precision/layout MMA rewrite is not justified by
the correctness-first plan.

The bounded layout prototype instead split the 16 query heads into two eight-head CTAs per row/split. Per-head FMA,
softmax, and V accumulation order remained unchanged; score shared memory and per-thread query/accumulator state
were halved, while CTA count and K/V staging doubled. The CUDA operator suite passed. The fixed 16K Wikipedia/
64-token screen produced the exact parent output hash but reached only 62.860 tok/s (62.901/62.820), versus the
confirmed 65.742 tok/s Phase-5 parent (-4.38%).

Decision: reject and fully remove the eight-head layout. The repeated K/V staging dominates any occupancy benefit.
Retain one 16-head CTA per D2 row/split and defer cluster/DSM sharing or a new native FP8 D512 MMA design until Linux
Nsight Compute can prove the limiting resource. Phase 6 makes no production source change. Raw ignored Systems,
failed-NCU orchestration, and screen artifacts are under the corresponding 2026-08-01 six-phase result trees.

## 2026-08-01 Phase 5: vectorized assistant BF16 GEMV

Hypothesis: the four-layer MTP assistant issues 42 row-parallel BF16 GEMVs for each two-draft proposal. Its scalar
kernel loads one BF16 coefficient and one FP32 activation per thread iteration. All assistant contracting widths
are even and all checkpoint/workspace regions are 256-byte aligned, so paired `__nv_bfloat162` and `float2` loads
can halve load/conversion instructions without changing storage precision or FMA count.

The retained kernel processes two adjacent columns per thread iteration and keeps FP32 accumulation plus the final
BF16 output boundary. In a child-node Nsight Systems 16K/32-token trace, the same 504 BF16 GEMV calls fell from
22.139 ms total / 43.927 us mean to 16.760 ms / 33.255 us (24.3% lower). Assistant output-head time remained
approximately 22.8 ms and is now the larger assistant primitive. The paired mapping changes the FP32 reduction
partition but not the represented inputs or outputs.

The initial fixed 16K Wikipedia/64-token screen reached 66.257 tok/s versus the fresh 64.506 tok/s parent (+2.72%)
with identical output IDs and identical 35/54 accepted/proposed counts. A one-warm-up/three-run confirmation with
Ordinary control measured 65.742 tok/s MTP and 38.178 tok/s Ordinary with exact cross-mode output identity. The
complete host/CUDA CTest suite passes.

The exact 1,135-token Wikipedia qualification used one alternating warm-up pair and three measured pairs. MTP
reached 64.031 tok/s median versus the retained 63.091 tok/s (+1.49%); samples were
64.168/64.031/63.908 tok/s with a 95% CI of `[63.713,64.359]`. Ordinary measured 38.037 tok/s. Every run retained
all 1,135 Target output IDs and SHA-256
`43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`, with no fallback or token-loop
allocation. The assistant's reordered reduction changed one draft decision over the workload: 501 groups,
1,002 proposed, 633 accepted, and 369 rejected, versus the prior 502/1,004/632/372. This reduces Target work by
one group and is disclosed; the profiled 0.448 ms/group GEMV saving independently accounts for most of the gain.
Workspace and KV allocations remain 741,486,080 and 369,098,752 bytes.

Decision: promote BF16x2 assistant GEMV as an exact-Target intermediate improvement. The result remains 0.789 tok/s
below the 64.82 tok/s 3/10 gate, so target completion is not claimed. Raw ignored screens, qualification JSON, and
Nsight reports are under
`benchmarks/results/2026-08-01/ebb6b6c/blackwell16gb-windows-six-phase/`.

## 2026-08-01 Phase 4: fixed-T=3 Q/K/V grouping and activation reuse

Hypothesis: the existing grouped T=3 Q/K/V launch dimensions its z-axis from the largest Q projection. Local
layers therefore launch 192 CTAs although only 128 64-row matrix tiles are active, and every CTA separately stages
the same three 3,840-element FP8 activations. A compact linear tile list can remove inactive K/V CTAs; assigning
two tiles per CTA also halves activation staging traffic while preserving each output's FP8 MMA and FP32
accumulation order.

The prototype generalized the fixed-three-row SM120 projection kernel to execute a linear Q/K/V tile list. Its
two-tile form used 64 local CTAs instead of the parent's 192. The CUDA operator suite passed. A first fixed 16K
Wikipedia/64-token screen reached 65.073 tok/s (65.085/65.062) versus the fresh 64.506 tok/s parent (+0.88%) with
the exact parent output hash. The required one-warm-up/three-run confirmation with Ordinary control, however,
measured 64.542 tok/s MTP and 38.009 tok/s Ordinary: exact across modes but only +0.06% from the parent and therefore
neutral. A one-tile-per-CTA control retained 128 active CTAs while removing inactive grid entries, but regressed to
60.802 tok/s (-5.74%).

Decision: reject and fully remove both tile-list variants. The existing z-grid's apparent overlaunch is not a
material end-to-end cost, and serial activation reuse reduces useful projection parallelism enough to erase its
traffic saving. The current production kernel already shares the quantized input buffer across Q/K/V and stages it
once per active CTA; further grouping at this granularity is exhausted. Proceed to the four-layer BF16 assistant,
which is independent of the 48-layer Target projection schedule.

## 2026-08-01 Phase 3: Ordinary local attention without split/merge

Hypothesis: Ordinary local attention pays for four 256-token split outputs and a separate LSE merge in each of 40
local layers. A single 1024-token CTA per KV head can cover the complete sliding window, write the normalized
result directly, and eliminate both the partial workspace traffic and merge launch.

The prototype reused the production FP8 decode-attention implementation with a separate Ordinary-only 1024-token
chunk. It reduced local-layer parallelism from 32 to eight split/query-group CTAs and performed one full-window
FP32 maximum, exponential sum, and V accumulation. The fixed 16K Wikipedia/64-token screen completed one warm-up
and two measured runs per mode, then failed exact Ordinary/MTP output equality. A diagnostic run reached only
31.382 tok/s Ordinary, substantially below the roughly 38--39 tok/s retained path, while unchanged D2 measured
62.904 tok/s in the same thermal sequence. Ordinary output SHA-256 changed to
`82af8a3534cdb38bb2c9f583b0f4f5c8135ce53fd55813a27579722dd2a447bf`; D2 retained
`8f9675f0f0f46162407c90c6cebb0684f4b00ec22c1545045826fd4f54a4da90`.

Decision: reject and remove the direct 1024-token kernel. Avoiding the merge does not compensate for the fourfold
loss of CTA parallelism, and the changed full-window reduction also violates the exact cross-mode output gate.
An exact in-CTA emulation of the four split reductions would preserve more ordering but cannot recover the lost
parallelism indicated here, so no more complex version is justified. Proceed to projection grouping and activation
reuse, which targets the larger fixed-T=3 projection cost without altering attention reductions.

## 2026-08-01 Phase 2: Ordinary-only FP8x4 global attention at 16K

Hypothesis: four-wide physical E4M3 K/V loads already improve Ordinary global attention at the 16K capacity tier,
but were neutral for fixed D2 when a previous experiment lowered the shared dispatch threshold. Giving only the
Ordinary path a 16K FP8x4 threshold could recover that gain while leaving the MTP kernel schedule unchanged.

The candidate changed only Ordinary dispatch; D2 retained the production 64K FP8x4 threshold. The fixed 16K
Wikipedia/64-token screen completed one warm-up and two measured runs per mode, but failed its final exact
Ordinary/MTP output gate and intentionally did not publish a passing result document. A diagnostic run measured
39.289 tok/s Ordinary and 65.610 tok/s MTP. MTP retained the scalar-parent output SHA-256
`8f9675f0f0f46162407c90c6cebb0684f4b00ec22c1545045826fd4f54a4da90`, while Ordinary produced the FP8x4
SHA-256 `8091eb94a963477896854e7071b344aafc9f21b3afff92649d2c64f2d5796a8b`. This reproduces the output of the
earlier shared-threshold experiment, where both Ordinary and D2 used FP8x4 and agreed exactly, but proves that
mixing the two reduction orders breaks the current exact cross-mode contract.

Decision: reject the Ordinary-only threshold. Its source was fully removed. Keep the common scalar schedule at
16K and the common FP8x4 schedule at 64K; do not trade cross-mode reproducibility for the small Ordinary gain.
Proceed to a direct Ordinary local-attention design, where the reduction order can be retained explicitly.

## 2026-08-01 Phase 1: fixed-D2 global split-size sweep

Hypothesis: global D512 attention in fixed D2 might benefit from a D2-specific split size. The retained 512-token
split launches 96 split CTAs at 16K context; reducing the chunk to 384 or 256 raises parallelism while preserving
the existing all-head GQA kernel and split/LSE-merge structure. A larger 1024-token split was ruled out before
measurement because its all-head score buffer would exceed the practical static shared-memory budget.

The fixed 16K Wikipedia/64-token development screen used one warm-up and two measured D2 runs. The fresh 512-token
parent reached 64.506 tok/s median (64.240/64.772). A 256-token candidate reached 62.613 tok/s
(61.742/63.484), a 2.93% regression. A 384-token candidate reached 65.316 tok/s (64.987/65.646), a 1.26%
throughput improvement, but changed the greedy output SHA-256 from
`8f9675f0f0f46162407c90c6cebb0684f4b00ec22c1545045826fd4f54a4da90` to
`6756d29240c8ea399b66cd620e753e8ec04c0200938dfdc95a58b7a5a356a7c8`. The 256-token output also differed.
Each candidate was internally deterministic, so the difference is attributable to the changed floating-point
reduction partition rather than run-to-run instability.

Correctness: the complete host/CUDA CTest suite passed for the split-size implementation, but model-level greedy
token identity did not. The candidate source was therefore fully removed. Raw ignored evidence is under
`benchmarks/results/2026-08-01/224bccb/blackwell16gb-windows-six-phase/`.

Decision: reject both D2-specific split sizes. The 384-token result is not promotable under the project's
correctness-first contract even though it is faster. Retain the 512-token geometry and proceed to the
Ordinary-only FP8x4 experiment without changing D2 numerical behavior.

## 2026-08-01 Windows fixed-D2 local shared-K/V attention

Hypothesis: the three consecutive Target rows in fixed D2 traverse almost the same 1024-token local-attention
window. The retained row-at-a-time path loads physical K/V three times in every one of the 40 local layers.
Staging the union once per KV head and split should remove redundant cache traffic without changing the FP8 cache,
softmax split boundaries, or verifier semantics.

Implementation: a fixed-three-row SM120 kernel stages 258 contiguous D256 FP8 entries for each of four 256-token
splits and eight KV heads. It evaluates all six row/query-head chains over the shared K/V tile, retains the existing
FP32 dot, softmax, value-accumulation, and four-split LSE merge order, and consumes speculative K/V directly before
the normal transactional cache append. The growing-prefix path explicitly masks inactive tokens; once full, the
three rows use the expected one-slot sliding-window shifts. Ordinary decode and the eight global layers are
unchanged. A separate global shared-K/V prototype was exact but rejected at 58.185 tok/s in the 16K/64 screen
(-3.15% from its 60.078 tok/s parent).

The final 16K/64 development screen used one warm-up and three measured MTP runs and reached 65.227 tok/s median,
versus the clean 60.078 tok/s parent (+8.57%). Exact Ordinary/MTP generation checks also passed at prompt lengths
128, 1022, and 1023, covering the growing window and its full-window boundary.

The exact Wikipedia 16K/1,135-token qualification used one alternating warm-up pair and three measured pairs. MTP
improved from the retained 59.277 to 63.091 tok/s (+6.43%); Ordinary measured 38.108 tok/s, for a 1.656x MTP
speedup. MTP samples were 63.261/63.085/63.091 tok/s with a 95% CI of `[62.897,63.395]`. Every run retained all
1,135 output IDs and SHA-256 `43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`, plus the exact
1,004/632/372 proposed/accepted/rejected counts over 502 groups. The result remains 1.729 tok/s below the 64.82
tok/s gate, so it is a promoted intermediate improvement rather than target completion.

Correctness: the complete host/CUDA CTest suite passes. The kernel introduces no precision, persistent-weight,
KV-cache, recurring-allocation, or workspace-size change. Raw Windows evidence is under the ignored
`benchmarks/results/2026-08-01/becbf9e/blackwell16gb-windows-mtp-shared-kv/` directory.

Decision: promote local D2 shared-K/V staging. Continue with additive projection/K=V handoff and global-layer work
to close the remaining 2.67% gap to 64.82 tok/s.

Post-promotion profiling attributes approximately 2.01 ms per D2 group to the new local split/merge kernels,
down from the pre-change estimate of 6.1 ms. Fixed-T=3 FP8 projections remain about 6.9 ms/group, while the Target
output head and global attention are each about 3.6 ms/group. Follow-up 16K/64 screens used the 65.227 tok/s
three-run winner as their retained reference:

| Follow-up candidate | MTP median | Delta | Decision |
|---|---:|---:|---|
| Eight-warps-per-CTA fixed-T=3 FP8 projection | 64.712 tok/s | -0.79% | Reject |
| Fuse global K=V BF16 boundary/RMSNorm handoff | 65.167 tok/s | -0.09% | Reject as neutral |
| One query head per local shared-K/V CTA | 64.685 tok/s | -0.83% | Reject |
| Enable FP8x4 global GQA at 16K | 65.183 tok/s | -0.07% | Reject for MTP; Ordinary screen improved to 38.999 tok/s |
| Parallelize six local softmax reductions | 65.106 tok/s | -0.18% | Reject as neutral |

All five candidates retained deterministic output in their screens and were removed from production source. The
Nsight Systems report and ignored raw screens are under
`benchmarks/results/2026-08-01/699889f/blackwell16gb-windows-mtp-local-shared-kv/`.

## 2026-08-01 Windows fixed-D2 optimization screen

The Windows RTX 5080 Laptop iteration loop now uses `tools/screen_mtp.py` with the first 2,048 tokens of the
pinned Wikipedia workload, 256 fixed output tokens, checkpoint-FP8 KV, D2, one warm-up, and two measured runs.
This approximately 30-second screen is a rejection filter only. It requires deterministic output, zero runtime
fallbacks/loop allocations, and can optionally require exact Ordinary/MTP token equality. The retained parent
screen median was 48.021 tok/s with output SHA-256
`b520280a6baeb969dc7c07525a289714419587bf871ebefa744a62910178d17f`.

Nsight Systems child-node tracing at 16K attributed approximately 6.8 ms per D2 Target group to fixed-T=3 FP8
projections, 6.1 ms to local attention, 3.2 ms to global attention, and 3.6 ms to the shared three-row Target
output head. The control, speculative-KV copy, commit, and branch kernels are individually negligible. The trace
is retained as `profile-current-16k-32.nsys-rep` under the ignored Windows optimization result directory.

| Candidate | Short-screen median | Delta from short parent | Decision |
|---|---:|---:|---|
| Fuse Gate/Up product quantization | 46.867 tok/s | -2.40% | Reject |
| Assistant block-sum warp reduction | 47.734 tok/s | -0.60% | Reject |
| Fused assistant BF16 Gate/Up/product | 47.662 tok/s | -0.75% | Reject |
| Assistant output-head blocks 2,048 / 8,192 | 46.894 / 47.596 tok/s | -2.35% / -0.88% | Retain 4,096 |
| Reuse T=3 local prefill attention | 43.862 tok/s | -8.66% | Reject |
| T=3 NVFP4 Gate/Up with eight warps | 47.125 tok/s | -1.86% | Reject |
| Concurrent three-row local decode attention | 47.986 tok/s | -0.07% | Reject as neutral |
| Four-way output-head loop unroll | 53.843 tok/s initial; 49.831 confirmation | +12.12% / +3.77% | Reject after clean 16K result is neutral |

The output-head unroll demonstrates why the short screen cannot promote code. An initial 16K/1,135-token run at
54.809 tok/s was invalidated after an unrelated GPU-heavy game was disclosed. The exact rerun began at 0% GPU
utilization and 0 MiB allocated VRAM and used one alternating warm-up pair plus three measured pairs. It produced
59.109 tok/s MTP median versus the retained 59.277 tok/s parent (-0.28%) and 38.003 tok/s Ordinary versus 37.876
tok/s (+0.34%); MTP samples were 59.148/59.109/58.971 tok/s. All runs retained the exact 1,135-token hash
`43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1` and the same 1,004/632/372
proposed/accepted/rejected counts over 502 Target batches. The clean result is neutral rather than a regression,
so the kernel change and every other candidate were removed. Only the screening workflow and the corrected 64.82
tok/s qualification gate are promoted.

## 2026-07-31 current Linux llama.cpp baseline

Hypothesis: The current upstream llama.cpp commit used by the Windows handoff can be rebuilt on Linux with the
same SM120a/CUDA/Flash-Attention/Q8-KV configuration and provide a reproducible external reference.

Configuration: llama.cpp `000547513f1530346ecd163db8b3e13962949961` (version 10210), `120a-real`,
`GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_NATIVE=OFF`, full GPU residency, split mode none, Flash Attention, Q8_0 K/V,
batch/ubatch 2048/512, 8 threads, poll 100, one server slot. Linux denied priority 2 without elevated rights, so
server runs explicitly used normal priority 0. The patched same-source converter preserved 144 NVFP4 MLP tensors
and stored 184 source FP8 attention tensors as Q8_0 with `--fp8-as-q8`. Target GGUF SHA-256 is
`0fc3dce6d631d1ee5ab5398f621b4bfe50591d01d08339659d554eb91e23091d`; assistant SHA-256 is
`7b82a9f31fa365fb8ce533424cfad6c5106086f40b3eade4d91d8c5bb63d8224`.

| Existing context | Prefill median tok/s | Decode median tok/s |
|---:|---:|---:|
| 128 | 2,627.35 | 35.84 |
| 512 | 3,208.42 | 36.86 |
| 2,048 | 2,812.06 | 36.19 |
| 8,192 | 2,618.83 | 35.61 |
| 16,384 | 2,517.67 | 34.68 |
| 32,768 | 2,140.81 | 32.01 |
| 65,536 | 1,610.62 | 28.45 |

Fixed Wikipedia 16K/1,135-token, three-warm-up/ten-run characterization: ordinary decode 33.386 tok/s median
(95% CI `[33.355, 34.114]`), D2 54.703 tok/s median (95% CI `[54.694, 54.727]`). D2 proposed/accepted/rejected
was 1,035/616/419 over 519 groups in every run. Ordinary and D2 are deterministic within llama.cpp but are not
required to share output IDs; the output hashes are retained separately. The exact shared prompt hash is
`d07ad4d805944f0b87869da0c5bb44d99e8c43c0eb57d05a108ad80a6abb51a8`.

Linux overview references on the same prompt and 0.90 GPU memory policy: gem16 ordinary/D2 were 31.472/46.248
tok/s in the matching 3/10 runs; direct vLLM FP8 ordinary/D2 were 35.100/56.355 tok/s in one-warm-up/one-run
characterizations. These are format- and runtime-disclosed references, not cross-engine parity claims. Full raw
artifacts, commands, model inventories, verbose residency log, and engine/system metadata are under
`benchmarks/results/2026-07-31/6e16dd1/blackwell16gb-linux-llama-current/`.

The baseline is not accepted as a quality or native-dispatch headline yet: current output quality remains to be
requalified, and SASS proves NVFP4 instruction availability but not profiler-level invocation attribution.

## 2026-08-02 scalar-order all-head GQA staging at 16K and 32K

Hypothesis: after the 64K all-head GQA winner, the same shared K/V staging can benefit smaller context tiers if it
retains their scalar dimension assignment rather than paying FP8x4 register/conversion costs or changing their
established FP32 reduction order.

Implementation: capacities from 16,384 through 65,535 now use the same 16-head/two-heads-per-warp GQA CTA and
512-token split as the 64K path, but each lane consumes dimensions `lane + 32*i` and performs scalar E4M3
conversion exactly as the parent grouped kernel. Capacities of at least 65,536 retain the committed FP8x4
specialization. Ordinary and fixed-D2 use the same specialization; D2 launches one all-head CTA grid per verifier
row so each row remains bit-identical to Ordinary Target evaluation. Capacities below 16K retain the four-head
grouped kernel.

| Existing context | Parent median tok/s | Candidate median tok/s | Delta | Parent/Candidate 95% CI |
|---:|---:|---:|---:|---:|
| 16,384 | 31.780 | 32.861 | **+3.40%** | `[31.753,31.866]` / `[32.841,32.920]` |
| 32,768 | 29.245 | 31.461 | **+7.58%** | `[29.215,29.399]` / `[31.439,31.532]` |
| 65,536 screen | 30.370 | 30.353 | -0.06% | existing FP8x4 path; identical checksum |

The qualified 16K/32K rows use three warm-ups and ten measured 256-token runs. Median/p95 ITL improves from
31.393/32.476 to 30.361/31.405 ms at 16K and from 34.062/35.374 to 31.737/32.720 ms at 32K. All parent/candidate
checksums match and workspace remains 721,880,832/748,097,280 bytes for the two qualified tiers.

On the exact Wikipedia 16K/1,135-token workload, three warm-ups and ten measured runs raise Ordinary from the
retained 31.472 to 32.919 tok/s (+4.60%) and fixed-D2 from 46.248 to 50.806 tok/s (+9.86%). Ordinary and D2 retain
all 1,135 IDs and SHA-256 `43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`; every D2 run
retains 1,004 proposed, 632 accepted, and 372 rejected tokens over 502 groups. The candidate remains below the
same-machine llama.cpp 33.386/54.703 Ordinary/D2 references and the 64.82 tok/s fixed-D2 target, so projection,
assistant, and verifier-output work remain open.

Correctness: host/CUDA CTest passes. The dedicated 64K three-row fixture requires bitwise equality between D2 and
three independent Ordinary GQA evaluations. `cuobjdump` reports 68 registers/thread for ordinary scalar GQA and
70 for scalar D2, 42,080 bytes compiler-accounted shared memory, and zero stack/local memory for both. No
precision, KV, persistent-weight, workspace, or token-loop allocation change is introduced.

Decision: promote scalar-order all-head GQA for 16K/32K and retain FP8x4 all-head GQA at 64K. Raw evidence is under
`benchmarks/results/2026-08-02/d5dcbea/blackwell-linux-global-gqa-scalar-tier16k/`.

## 2026-08-02 all-head GQA staging for 64K global decode

Hypothesis: the eight global layers have one D512 K/V head shared by all 16 query heads, so loading every physical
K/V row independently for four query groups wastes long-context bandwidth. Assigning two query heads to each of
eight warps and staging 16 contiguous E4M3 rows once per CTA can remove this four-way cache traffic while retaining
the selected 512-token split, online-softmax, merge, and per-output accumulation orders.

A corrected llama.cpp comparison motivated this implementation. The older Nsight `llama-bench -p N -n 4` files
ran prompt processing and generation as separate tests (`n_prompt: 0` for the latter); their large
`mul_mat_q<...,128>` totals were prefill, not context-preserving M=1 decode. A new combined `-pg 16384,4` trace
shows actual decode uses `mul_mat_vec_q`: approximately 11.15 ms/token for NVFP4 and 6.62 ms/token for Q8_0,
versus gem16's 8.59/5.17 ms FP4/FP8 projection families. That evidence rejects copying llama.cpp's prefill MMQ
schedule into gem16 T=1 and instead confirms attention as the remaining cross-engine structural gap.

Implementation: for capacities of at least 65,536, one CTA now covers all 16 global query heads for each retained
512-token split. Each warp owns two heads; 16-token K/V tiles are loaded once with aligned `cp.async`, decoded as
E4M3x4, and reused by all warps. The ordinary 512-token split boundaries and FP32 LSE merge topology are unchanged.
The fixed-D2 verifier launches the same primitive independently for each of its three rows, preserving exact
ordinary Target arithmetic while reducing each row from four query-group K/V reads to one. Shorter tiers and local
attention are unchanged.

| Existing context | Parent median tok/s | Candidate median tok/s | Delta | Parent/Candidate 95% CI |
|---:|---:|---:|---:|---:|
| 16,384 | 31.767 | 31.673 | -0.30% | unchanged path; identical checksum |
| 32,768 | 29.266 | 29.278 | +0.04% | unchanged path; identical checksum |
| 65,536 | 27.261 | 30.263 | **+11.01%** | `[27.244,27.291]` / `[30.245,30.287]` |

The 64K result uses three warm-ups and ten measured 256-token runs. Median/p95/p99 inter-token latency improves
from 36.642/38.041/39.484 ms to 32.978/34.440/36.184 ms. All parent and candidate runs retain checksum
`8043681594391854731`; workspace remains 800,530,176 bytes. Child-node profiling reduces the eight global split
kernels from 8.112 to 3.749 ms/token (-53.8%). The GQA kernel uses 64 registers/thread, 41,056 bytes static shared
memory, and zero local memory/thread.

Correctness: complete host/CUDA CTest passes, including the 64K global FP8 reference fixture. A 16K Wikipedia
prompt with a 64K cache tier produces 256/256 identical Ordinary and Fixed-D2 Target tokens with SHA-256
`9b410a948744fed42084241a32f6fa1538ead22fff1d25194ae70e658f866ae8`; the D2 screen reaches 55.408 tok/s with
145 accepted and 75 rejected proposals over 110 groups. No precision, persistent weight, KV-cache, workspace, or
token-loop allocation change is introduced.

Decision: promote the all-head GQA staging path for capacities of at least 64K and retain the existing scalar/grouped
path below it. Raw evidence is under
`benchmarks/results/2026-08-02/da65217/blackwell-linux-global-gqa-tile16-chunk512/`; the corrected llama.cpp trace
is under `benchmarks/results/2026-08-02/da65217/blackwell-linux-kernel-comparison/`.

## 2026-08-01 vectorized FP8 global decode tier at 64K

Hypothesis: once the contiguous global FP8 cache reaches 64K, grouping four adjacent D512 K/V dimensions into one
aligned 32-bit load and one E4M3x4 conversion can reduce cache-load and conversion instruction pressure enough to
offset a higher register count. Shorter capacities retain the scalar kernel. The fixed-D2 verifier uses the same
selected arithmetic so speculation remains exact against ordinary Target output.

Implementation: ordinary global decode uses four-wide QK and value accumulation only when cache capacity is at
least 65,536. The three-row global verifier dispatches the corresponding four-wide kernel at the same threshold.
Local D256 attention, global split size, online-softmax and merge topology, FP8 cache bytes/scales, workspace,
checkpoint precision, and all shorter context tiers are unchanged.

| Existing context | Parent median tok/s | Candidate median tok/s | Delta | Parent/Candidate 95% CI |
|---:|---:|---:|---:|---:|
| 16,384 | 31.691 | 31.838 | +0.47% | `[31.681,31.702]` / `[31.793,31.922]` |
| 32,768 | 29.290 | 29.508 | +0.74% | `[29.257,29.417]` / `[29.306,29.530]` |
| 65,536 | 25.701 | 27.164 | **+5.69%** | `[25.447,25.727]` / `[27.149,27.229]` |

The 64K result uses three warm-ups and ten measured 256-token runs. Median/p95 inter-token latency improve from
38.955/40.425 ms to 36.726/38.065 ms. Workspace remains 800,530,176 bytes and recurring allocation remains false.
The shorter 16K/32K 3/10 checks retain the parent checksums because they dispatch the scalar tier.

Correctness: the 64K operator comparison against the score/softmax/value reference reports maximum absolute error
`3.98606e-7`, RMS `6.24295e-8`, and cosine `1`; four repeated executions are bit-deterministic. Parent and candidate
first-token full-vocabulary logits are byte-identical. Autoregressive parent/candidate output may later diverge
because QK dimension assignment changes the valid FP32 reduction order; this is recorded rather than mislabeled as
a quality failure. Candidate ordinary and fixed-D2 at 64K match all 256 generated IDs. The fixed Wikipedia 16K D2
workload retains all 1,135 IDs and SHA-256 `43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`.
Host/CUDA CTest pass, including a new 64K synthetic reference case.

Adjacent child-node profiles reduce the eight ordinary global split kernels from 9.430 to 8.112 ms (-14.0%). The
ordinary vector kernel uses 66 registers/thread versus 56, 9,248 bytes static shared memory, and zero stack/local
memory. The vector D2 kernel uses 93 registers/thread, 25,632 bytes shared, and zero stack/local memory. Grid,
workspace, persistent weights, and KV bytes are unchanged. Three exact structural probes were removed before this
winner: flattening grouped FP8 bindings regressed 16K/64K by 8.05%/5.41%, dual-N-tile FP8 O regressed by
3.83%/1.86%, and two vocabulary rows per output-head warp moved them by -1.16%/+0.13%.
Decision: promote the vector path solely for capacities
of at least 64K and retain scalar dispatch below it. Raw evidence is under
`benchmarks/results/2026-08-01/6e16dd1/blackwell-linux-decode-phase2-global-fp8x4-tier64k/`.

## 2026-08-01 global decode-attention phase-1 candidates rejected

Hypothesis: the widening long-context gap can be reduced by changing the D512 global decode primitive without
changing the checkpoint, KV format, or ordinary sampling semantics.

The first split-tier experiment used a 1,024-token global split at capacities of at least 65,536. Its Nsight
microprofile improved the split kernel from 11.00 ms to 9.16 ms and the merge from 0.234 ms to 0.091 ms at 65K,
but the qualified 3-warm-up/10-run end-to-end result regressed from 25.567 to 25.229 tok/s (-1.32%). The median
inter-token latency also regressed from 38.950 to 39.640 ms. The candidate was removed; D2 remained unchanged.

A second candidate vectorized the global K/V FP8 loads and grouped four value dimensions per thread. Short
screening was promising (33.169 versus 32.381 tok/s at 16K and 27.344 versus 25.517 tok/s at 64K, each with one
warm-up and three measured runs), but a 256-token deterministic comparison matched only 20/256 tokens. Keeping
scalar Q/K accumulation and vectorizing only values improved that to 67/256, but still changed the sequence. A
separate vectorized global merge showed only +1.0% at 16K and +0.4% at 64K in the short screen and likewise failed
the 256-token exact comparison. A warp-shuffle variant reduced each global K row to four 32-bit loads per lane
while reconstructing the original scalar dimension order; it still matched only 67/256 tokens and was reverted.
A GQA shared-K prototype then fused the four query groups into one CTA and
staged each 512-byte key row in shared memory. It preserved the scalar arithmetic but serialized the query-group
work: 27.905 versus 32.100 tok/s at 16K and 20.627 versus 25.525 tok/s at 64K. A 768-token split tier was also
screened; it was slightly faster at 16K but slower at 64K (25.400 versus 25.746 tok/s). It was removed immediately.
A read-only `__ldg` load experiment preserved scalar arithmetic but was not a winner either: it moved the short
screen from 32.150 to 32.443 tok/s at 16K and from 25.412 to 25.266 tok/s at 64K. These variants were all
reverted rather than trading numerical reproducibility for a screening win.

At the end of phase 1 the production source and `blackwell-release` binary returned to the scalar 512-token global
path. The separately qualified phase-2 entry above later promoted vectorized physical FP8 handling only for the
64K-and-above capacity tier; all other candidates in this section remain removed. Raw rejected-candidate runs and
direct output comparisons remain under `benchmarks/results/2026-08-01/6e16dd1/`.

## 2026-07-30 repository-local six-item media conversation through 128K

Hypothesis: Three generated images and three independently sourced
public-domain speech excerpts can replace machine-local benchmark dependencies,
increase media diversity, and remain retrievable in one sampled-D2 session
through 128K.

Implementation: The default long-server root alternates image/audio items from
`benchmarks/media/suite.json`, verifies every SHA-256 before server startup, and
asks for image codes/counts plus distinctive speech. The root uses no reasoning
and a 384-token visible budget. Six final per-asset turns and ordinary checkpoint
probes retain sampled fixed-D2 and bounded reasoning. The remaining one-slot,
FP8-KV, Gemma-sampling, telemetry, warm-up, and three-run boundaries are unchanged.

Correctness: The 1,878-token root returns `ORBIT 47`, `CEDAR 82`, `3` purple
planters, `HARBOR 19`, `4` red sailboats, and the locked Alice/Moby Dick/Pride
phrases (`sister`/`bank`, `lexicons`/`flags`, `wife`/`neighborhood`). Six
separate final turns beginning at 131,329 input positions retrieve every locked
term again; all manifest checks pass.

| Target | Actual input | New tokens | Engine prefill | New-token prefill | First delta | Decode | MTP acceptance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 3,755 | 69 | 63.27 ms | 1,122.14 tok/s | 142.03 ms | 60.28 tok/s | 82.1% |
| 8K | 7,972 | 71 | 70.46 ms | 1,007.64 tok/s | 147.59 ms | 61.99 tok/s | 86.5% |
| 32K | 32,549 | 71 | 118.55 ms | 587.40 tok/s | 205.67 ms | 53.13 tok/s | 95.7% |
| 64K | 65,440 | 72 | 180.42 ms | 410.15 tok/s | 287.55 ms | 39.41 tok/s | 90.3% |
| 128K | 130,963 | 72 | 296.18 ms | 242.22 tok/s | 461.14 ms | 26.89 tok/s | 87.5% |

The 740 telemetry samples observe at most 13,284 MiB, 81.94 W, 2,640 MHz,
and 69 C. Raw evidence is retained under
`benchmarks/results/2026-07-30/f21eeec-worktree/blackwell16gb-linux/server-long-multimedia-v1.json`.
The directory labels the uncommitted media-suite worktree explicitly; this is a
full 1/3 characterization, not the publication-grade 3/10 repetition contract.

## 2026-07-30 sampled-D2 multimodal server conversation through 128K

Hypothesis: A single 262,144-position server slot should preserve an initial
image/audio turn while incremental text grows through 128K, keep the exact
resident cache, execute checkpoint-recommended sampling through fixed D2, and
make prompt/decode/stream latency degradation observable at realistic depths.

Implementation: The managed harness starts a fresh one-slot server with FP8 KV,
Google's pinned assistant, fixed D2, temperature 1.0, top-k 64, and top-p 0.95.
It sends the real `natural_scene.png` and 17.26-second `freeman.wav`, extends one
linear Responses chain with prose, measures one warm-up plus three streamed
turns near each context tier, and performs a final media retrieval. Per-request
Prometheus deltas separate engine prompt and decode time; SSE timestamps expose
first delta and MTP burst intervals. Seventeen filler requests retain their own
large-suffix prefill evidence.

Correctness: The 644-token multimodal root identifies sign `24`, creation
stories, cultures, and the afterlife. At 131,335 input tokens the final turn
still states `24` and explicitly confirms the afterlife. Every measured tier
executes only D2 groups, with zero D1/D4 groups and zero ordinary fallback. The
workload exposed and fixed delayed/multiple reasoning-channel accounting that
previously poisoned the chain at 32K.

Measured medians:

| Target | Actual input | New tokens | Engine prefill | New-token prefill | First delta | Decode | MTP acceptance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 1,788 | 69 | 58.00 ms | 1,200.98 tok/s | 129.57 ms | 54.58 tok/s | 63.7% |
| 8K | 7,985 | 71 | 76.03 ms | 949.49 tok/s | 148.17 ms | 52.31 tok/s | 66.7% |
| 32K | 32,508 | 70 | 121.77 ms | 579.43 tok/s | 204.67 ms | 44.58 tok/s | 71.0% |
| 64K | 65,362 | 72 | 180.07 ms | 399.85 tok/s | 289.92 ms | 38.97 tok/s | 89.9% |
| 128K | 130,969 | 72 | 303.48 ms | 237.24 tok/s | 465.77 ms | 27.72 tok/s | 95.0% |

Accepted D2 groups publish token bursts, so pooled median SSE delta intervals
are near zero; p95 grows from 44.90 ms at 2K to 113.27 ms at 128K and is the
more useful interactive tail measure. Large filler chunks separately fall from
3,832 tok/s for 2,401 new tokens around 4K total context to 733 tok/s for 34,216
new tokens ending around 128K. These are exact incremental prefill boundaries,
not the small-suffix row above.

Resource evidence: 645 continuous samples report 13,284 MiB maximum GPU memory,
81.52 W maximum power, 2,467 MHz maximum SM clock, and 67 C maximum temperature.
Raw evidence is retained at
`benchmarks/results/2026-07-30/c3b4907-worktree/blackwell16gb-linux/server-long-conversation.json`.
This is one evolving conversation rather than ten independently reconstructed
128K prompts; tier distributions intentionally span nearby actual contexts.

## 2026-07-30 HTTP server benchmark foundation

Hypothesis: Separate complete HTTP root, resident continuation, live SSE, and
multi-slot contention boundaries so protocol-facing work is measurable without
calling it core GPU throughput.

Implementation: `tools/benchmark_server.py` runs each path with independent
warm-ups and retained raw measurements, reports Student-t 95% mean intervals,
p95/p99, exact cache usage, first streamed delta, per-lane concurrency data,
Prometheus counter deltas, and continuous NVML-backed `nvidia-smi` telemetry. It
refuses to overwrite an existing artifact.

Characterization: On the Linux 16 GB Blackwell machine, a greedy 2K-context,
64-output, two-slot 3-warm-up/10-run screen measures median HTTP wall times of
1,723.93 ms for new roots, 770.27 ms for resident continuations, 1,750.73 ms for
streamed roots, and 1,573.94 ms for two concurrent resident continuations. The
streaming median first delta is 61.82 ms. Median resident cached-token usage is
21 tokens; the two-lane sum is 52. The complete harness generated 2,821 output
tokens over 104 successful requests with no failures or disconnects.

Resource evidence: 348 continuous samples report 10,808 MiB maximum GPU memory,
82.03 W maximum board power, 2,190 MHz maximum SM clock, and 67 C maximum
temperature. Clocks range from 345 to 2,190 MHz because they were not locked.
These values are server-boundary characterization only, not a kernel promotion
or cross-engine speed claim. Raw JSON is retained under
`benchmarks/results/2026-07-30/8ccb5a0-worktree/blackwell16gb-linux/server.json`.

## 2026-07-29 device-routed bounded reasoning in fixed D2

Hypothesis: Carry response-channel and reasoning-budget state in the existing fixed-D2 device control and route
only ambiguous boundary rows through ordinary Target decode. This should retain exact bounded-thinking semantics
while removing the prior ordinary-only reasoning phase and its host-mediated transition back to MTP.

Implementation: A single outer CUDA conditional loop selects a complete D2 group or one ordinary Target row.
Ordinary rows publish normalized hidden state into the MTP workspace, and a device continuation kernel selects the
next route. D2 is allowed with at least three safe reasoning/output slots; partial markers, the last one or two
reasoning slots, exact forced close, and short tails use the ordinary branch. The channel tracker consumes every
committed token sequentially, including natural close and following answer tokens within one accepted group.

Correctness/profile evidence: Host and CUDA CTest suites pass. The CUDA transition fixtures pass Compute Sanitizer
memcheck with zero errors. The sampled D1/D2/D4 validator preserves ordinary same-seed output for seeds 0, 1, and
42. Resident two-turn validation preserves complete ordinary/MTP responses for forced 8/8 closures and for a
longer forced 128/128 plus natural 92/128 pair. Nsight Systems records exactly one `cudaGraphLaunch` and one
`gem16.mtp.fixed_d2_chain` range for the profiled generation request, confirming that group and channel transitions
do not require a blocking host control roundtrip. Short local throughput observations are functional
characterization only; no new performance number is promoted without the required repeated benchmark and resource
telemetry.

## 2026-07-29 sampled MTP resident-chat and Linux qualification

Hypothesis: Reuse exact batched Target verification for sampling by assigning each verifier row the ordinary
Target RNG step and repetition history, then carry that state through the existing fixed-D2 conditional graph and
mapped-pinned streaming ring. For Google's top-k-64 profile, avoid probability preparation and scans beyond the
64 sorted candidates.

Implementation: Fixed-shape MTP now materializes softcapped Target logits for every row, constructs independent
repetition masks from the committed history and proposal prefix, applies the existing seeded sampler, accepts only
Target-sample-equal proposals, and commits the emitted row's repetition mask. `MtpDeviceControl` carries the
sampling step. Fixed D2 captures sampled selection and a sampled ordinary tail; D1/D4 remain direct. Chat parses
and defaults to the pinned generation profile (`temperature=1.0`, `top_k=64`, `top_p=0.95`) while `--greedy`
remains explicit. Sampling probability preparation/scan uses only the top-k prefix; radix sorting is unchanged.

Correctness: Ordinary and MTP outputs match for D1/D2/D4 over multiple seeds, repetition penalty 1.1, checkpoint
FP8 and BF16 K/V, and a local-ring-wrap fixture. A real resident two-turn chat produces `Blau` then `Blau` in both
modes and reports GPU chaining on both MTP turns. The 3-warm-up/10-measured Linux Wikipedia run preserves one
1,114-token stop-terminated output in all 26 ordinary/MTP executions, SHA-256
`3bf1d6f1750345a3d9732950885275a660fcf0ace0f6ded051aeead6a916bd3a`. Every MTP run proposes 1,002 drafts,
accepts 612, rejects 390, and executes 501 groups. CTest and Python tests pass.

Performance: Sampled ordinary reaches 31.450 median tok/s with 95% mean CI `[31.427,31.546]`; sampled fixed D2
reaches 46.234 (`[46.093,46.268]`), a 1.470x improvement (+47.0%). Workspace is 719,728,128 bytes. An adjacent
Linux greedy 3/10 regression preserves the 1,135-token golden hash and reaches 47.117 versus 31.634 ordinary. The
sampled path is retained as a correct material end-to-end win, but neither Linux result meets the existing 50 tok/s
performance target and neither run captured continuous power/clock/thermal telemetry. Further performance claims
require profiling rather than weakening sampled identity. Local raw evidence is under
`benchmarks/results/2026-07-29/c482926-worktree/blackwell16gb-linux-sampled-mtp/`.

## 2026-07-28 qualified GPU-chained fixed-D2 path

The final Wikipedia 16K qualification alternates ordinary and fixed-D2 order within three warm-up pairs and ten
measured pairs. All 26 executions emit the same 1,135 token IDs with SHA-256
`43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`. Ordinary measures 36.788 tok/s median
(36.776 mean, standard deviation 0.085, 95% mean CI `[36.715,36.837]`). GPU-chained D2 measures 54.903 tok/s median
(54.845 mean, standard deviation 0.402, CI `[54.557,55.132]`), a 1.492x throughput speedup (+49.2%).

Every measured D2 run reports 1,004 proposed, 632 accepted, and 372 rejected drafts over 502 verifier groups, mean
accepted length 1.259, zero ordinary fallback, and 706,913,280 reusable workspace bytes. The 50 tok/s minimum is
qualified; the 55 tok/s stretch threshold is missed by 0.097 tok/s. No continuous power, clock, or thermal
telemetry was captured in this run. Raw data and exact pair ordering are retained at the ignored path
`benchmarks/results/2026-07-28/b07b178/blackwell16gb-windows-mtp-streaming/qualification.json`.

## 2026-07-28 GPU tail and mapped-pinned asynchronous streaming

The fixed-D2 root graph now contains a dependent ordinary-tail conditional loop. When the D2 loop stops with one
or two output slots, the tail initializes ordinary decode directly from `MtpDeviceControl`, runs the exact target
forward, updates cache/position/stop state, and repeats until the budget is empty or a stop token is emitted. No
host scheduler decision occurs between the first D2 group and the final token.

Verified D2 and tail outputs are simultaneously published through a fixed 256-token SPSC ring allocated with
`cudaHostAllocMapped`. The single GPU producer waits only when `producer - consumer` reaches capacity, writes token
payloads, executes a system fence, and advances the producer with a system-scope atomic. The host uses C++20
`atomic_ref` acquire/release operations, invokes callbacks while `cudaStreamQuery` reports active compute, and
advances the consumer without synchronizing the compute stream. Callback failure sets a mapped cancellation flag;
the chain exits at the next group boundary. A CUDA fixture verifies ordered mapped publication and graph reset, then
holds the consumer until all 256 slots fill, observes one backpressure event, releases it, and completes 258 outputs.

The exact Wikipedia 16K one-warm-up/three-run screen reaches 55.009 tok/s median (54.937 mean), versus 55.063 tok/s
before streaming. All three outputs retain the 1,135-ID SHA-256
`43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`, with 632 accepted and 372 rejected drafts
over 502 groups and the same stop token. The ring adds 1,088 mapped pinned bytes; reported reusable device workspace
remains 706,913,280 bytes on the 24,576-position benchmark plan. The 32K graph-associated allocation rises from
20,971,520 to 23,068,672 bytes for the captured ordinary-tail body.

The exact 16K/256 Nsight trace contains one `cudaGraphLaunch`, five whole-process stream synchronizations, and one
4.773-second chained range. None of those synchronizations occurs in the callback polling boundary. Initialization
captures the additional tail body, so whole-process launch API calls rise to 17,171; recurring decode remains one
conditional graph launch. This phase is promoted because it preserves exactness, stays above 55 tok/s in the
required milestone screen, and removes the final blocking output-callback boundary.

## 2026-07-28 GPU-chained fixed-D2 conditional graph

The pinned CUDA 13.3 runtime supports conditional `while` graph nodes on the Windows SM120 target. The complete
fixed-D2 group is now the body of one such node. Its final device kernel writes only target-verified tokens and
proposal IDs to fixed arena buffers, accumulates exact MTP counters, advances `MtpDeviceControl.current`, and sets
the next loop condition. The host supplies one initial control record and receives outputs only after the chain
finishes; no D2H/H2D dependency exists between groups. Capacity below three exits to the existing exact D1 or
ordinary tail, while a stop token exits immediately.

The required Wikipedia 16K screen with one warm-up and three measured runs reaches 55.063 tok/s median (55.081
mean), versus 54.783 tok/s for one-host-replay-per-group. All runs retain the exact 1,135-ID SHA-256
`43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`, 632 accepted and 372 rejected drafts over
502 groups, deterministic stop semantics, and zero token-loop allocations. At this workload's 24,576-position
budget, output/proposal/aggregate device storage raises reported workspace from 706,618,368 to 706,913,280 bytes.
The matching pinned host payload is allocated during initialization. At the 262,144-position maximum, the device
and pinned-host chain payloads are each approximately 3.0 MiB.

The exact 16K/256-output Nsight trace contains one `cudaGraphLaunch`, six whole-process `cudaStreamSynchronize`
calls, and one 4.684-second `gem16.mtp.fixed_d2_chain` range covering all 111 groups. Host-replayed group capture had
111 graph launches and 116 synchronizations. Whole-process `cudaLaunchKernel` API calls rise from 14,770 to 16,205
because initialization captures one additional complete body graph; recurring decode still has no host kernel
dispatch. Graph-associated device bytes rise from 14,680,064 to 20,971,520. A focused CUDA fixture executes two
conditional iterations and verifies aggregate counters, output placement, proposal order, and termination.

The measured throughput increment is intentionally described as modest: complete group capture already removed
most launch overhead, so the remaining decode is GPU-kernel dominated. The architectural benefit is removal of the
per-group host dependency required for subsequent nonblocking streaming. Final tail handling and callback streaming
remain the next phase.

## 2026-07-28 complete fixed-D2 MTP group graph

The fixed-D2 checkpoint-FP8 path now captures one complete speculative group: both assistant proposal steps,
controlled verification-input construction, embedding, all 48 target layers, final normalization and output
selection, exact GPU acceptance, transactional KV/hidden commit, device-control transition, and compact result
copy. Dynamic row positions and circular-cache slots are read from arena-backed control records, so the same graph
replays across the full context. Capture, instantiation, and its 14,680,064 device bytes are prepared before the
token loop. BF16 KV, contexts at or below the 1,024-token local window, adaptive MTP, and D1/D4 tails retain their
existing exact direct paths.

On the Windows RTX 5080 Laptop with CUDA 13.3, the exact Wikipedia 16K workload with one warm-up and three measured
runs reaches 54.783 tok/s median, compared with 45.217 tok/s at the preceding device-control milestone (+21.2%).
Every run emits the same 1,135 IDs and SHA-256
`43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`, with 632 accepted and 372 rejected drafts
over 502 D2 groups. The reported workspace remains 706,618,368 bytes and token-loop allocations remain false. A
1,022-token prompt plus 16 generated tokens crosses the local-cache ring boundary and remains ordinary-identical
through 13 graph replays and one direct D1 tail.

The exact 16K/256-output Nsight trace contains 111 complete graph replays. Whole-process `cudaLaunchKernel` API
calls fall from 185,830 in the direct device-control trace to 14,770, while `cudaGraphLaunch` appears exactly 111
times. Stream synchronizations are 116 versus 117 previously, and the `gem16.mtp.fixed_d2_graph` NVTX range averages
42.768 ms/group. These counts include model initialization, prefill, and graph preparation; they demonstrate removal
of recurring per-kernel host dispatch without claiming that unchanged GPU kernel work disappeared. The remaining
production boundary is one host synchronization per group, targeted by the next GPU-chaining phase.

## 2026-07-28 Windows MTP device-control baseline

Commit `b5ca0ef` was profiled on the Windows RTX 5080 Laptop GPU with CUDA 13.3 and Nsight Systems 2026.1.3. The
direct `gem16-run` command used the exact 16,384-token Wikipedia prompt (little-endian token-ID SHA-256
`d07ad4d805944f0b87869da0c5bb44d99e8c43c0eb57d05a108ad80a6abb51a8`), checkpoint-FP8 KV, the pinned official
assistant, fixed D2, the checkpoint stop/suppression controls, and a 256-token output limit. The trace contains 111
complete D2 groups. CPU context-switch tracing was unavailable without administrator privileges; CUDA API, GPU,
memory, and NVTX traces are present. The raw report remains ignored under
`benchmarks/results/2026-07-28/b5ca0ef/blackwell16gb-windows-mtp-device-control/nsys/`.

The proposal CPU range averages 1.051 ms/group and the verify/accept/commit range averages 45.964 ms/group, for an
approximately 47.015 ms sequential host-controlled group. The verify range projects to a 41.407 ms GPU critical
span. A representative middle group contains 1,407 `cudaLaunchKernel` API calls; its host API time is 16.467 ms in
launch calls and 26.208 ms in the final `cudaStreamSynchronize`. The synchronization time includes outstanding GPU
work and therefore is not removable overhead by itself. The roughly 5--6 ms difference between the sequential CPU
ranges and projected GPU span is the bounded scheduling/control opportunity; it is large enough to justify the
incremental complete-group graph and GPU-chaining roadmap, but it is not a predicted speedup.

For the representative group, fixed-T3 FP8 projection kernels consume 16.058 ms of summed GPU kernel time, global
verifier attention 7.504 ms, local verifier attention 6.183 ms, the three-row target output head 2.951 ms, and the
assistant output head plus BF16 GEMV 2.782 ms. These measurements reinforce the current decision: do not reopen the
rejected small kernel-geometry probes; first remove the per-group host scheduling boundary without changing target
arithmetic, acceptance, or commit semantics. The next code milestone is an arena-backed `MtpDeviceControl` with
host/device transition parity while the existing host loop remains authoritative.

## 2026-07-29 Reopened exact D2 structural sprint

The retained `fffefcb` result is 46.422 tok/s, approximately 48.66 ms per D2 group, and needs about 3.5 ms/group
to reach the 50.0 tok/s gate. The clean-head profile identifies two structural targets large enough to matter:
the three-row target output head at approximately 4.8 ms/group and local verifier attention at approximately
6.9 ms/group. Small CTA and launch-count probes are not reopened.

The first hypothesis is that the BF16 tied head is instruction-limited as well as bandwidth-limited: its current
warp-row kernel loads each weight once but executes three scalar FP32 FMA chains and warp reductions. A fixed T3
SM120 BF16 Tensor-Core kernel can retain one weight read, pad only the inactive M dimension in transient workspace,
apply the exact softcap and suppression rules, and reduce candidates on GPU. Its expected limiting resources are
the 2.0 GB tied-weight traversal and Tensor-Core tile scheduling. Its MMA reduction order differs from the ordinary
warp-row head, so it is a candidate only: the existing kernel remains the reference, and any changed Wikipedia ID,
hash, or 632/502 acceptance result rejects the implementation regardless of speed. If it cannot save materially,
the next candidate is local T3 attention with direct tentative K/V reads and enough split parallelism to avoid the
under-occupancy of the previously rejected batch prototype.

The fixed T3 WMMA prototype padded the three BF16 rows to M16, traversed each tied-weight byte once, retained the
softcap and suppression rules, and preserved all 1,135 IDs, the fixed hash, and 632/372 acceptance. It reached only
42.460 tok/s and 53.20 ms/group in the zero-warm-up/one-run screen. After removing it and rebuilding, the immediately
adjacent scalar-head run under the same zero-warm-up/one-run command reached 46.662 tok/s and 48.41 ms/group,
consistent with the retained 46.422 characterization. The 9.0% adjacent gap is therefore not a missing-warm-up
artifact. Padding three rows to 16 makes Tensor-Core work and tile-management cost exceed the scalar warp-row head
despite unchanged weight traffic. The complete kernel and API changes were removed; only the ignored raw candidate result is retained.
Decision: reject target-head Tensor Cores at T3 and proceed to local attention rather than tuning this geometry.

The direct-tentative-K/V local verifier was exact in both concurrent and sequential forms but reached only
46.240 and 45.600 tok/s after address-strength reduction; the initial modulo-heavy sequential form reached
41.095 tok/s. It removed cache backup/append/restore work, but simultaneous rows contend for bandwidth while
sequential direct addressing does not recover the removed launch cost. Direct quantization into retained MTP K/V
was also neutral at 46.228 tok/s. Both families were removed. Eight-head global grouping, whole-activation NVFP4
staging, output-head hidden staging, and eight-warp or `cp.async` FP8 staging all regressed and were removed.

A fresh 256-token profile showed the fixed T3 FP8 projections at 8.06 ms/group. The retained kernel now stages the
three source E4M3 activation rows once per four-warp CTA in 11.25–24 KiB of dynamic shared memory, then preserves
each independent K32 MMA chain and output scaling. Profiled fixed-T3 projection time falls from 8.06 to 6.44
ms/group. In the controlled Wikipedia comparison, both baseline and candidate used three full warm-ups and five
measured runs. Baseline median was 45.805 tok/s (95% CI 45.753–45.843); staged FP8 median was 47.432 tok/s (95% CI
47.349–47.476), a 3.55% throughput gain. All five candidate runs emitted the fixed 1,135-ID hash, accepted/rejected
632/372 drafts over 502 groups, and reported no fallback or token-loop allocation. The earlier 48.354 tok/s value
was a valid zero-warm-up single-run screen, not the repeated result. Workspace remains 706,618,112 bytes. Decision:
retain scalar shared-activation staging for both T3 grouped Q/K/V and O; the 50 tok/s gate remains unmet and this
3/5 characterization is not the required 3/10 qualification.

## 2026-07-29 Exact D2 verifier specialization sprint

A clean-head Direct-O profile at `50ba3a5` measures 52.107 ms per verifier group and 43.745 profiled tok/s over
111 D2 groups. Global/local split-online attention consume 10.95/6.93 ms per group, followed by NVFP4 Gate/Up,
Down, the target output head, and FP8 Q/K/V. The retained bounded candidates specialize only the fixed three-row
D2 verifier: global attention shares historical K/V loads while preserving independent row reductions, FP8 Q/K/V
and O share each weight load across three independent K32 accumulators, the output head compiles exactly three
softcapped rows, and NVFP4 Down groups eight independent output warps per CTA. Their resource usage is respectively
72/48/40/64 registers per thread, with zero stack or local memory. The global path adds 4,727,808 bytes at the 24K
profile through an arena-planned row-strided partial workspace; persistent bytes are unchanged.

The retained combination emits all 1,135 ordinary IDs with SHA-256
`43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`, preserves 632 accepted and 372 rejected
drafts over 502 groups, reports zero fallback and token-loop allocation, and reaches 46.422 tok/s in the full
one-run candidate screen. This is +4.4% over the retained 44.347 tok/s Direct-O candidate, but remains below the
50.0 tok/s qualification gate, so no 3/10 qualification or 55 tok/s work is authorized. CTest passes. Rejected
exact candidates are removed: batched local attention is neutral to slower, two-head global grouping is noisy and
slower in median, eight-warp Gate/Up and 16-warp target output CTAs regress, 2,048 assistant output blocks regress,
and skipping softcap shows no measurable potential and is forbidden for production regardless. Decision: retain
the cumulative exact D2 improvement, mark the 50 tok/s objective unmet, and close the bounded verifier sprint.

## 2026-07-29 Exact short-batch O-projection candidate

Before edit, a fresh Nsight Systems run on the 16K/256-output exact D2 path measures 53.081 ms per verifier group.
Within `gem16.mtp.verify_accept_commit`, the FP8 CUTLASS O projection consumes 530.958 ms over 5,328 layer calls,
or 4.783 ms/group and 9.16% of scoped kernel time. It launches an M128/N128 plan for only three verifier rows.
Hypothesis: dispatching MTP O through the existing decode-order direct FP8 MMA over each of the three rows will
preserve ordinary decode's K32 accumulation exactly while avoiding short-batch CUTLASS tile/setup overhead. The
expected limiting resource is weight traffic plus launch latency; Q/K/V already uses the same direct arithmetic.
The numerical contract is unchanged per row, the prefill CUTLASS route remains the fallback/reference path, and the
candidate is rejected unless all 1,135 Wikipedia IDs remain exact and D2 throughput improves materially. The first
full 16K candidate run emits all 1,135 retained IDs (SHA-256
`43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`), retains 632 accepted drafts over 502 D2
groups, and reaches 44.347 tok/s. CUDA/unit tests pass. This is a one-run candidate, not a 3/10 qualification;
keep the direct O path for the subsequent structural split and requalify only after the bounded sprint ends.

## 2026-07-29 Revised exact MTP target

The competitive gate is now 50.0 effective target-verified tok/s minimum and 55.0 tok/s stretch on the fixed 16K
Wikipedia workload. Current exact gem16 D2 is 42.639 tok/s, mean accepted drafts are 1.259, and verifier groups take
52.98 ms. Holding acceptance constant, 50 requires at most 45.18 ms/group (-14.7%) and 55 requires at most 41.07
ms/group (-22.5%). The minimum exceeds current llama.cpp's controlled fixed-1,135-token D2 characterization of
48.38 tok/s; the stretch target also clears its 50.21 tok/s stop-terminated result despite that run's different
output semantics. No benchmark or kernel changed in this entry. Decision: retain one bounded exact-verifier sprint,
qualify the first result at or above 50 with the full exact 3/10 gate, and continue toward 55 only for material,
profile-supported exact candidates.

## 2026-07-29 External Gemma 4 MTP feasibility matrix

The same 16,384-token Wikipedia prompt was run through direct-checkpoint vLLM 0.25.1 and llama.cpp's official
Gemma 4 assistant path. A one-line semantic vLLM patch replaces suppression-token list indexing, which performs a
forbidden CPU index transfer during CUDA capture, with two scalar indices. Graph D1/D2/D4 screens reach
49.59/58.69/56.06 tok/s. D2 then reaches 57.390 median tok/s over 3 warm-ups and 10 measured runs (mean 57.419,
95% CI `[57.370,57.468]`), with 556 accepted drafts, 513 groups, and sampled peak memory 14,166 MiB. A separate
fixed-1,135-token screen reaches 57.363 tok/s and 35.75 ms/group.

Current upstream llama.cpp `da5b4486` directly executes a BF16 GGUF converted from Google's assistant, offloads all
49 target and 5 assistant layer groups, and logs three Layer-46 plus one Layer-47 shared-KV binding. On the fixed
1,135-token screen it reaches 28.56 ordinary and 48.38 D2 tok/s; stop-terminated D2/D4 screens reach 50.21/49.75
tok/s. The target is the patched closest-parity GGUF with BF16-mapped attention and Q8_0 KV.

Neither external MTP route is exact against its own ordinary greedy route: fixed-length vLLM first differs at index
2 and llama.cpp at index 133. Their results are not promoted as baselines. They answer the hardware question:
vLLM performs a D2 verifier group in 35.75 ms, below the 37.65 ms required for 60 tok/s at gem16's measured
acceptance, while exact gem16 currently takes about 52.98 ms. Decision: retain external results only as a bound and
allow one final bounded exact-verifier sprint. Raw data remains under the ignored result tree; committed summaries
are `benchmarks/baselines/{vllm,llama_cpp}/mtp-characterization.json`.

## 2026-07-28 GPU MTP transaction and exact short-batch verifier kernels

A 16K/256 D2 Nsight trace after long-context correctness restoration attributes verifier GPU time primarily to
three direct projection families and split-online attention. GPU acceptance/commit first removes the assistant
D2H synchronization and draft H2D copy: drafts remain device-resident, target predictions are compared on GPU,
stop IDs are applied there, and fixed kernels commit tentative K/V plus one hidden row before a single compact
pinned result. The complete 1,135-token run remains exact and moves from 35.184 to 35.340 tok/s; its principal value
is one synchronization per group and a graphable transaction boundary. Workspace rises by 15,360 bytes at context
128. FP8 ring-wrap memcheck reports zero errors.

The profile then proves the existing FP8 batch kernel wastes a 128-token staging tile on D1/D2/D4. Dispatching T≤5
through the same decode-order direct MMA over 2/3/5 rows preserves all IDs and raises D2 to 39.150 tok/s. The NVFP4
Down batch has the same mismatch; one unstaged 16-token tile with four warps preserves the K64 MMA order and raises
D2 to 43.200 tok/s. Resource capture reports 40 registers/thread for FP8 Q/K/V and 64 for NVFP4 Down, zero local
bytes/thread, 128-thread CTAs, and expected `QMMA.16832.F32.E4M3.E4M3` plus
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` SASS.

Rejected exact candidates are retained only as evidence: 48 verifier-suffix graphs measure 35.291 versus 35.340
tok/s and add 6–8 MiB; T1 Gate/Up measures 44.381 versus 44.498 tok/s; eight global query heads per K/V load move
44.498 to only 44.661 tok/s; sixteen heads regress to 43.512 tok/s; an eight-warp unstaged Down measures 44.449
versus 44.498 tok/s. All were removed. Explicit adaptive mode selects D4/D2/D1 from context and 16-group acceptance
windows, then uses bounded ordinary fallback below the profiled break-even threshold. Explicit D1/D2/D4 behavior
is unchanged.

Qualification uses three alternating warm-up pairs and ten alternating measured ordinary/D2 pairs on the exact
16,384-token Wikipedia prompt. Ordinary median is 31.798 tok/s (mean 31.794, 95% CI `[31.783,31.806]`); D2 median
is 42.639 tok/s (mean 42.641, 95% CI `[42.623,42.658]`), a 1.341x throughput speedup (+34.1%). Every measured run
emits the same 1,135 IDs, mean accepted length is 1.259 in all D2 runs, fallback/allocation counters remain zero,
and sampled peak GPU memory is 10,838 MiB. CTest, Transformers D2 drafts, D1/D2/D4 short identity, BF16, local-ring
wrap, stop handling, adaptive identity, memcheck, and prefill allocation-boundary checks pass. Raw qualification and
telemetry are retained under the ignored `18ff81e-worktree/blackwell16gb-mtp-performance/qualification/` path.

## 2026-07-28 restore exact Wikipedia 16K MTP and accelerate assistant attention

Using D2 with no warm-up and one run per correctness candidate isolated the generated-index-68 divergence to FP8
CUTLASS target Q/K/V. Restoring the decode-order direct grouped Q/K/V batch while retaining exact CUTLASS O makes
the complete 1,135-token MTP output equal ordinary. The direct correction measures 30.031 tok/s; retaining CUTLASS
O measures 30.692 tok/s. Reusing the qualified split-online FP8 decode-attention kernel for the assistant's
long-context shared K/V raises D2 to 35.184 tok/s without changing acceptance (632 accepted of 1,004 proposed;
mean accepted length 1.259) or target output. Compared only as characterization, this is +10.7% over the retained
31.775 ordinary median; it is not a qualification because the ordinary result used three repetitions.

A new target-global multi-row kernel shared K/V loads across D2's three verification rows and preserved all 1,135
IDs, but reached only 34.767 tok/s. Its larger shared/register footprint lost to the independent-row route, so the
candidate was removed. CTest, the two-draft Transformers fixture, FP8 local-ring wrap, and active FP8 memcheck above
1,024 positions pass. Runtime JSON exposes `fp8_online_split_long_reference_short` for assistant attention.

## 2026-07-28 Wikipedia 16K MTP characterization

The exact 16,384-token Wikipedia summarization workload was run at commit `2dba16d` with one warm-up and three
measured repetitions per mode. Ordinary/D1/D2/D4 median decode is 31.775/29.634/31.702/28.866 tok/s. Relative to
ordinary, D1 is -6.74%, D2 -0.23%, and D4 -9.15%; mean accepted lengths are 0.740/1.240/1.755. Thus the current MTP
path has no long-context throughput win even before qualification.

All modes are internally deterministic, but all MTP variants share an output that first differs from ordinary at
generated index 68 and stops at 979 rather than 1,135 tokens. This violates exact ordinary/MTP equivalence and
also gives later measurements different decode trajectories. No speedup is claimed. The result selects
long-context exact target verification and assistant global-attention cost as blockers before more 16K performance
work. Raw results remain under the ignored
`benchmarks/results/2026-07-28/2dba16d/blackwell16gb-wikipedia16k-mtp/`.

## 2026-07-27 batched exact MTP target verification

The first speedup-path change replaces serial target verification with a fixed-shape causal batch over the input
plus up to four assistant drafts. Per-layer speculative K/V rows are retained in a fixed arena; local-ring slots
are restored after each speculative attention pass and only a host-confirmed prefix is committed. This is a
correctness checkpoint, not a promoted performance path. FP8/BF16 short sequences and a 1,026-token local-ring
wrap retain ordinary greedy IDs, the independent four-draft Transformers fixture passes, CTest passes, and active
BF16 memcheck reports zero errors.

The initial natural-chat characterization had mean accepted length 1.89 but reached only 35.44 tok/s versus 36.20
ordinary. Profile attribution selected target MLP and FP8 projections. The promoted exact verifier keeps each
recursive assistant token on device until the group finishes, uses the byte-qualified fused native Gate/Up/GELU
batch operator, and uses the existing FP8 CUTLASS batch projections. Under 3 warm-ups and 10 alternating
context-512 runs on the same 53-token/256-output natural prompt, median MTP is 42.897 tok/s (mean 42.842,
standard deviation 0.213) versus ordinary 35.270 tok/s (mean 35.207, standard deviation 0.147): +21.6% effective
throughput. All MTP outputs equal ordinary greedy output; mean accepted length is 1.886 with 89 target batches.

The native-NVFP4 fused Gate/Up profile reduces the former separate Gate/Up family, but direct batch attention,
FP8 CUTLASS projections, batched output head, and assistant BF16 GEMVs remain significant. A causal-prefill
attention candidate reaches 55.06 tok/s but diverges at output step 15; NVFP4 CUTLASS verifier projections also
diverge on the natural sequence. Both candidates were removed. The remaining path is a workload-specific MTP win,
not a 60 tok/s result. The raw artifact is ignored under
`benchmarks/results/2026-07-27/4e5ac50-worktree/blackwell16gb-mtp-fused-verifier/`.

Kernel characterizations below are development evidence, not accepted end-to-end benchmark claims. They use one
deterministic activation and the pinned Layer-0 checkpoint tensors on the current Windows Blackwell machine.
Repeated isolated projection measurements keep one 33.3 MB tensor family hot in cache; do not add their times to
estimate a layer. The complete MLP row cycles through the 99.5 MB three-projection working set and is the more useful
decode characterization.

## 2026-07-27 Linux baseline refresh

After returning from Windows development, commit `304a113` was configured and built from a clean Linux
Blackwell-release tree. Host and CUDA CTest pass. The current direct-load checkpoint-FP8 plan was then measured at
batch one with three warm-ups and ten runs:

| Workload | Median throughput | Median TTFT | p50/p95/p99 ITL | Determinism |
|---|---:|---:|---:|---|
| Prefill 128 | 2,575.72 tok/s | 49.70 ms | — | one first-token ID |
| Prefill 512 | 4,277.03 tok/s | 119.78 ms | — | one first-token ID |
| Prefill 2,048 | 4,318.31 tok/s | 475.05 ms | — | one first-token ID |
| Prefill 8,192 | 3,674.54 tok/s | 2,229.39 ms | — | one first-token ID |
| Decode 128/256 | 32.912 tok/s | 58.59 ms | 30.38/32.11/32.42 ms | one checksum |
| Decode 2,048/256 | 33.082 tok/s | 483.25 ms | 30.20/30.83/31.35 ms | one checksum |
| Decode 8,192/256 | 32.380 tok/s | 2,253.46 ms | 30.84/31.39/31.66 ms | one checksum |

The 128/512/2,048 prefill samples are visibly bimodal because clocks were not locked and short executions alternate
between power states; retain their raw distributions and do not overstate small differences. The 8K samples are
stable, with throughput 95% CI `[3,663.63, 3,680.90]`. A current Nsight trace attributes 21.9%/20.9%/20.2%/17.9%
of GPU kernel time to global attention, local attention, FP8 projection GEMMs, and NVFP4 projection GEMMs. This
supersedes the pre-CUTLASS 65% projection diagnosis and selects attention staging as the next measured target.
Artifacts remain under
`benchmarks/results/2026-07-27/304a113/blackwell16gb-linux-refresh/`.

## 2026-07-27 exact decode-boundary and controlled Q/K fusion

A fresh whole-model graph-node profile at context 8K identified 1,588 kernel nodes per output token. Ordinary
decode still launched pointwise sequences already fused and qualified for prefill, while Q/K used eight separate
rounding, normalization, and RoPE launches per layer. Decode now reuses the exact RMSNorm/FP8,
RMSNorm/NVFP4, and Gate/Up/GELU/NVFP4 boundaries. A new controlled Q/K kernel reads the dynamic graph position and
uses the same initialization-time exact local/global RoPE tables while preserving every BF16 boundary.

The final graph has 964 kernel nodes/token (-39.3%). Under graph-node tracing, summed kernel time falls from
31.517 to 30.447 ms/token (-3.39%) even though projection, attention, and output-head kernels are individually
slower in the candidate trace, making the direction conservative. Final 3-warm-up/10-run medians are
34.446/34.257/33.545 tok/s at context 128/2,048/8,192. The final 8K result is +2.10% over the nearby 32.853 tok/s
parent and reaches 88.1% of the retained 38.056 tok/s direct-vLLM characterization, versus 85.6% previously. The
new Q/K kernel uses 24 registers, 2,048 bytes shared memory, and zero stack/local memory. Arenas are unchanged;
peak process VRAM is 9,852 MiB and Nsight observes no token-loop allocation.

CTest, exact-blue, 129/257 boundaries, teacher-forced 121/127 Top-1 plus 127/127 Top-5/Top-20, sampled CPU/GPU
selection, byte-exact local/global controlled-Q/K fixtures, memcheck, racecheck, and deterministic decode checks
pass. Qualification also added the missing shared-reduction consumption barrier to the reused decode RMSNorm/FP8
kernel; the final benchmark includes that correctness fix. NVFP4 two/eight-warp geometries were neutral; a
combined attention-residual/MLP-quantization kernel regressed
about 1.7%; both experiments were removed. Artifacts are under
`benchmarks/results/2026-07-27/4096fc8-worktree/blackwell16gb-linux-decode-sprint/`.

## 2026-07-27 sampled whole-model graph and operator isolation

Sampling is moved from the monolithic inference translation unit into `src/cuda/sampling/`, with shared host
validation, synthetic CUDA processor/RNG tests, and model-independent radix-sort graph capture/replay. The sampled
whole-model graph now copies token, position, suppression count, and RNG step through its fixed control record,
marks repetition history, performs exact selection, and copies one token to the host. Diagnostic state/logit
captures keep the direct path.

Final nearby context-128, 256-output, 3-warm-up/10-run measurements are 32.989 tok/s for top-k-64 sampling,
33.003 tok/s for unfiltered full-vocabulary sampling, and 32.839 tok/s for greedy. The small positive deltas are
treated as parity/run variance, not sampling speedups. The atomic repetition bitset removes duplicate-token write
races. A CUB double-precision inclusive probability scan plus final binary searches replace the bring-up serial
scan and bring unfiltered sampling from 22.166 tok/s to parity. Sampling workspace overhead is 7,408,128 bytes.
All modes are deterministic. A four-output Nsight trace records exactly three `cudaGraphLaunch` calls for the three
post-prefill outputs. All four `cudaMalloc` calls end before `gem16.initialize`, and none occur in sampled prefill
or decode. This closes the sampling implementation gate. Artifacts are under
`benchmarks/results/2026-07-27/f1730fd-worktree/blackwell16gb-linux-sampling-graph/`.

## 2026-07-27 exact GPU sampling bring-up

The first explicit sampling plan applies temperature, exact top-k/top-p/min-p, full-history repetition penalty,
and seeded SplitMix64 selection on the GPU. Sampling-disabled execution retains the prior fused greedy graph and
does not allocate sampling workspace. An adjacent context-128, 256-output, 3-warm-up/10-run characterization
measures 32.596 sampled tok/s versus 32.819 greedy tok/s (`0.9932x`, a 0.68% reduction). Workspace rises from
661,288,704 to 666,829,056 bytes (+5,540,352 bytes); model, cache, and graph allocations are unchanged.

A four-output Nsight trace attributes 0.231 ms total to CUB radix onesweep, 0.027 ms to final selection, and
0.017 ms to logit preparation, versus 13.409 ms in the shared fused output head. The preparation/selection kernels
use 24/36 registers with zero stack or local memory. All four observed `cudaMalloc` calls end before
`gem16.initialize`; sampled prefill/decode contains none. The CPU full-logit oracle selects token 532 from the same
three eligible tokens, repeated seeded runs match, top-k 1 matches greedy, and changing the seed changes a
multi-step sequence. This serial bring-up selector was subsequently replaced by the parallel probability-scan plan
recorded above.
Artifacts are under
`benchmarks/results/2026-07-27/61c141d-worktree/blackwell16gb-linux-gpu-sampling/`.

## 2026-07-27 current-commit cross-engine characterization

At commit `c93a40d`, fresh same-machine 3-warm-up/10-run measurements compare gem16 with direct-checkpoint vLLM
0.25.1 using checkpoint FP8 KV and with the patched closest-parity llama.cpp candidate using BF16 KV and
BF16-mapped attention weights. gem16 reaches 57.1%/67.0%/77.0%/77.5% of vLLM prefill at 128/512/2,048/8,192 and
84.7%/85.5%/85.6% of vLLM decode at 128/2,048/8,192. Against llama.cpp it reaches
110.9%/166.6%/177.8%/164.8% in prefill and 112.6%/117.0%/116.4% in decode.

The result is directional, not exact parity: timing boundaries differ, llama.cpp formats differ, vLLM's FP4
autotuner records OOM/default fallbacks and an untuned 8K shape, and continuous power/clock telemetry is absent.
The bounded attention-staging sprint is therefore closed rather than presented as vLLM parity. Current Nsight
leaves FP8 GEMMs and global attention tied at approximately 21.9% each, with no further low-risk staging change
selected. Work moves to required GPU sampling while the retained profile remains the basis for a later performance
sprint. Full methodology and raw samples are under
`benchmarks/results/2026-07-27/c93a40d/blackwell16gb-linux-cross-engine/`.

## Wikipedia 16K end-to-end characterization

At base commit `7d29580`, a pinned Wikipedia summarization workload supplied the exact same 16,384 prompt token IDs
to gem16, vLLM 0.25.1, and the patched same-source llama.cpp candidate. Each engine used batch one, greedy
selection, three warm-ups, ten measurements, and an 8,192-token output limit with normal EOS handling. Median
prefill was 1,897.37/4,328.03/2,160.83 tok/s and median decode was 31.324/33.971/28.843 tok/s respectively.
gem16 therefore reached 43.84% of vLLM prefill and 92.21% of vLLM decode, while reaching 87.81% of llama.cpp
prefill and 108.60% of llama.cpp decode. gem16 and vLLM used FP8 KV; llama.cpp used Q8_0 KV and its
closest-parity GGUF maps source FP8 attention weights to BF16.

All representative outputs were plausible German summaries. vLLM generated the same 1,215-token output in all ten
runs and llama.cpp the same 1,088-token output. gem16 generated 1,021-1,254 tokens with ten distinct hashes.
This nominally greedy non-determinism makes the retained result a development characterization. It was subsequently
traced to a shared-memory race in the long-context online decode-attention reductions and fixed after the benchmark;
the original samples remain unchanged as provenance. Full methodology and retained samples are in
`benchmarks/baselines/wikipedia_summary_16k/`.

The smaller correction gate uses 512 prompt tokens, 256 forced generation steps, and five fresh engine processes.
It produces one hash before and after the fix because the race did not manifest at that history length. The
minimized failing gate instead uses the same 16K prompt with only 64 generation steps: before the fix, three runs
produce three hashes and diverge at output steps 22 and 59; afterward, five runs have one hash. Targeted CUDA
Racecheck reports zero Split/Merge hazards after the added result-consumption barriers. This entry is correctness
evidence. The subsequent engine-only full-workload rerun produces one 1,106-token hash across all ten measurements.
Its median prefill/decode results are 1,892.37/31.216 tok/s versus 1,897.37/31.324 tok/s originally, or
-0.26%/-0.34%. Median TTFT/ITL move from 8,635.11/31.925 ms to 8,657.92/32.035 ms. Because the new barriers execute
only in decode while prefill shifts by a similar fraction, treat the decode difference as a sub-percent combined
barrier and system-drift cost, not a fully isolated kernel delta.

## Native prefill and long-context decode characterization

The 2026-07-26 long-context decode promotion replaces the three-kernel score/softmax/value path above 512 planned
positions with product-shape checkpoint-FP8 online attention. Local D256 layers group the two queries sharing each
KV head and split the 1,024-token ring into 256-token ranges. Global D512 layers process four of the sixteen
queries sharing the sole KV head together and split the growing cache into 512-token ranges. CTAs write normalized
partial outputs plus FP32 log-sum-exp state; one small graph-captured kernel merges those states. Plans through 512
positions retain the prior path because the split plan under-occupies the GPU there.

On the Linux RTX 5080 Laptop, the promoted 8,192-context path reaches 30.02 output tok/s with 33.21/35.19/36.65 ms
p50/p95/p99 latency over 3 warm-ups and 10 measured runs of 64 output tokens. All ten runs produce checksum
`17504476492555856403`. The immediately preceding development tree was reported at approximately 15 tok/s at 8K;
that observation has no retained adjacent raw artifact and is therefore not used as an accepted speedup claim.
The older retained orientation point is 12.68 tok/s but also predates later projection improvements. A
CUDA-Graph-node Nsight trace attributes approximately 2.80 ms/token to local split plus merge and 2.32 ms/token to
global split plus merge. The attention operator agrees with the score-matrix oracle at max absolute error
`3.73e-8` local and `1.86e-7` global, with cosine 1.0 for both production shapes. Context 128 retains its prior path
and measures 32.13 tok/s in the 1/3 regression check.

The active work program, required gates, and promotion policy are fixed in
[`docs/PREFILL_OPTIMIZATION_PLAN.md`](PREFILL_OPTIMIZATION_PLAN.md). At `1bc942b`, the Linux 512-token median is
698.25 tok/s versus the retained vLLM orientation point of 6,146.50 tok/s. A direct Nsight comparison attributes
approximately 289.78/199.77/131.13 ms of gem16 GPU time per execution to NVFP4 projections, attention, and FP8
projections, versus 24.23/13.11/27.15 ms for vLLM. The plan therefore prioritizes online Tensor-Core attention,
then larger deterministic prompt chunks, large pipelined NVFP4 CTA tiles, large/grouped FP8 projection tiles, and
only then residual profile-proven fusions. These values identify work; they are not parity benchmark claims.

The first consolidated Linux run at base commit `960528d` fixes the production plan to native SM120 projections,
chunked/fused prefill, separate Gate/Up/GELU, complete decode graphs, and fused output reduction. It uses checkpoint
FP8 KV and the standard 3 warm-up/10 measured policy. These ratios are orientation only: retained llama.cpp and
vLLM artifacts use BF16 KV, llama.cpp maps attention weights to BF16, and prefill timing boundaries differ.

| Workload | gem16 median | llama.cpp | vLLM | gem/llama | gem/vLLM |
|---|---:|---:|---:|---:|---:|
| Prefill 128 | 577.15 tok/s | 2,214.70 | 4,679.49 | 0.261x | 0.123x |
| Prefill 512 | 442.68 tok/s | 2,628.27 | 6,146.50 | 0.168x | 0.072x |
| Decode context 128, 256 tokens | 24.57 tok/s | 29.67 | 37.06 | 0.828x | 0.663x |

Decode p50/p95/p99 inter-token latency is 40.61/42.66/43.05 ms and all ten measured runs have the same output
checksum. Raw samples and machine metadata are under
`benchmarks/results/2026-07-25/960528d/blackwell16gb-linux/`. Results remain unqualified because quality acceptance,
native runtime-dispatch profiling, and continuous power/clock/thermal telemetry are still open.

The fixed prefill chunk was widened on the same Linux machine after the initial capture. At context 128/512, the
128-token plan retains the exact first-token IDs and improves the 3-warm-up/10-run medians from 577.15/442.68 to
725.48/545.25 tok/s (+25.7%/+23.2%), reducing median TTFT from 222.16/1156.59 to 176.53/939.02 ms. A 129-token
and a 257-token prompt produce exactly identical eight-token output-token sequences against a separately built
32-token baseline. Nsight Systems confirms 73,728 to 18,432 GPU operations across the two profiled runs (4x fewer)
and 2.38 s to 1.89 s projected GPU time. Raw runs are under
`benchmarks/results/2026-07-25/36c5041-worktree/blackwell16gb-linux-chunk128/`; the result reports the selected
chunk size. The context-budgeted selector keeps the score arena at or below 512 MiB for long-context plans.

The next Linux promotion makes every NVFP4 prefill warp consume two consecutive 16-token MMA tiles while loading
each Gate, Up, or Down weight fragment once. Against a separately built `8f05333` reference under the same 3/10
policy at context 512, median throughput rises from 542.58 to 587.68 tok/s (+8.31%) and median TTFT falls from
943.64 to 871.23 ms (-7.67%). Nsight Systems attributes 669.99 ms to the reference NVFP4 projection kernels and
547.20 ms to the promoted kernels (-18.33%) across two prefill executions. The 129- and 257-token prompts retain
exactly identical eight-token sequences, the exact-blue fixture passes, CUDA/unit tests pass, and the hot kernels
use no stack or local memory. Raw samples and profile summaries are under
`benchmarks/results/2026-07-25/8f05333-worktree/blackwell16gb-linux-nvfp4-tile2/`.

Applying the same two-tile mapping to the FP8 Q/K/V/O batch projections raises the context-512 median from 587.87
to 605.33 tok/s (+2.97%) against a separately built `b032e6f` reference and lowers median TTFT from 870.95 to
845.82 ms (-2.89%). Nsight Systems measures 280.18 to 245.73 ms (-12.30%) in FP8 projection kernels across two
prefill executions. The kernel uses 56 registers with no stack or local memory; exact-blue and the exact 129/257
eight-token sequence gates pass. Evidence is under
`benchmarks/results/2026-07-25/b032e6f-worktree/blackwell16gb-linux-fp8-tile2/`.

The fused checkpoint-FP8 attention QK phase next replaces scalar byte loads with aligned 16-byte loads while
retaining the exact serial FMA order. Against a separately built `c0c9b42` reference at context 512, the 3/10
median improves from 603.42 to 698.25 tok/s (+15.72%) and TTFT falls from 848.50 to 733.27 ms (-13.58%). Nsight
Systems measures 705.49 to 399.53 ms (-43.37%) for fused attention across two prefill executions and -14.16% total
projected prefill GPU time. A dedicated 32-dimensional FP8 operator fixture is bit-identical to the scalar
score/softmax/value chain; exact-blue and the 129/257 eight-token sequences also match. Evidence is under
`benchmarks/results/2026-07-25/c0c9b42-worktree/blackwell16gb-linux-vectorized-attention/`.

The Gemma-specific online-attention promotion then replaces the score matrix and scalar QK/PV loops with distinct
local D256 and global D512 BF16 Tensor-Core kernels, FP32 online-softmax state, and direct current-chunk/cached K/V
staging. At context 512 the 1,024-token plan reaches 973.15 tok/s median versus the preceding 698.25 tok/s
orientation point (+39.4%) and lowers median TTFT to 526.17 ms. Nsight attributes 24.41 ms per execution to both
online attention families, down from 199.77 ms (8.18x), while GPU operations fall from about 9,235 to 2,311. The
remaining per-execution costs are approximately 286.97 ms NVFP4 projections and 122.95 ms FP8 projections, making
the large NVFP4 CTA phase the next bottleneck. Local/global operator errors are respectively bounded by
max-absolute 0.001013/0.000538 with cosine at least 0.999993/0.999997; both hot kernels have zero stack and local
memory. Direct vLLM boundary fixtures place the vLLM Top-1 at engine rank 1 for both 129 and 257 prompt tokens.

A 3-warm-up/10-run chunk comparison selects 1,024 as the sole checkpoint-FP8 plan. At 2,048 prompt tokens its
915.24 tok/s median and 95% CI `[913.44, 918.03]` beat the 512-token plan's 893.60 tok/s and
`[892.82, 894.88]` (+2.42%). Context-512 medians, 973.15 and 976.41 tok/s, are statistically indistinguishable;
the longer plan therefore decides the selection. Reusable workspace rises from 218,451,456 to 435,275,264 bytes
at context 2,048 and remains within the 16 GB budget. The fixed exact-blue generation, full CTest suite, and direct
boundary gate pass. The production-path teacher-forced comparison remains 118/127 Top-1, while its Top-5 coverage
is 126/127 after logit capture was corrected to stop bypassing batch prefill.

The first large-M NVFP4 step refactors the production batch kernel so a warp retains every 8-column packed-weight
and scale fragment across eight independent `m16n8k64` operations, covering 128 prompt rows instead of 32 without
changing any tile's K accumulation. Relative to the immediately preceding `a375583` evidence, 3/10 median
throughput rises from 948.73/973.15/915.24 to 1,024.10/1,095.56/1,032.91 tok/s at 128/512/2,048
(+7.9%/+12.6%/+12.9%). Median TTFT falls to 125.04/467.70/1,982.74 ms. At 512 and 2,048, the before/after 95%
confidence intervals do not overlap. Nsight reduces NVFP4 time from 286.97 to 233.33 ms per 512-token execution
(-18.7%) and total GPU time from about 540.25 to 493.53 ms (-8.6%). The selected kernel uses 128 registers and
zero stack/local memory; the attempted M256 extension is rejected at 255 registers plus a 248-byte stack frame.
CTest, exact-blue, both vLLM boundary logits, and every aggregate metric in the 12-prompt/127-position suite are
unchanged. No workspace, weight layout, or persistent memory changes.

The next CTA promotion groups eight of those warps into an M128xN64 block and cooperatively stages each K64
activation slice and its E4M3 scales once in shared memory. Against a separately built `2366c03` reference, 3/10
median throughput rises from 1,000.70/1,099.00/1,031.94 to 1,260.67/1,427.00/1,307.64 tok/s at
128/512/2,048 (+26.0%/+29.8%/+26.7%); median TTFT falls by 20.6%/23.0%/21.1%. At 512 and 2,048 the 95%
confidence intervals do not overlap. Adjacent Nsight profiles reduce Gate+Up from 149.97 to 75.94 ms per
execution (-49.4%), Down from 84.74 to 38.63 ms (-54.4%), and total NVFP4 projection time from 234.71 to
114.58 ms (-51.2%). The CTA uses 123 registers, 5,632 static shared bytes, and zero stack/local memory. It adds no
arena or persistent allocation and leaves packed weights in their source layout. CTest, exact-blue, vLLM boundary
rank/logprob metrics, and all 12-prompt/127-position teacher-forced aggregates are identical to the parent.
Evidence is under
`benchmarks/results/2026-07-25/abb430c-worktree/blackwell16gb-linux-nvfp4-cta-m128n64/`.

The CTA's synchronous activation stage is next replaced directly by two ping-pong buffers populated with 16-byte
packed-activation and 4-byte scale `cp.async` transfers. A neighboring 30-run context-512 comparison raises mean
throughput from 1,412.39 to 1,439.57 tok/s (+1.92%) and median from 1,413.45 to 1,437.55 (+1.71%); paired
differences have a 95% interval of +10.68 to +43.69 tok/s. The 2,048-token 3/10 median rises from 1,307.64 to
1,329.51 tok/s (+1.67%) with non-overlapping confidence intervals. Nsight reduces NVFP4 time from 119.04 to
98.94 ms per execution (-16.9%) and projected total GPU time from 364.08 to 336.71 ms (-7.5%). The generated SASS
contains `LDGSTS`; the selected kernel uses 124 registers, 10,240 shared bytes, and zero stack/local memory. All
fixed generation, boundary-logit, and teacher-forced metrics remain identical. Evidence is under
`benchmarks/results/2026-07-25/d8b73ce-worktree/blackwell16gb-linux-nvfp4-async-pipeline/`.

The following console characterizations were collected on the same Windows Blackwell development machine after
commits `0d2065e` and `914aba1`, using direct checkpoint loading, checkpoint FP8 KV, native SM120 projections, and
the opt-in fused Gate/Up path. They are not accepted benchmark artifacts: 128 and 512 prefill use the full 3 warm-up/
10 measured policy, while the expensive 2K and 8K scaling points deliberately use fewer repetitions. The existing
llama.cpp and vLLM values come from their separately retained development runs, so ratios are orientation rather
than parity claims.

| Prompt | gem16 prefill tok/s | Repetitions | llama.cpp | vLLM | gem/llama | gem/vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 87.72 | 3/10 | 2,215 | 4,679 | 0.040x | 0.019x |
| 512 | 82.42 | 3/10 | 2,628 | 6,146 | 0.031x | 0.013x |
| 2,048 | 69.92 | 1/3 | 2,539 | 4,913 | 0.028x | 0.014x |
| 8,192 | 55.83 | 1/1 | 2,362 | 3,929 | 0.024x | 0.014x |

| Existing context | gem16 decode tok/s | gem p50/p95/p99 ms | Repetitions | llama.cpp | vLLM | gem/llama | gem/vLLM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 21.02 | 47.70 / 50.69 / 52.17 | 3/10 | 29.67 | 37.06 | 0.709x | 0.567x |
| 2,048 | 15.71 | 63.50 / 67.04 / 70.69 | 1/3 | 28.87 | 35.98 | 0.544x | 0.437x |
| 8,192 | 12.68 | 78.66 / 80.95 / 88.14 | 1/1 | 28.08 | 35.36 | 0.451x | 0.358x |

The original implementation result was positive for capability and negative for competitiveness: native prefill
preserved serial-path output checksums at 8 and 128 tokens and supported the hybrid cache through 8K in real runs,
but its token-parallel MMA grid remained 25x–75x behind the current prefill references. The first true M-dimensional
tile below removes that specific bottleneck. It is still well behind the reference engines; the follow-up attention
fusion removes two launches per layer, while wider/pipelined projection tiles remain the next prefill work.

| Date | Commit | Hypothesis | Configuration | Before | After | Quality delta | VRAM delta | Decision |
|---|---|---|---|---:|---:|---:|---:|---|
| 2026-07-28 | `e461d4d` worktree | A fixed-address device control record can prove GPU/host MTP scheduling parity before graph capture without changing exact output or the recurring synchronization boundary | Windows RTX 5080 Laptop, CUDA 13.3, exact Wikipedia 16K prompt, checkpoint FP8 KV, official assistant D2; 1 warm-up/3 measured; matching 256-output Nsight traces | Pre-change 46.000 tok/s median; 185,830 launches, 117 stream synchronizations, 14,594 async copies; 706,618,112-byte workspace | 45.217 tok/s median; 185,830 launches, 117 stream synchronizations, 14,705 async copies; 706,618,112-byte workspace | All three runs retain 1,135 IDs and SHA-256 `43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`; 632/372 accepted/rejected over 502 groups; CUDA fixture covers acceptance 0/1/2 and stop truncation; BF16 D1/D2/D4 and FP8 D2 equal ordinary; zero fallbacks/loop allocations | No reported workspace growth; transaction expansion is absorbed by existing 256-byte arena alignment | Promote as a bounded correctness foundation; the 1.7% short-run median reduction and slower profiled range are not attributed to the 111 tiny H2D records because the trace is clock/noise sensitive; require the next fixed-D2 graph phase to eliminate launch overhead materially |
| 2026-07-27 | `4dc1020` worktree | Global cache positions are already in bounds and contiguous, so removing redundant ring-style modulo eliminates repeated integer division without changing any address | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV; two adjacent parent/candidate 8K pairs with 3 warm-ups/10 measured plus adjacent Nsight | Combined parent 3,801.98 tok/s, 2,154.66 ms TTFT; global kernel 1.03955 s/two prefills | Combined candidate 3,827.47 tok/s (+0.67%), 2,140.32 ms TTFT (-0.67%); global kernel 0.96426 s (-7.24%); total profiled kernels -1.08% | Global operator fixtures and CTest pass; exact-blue and 129/257 boundaries pass; teacher-forced remains 121/127 Top-1 and 127/127 Top-5/Top-20; 8K decode checksum unchanged | No arena, workspace, cache, or kernel-resource change; 254 registers, 99,328 bytes shared, zero stack/local | Promote direct absolute indexing for contiguous global K/V; retain modulo only for local rings; reject paired global-query BF16 stores after an adjacent -0.42% result |
| 2026-07-27 | `304a113` worktree | Aligned 16-byte local K/V loads, E4M3x4 conversion, and paired BF16 stores can remove scalar staging overhead without changing attention arithmetic | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV; prefill 128/512/2,048/8,192 with 3 warm-ups/10 measured; adjacent 8K Nsight | 2,575.72/4,277.03/4,318.31/3,674.54 tok/s; 8K local kernel 0.96996 s/two prefills | 2,544.03/4,447.47/4,433.51/3,815.49 tok/s (-1.23%/+3.99%/+2.67%/+3.84%); 8K TTFT 2,229.39 to 2,147.04 ms; local kernel 0.70919 s (-26.88%); total profiled kernels -4.39% | Local CUDA fixtures and CTest pass; exact-blue and 129/257 boundaries pass; teacher-forced remains 121/127 Top-1 and 127/127 Top-5/Top-20; 8K decode checksum unchanged | No arena, workspace, persistent allocation, or shared-memory growth; 254 registers, 66,560 bytes shared, zero stack/local | Promote vector FP8x4/BF16x2 local staging; short-context samples are clock-bimodal, stable 8K intervals do not overlap |
| 2026-07-26 | working tree | Overlay the mutually exclusive BF16 K/V operands and use the recovered shared space for raw-FP8 ping-pong buffers, overlapping V copies with QK and next-K copies with PV | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV; 8K prefill, adjacent Nsight and 3 warm-ups/10 measured | 3,683.18 tok/s, 2,224.17 ms TTFT; global kernel 1.03791 s/two prefills | 3,704.64 tok/s (+0.58%), 2,211.28 ms TTFT (-0.58%), 95% throughput CI `[3,692.82, 3,709.30]`; global kernel 0.98382 s (-5.21%); exploratory 16K 3,154.66 tok/s (+1.91%) | Global operator max abs 0.000538, RMS 0.000126, cosine 0.999997; exact-blue and 129/257 vLLM Top-1 pass; teacher-forced 121/127 Top-1 and 127/127 Top-5/Top-20; restored 8K decode checksum deterministic | No arena, persistent allocation, workspace, or shared-memory growth; global CTA remains 96 KiB | Promote asynchronous raw-FP8 double buffering for global prefill; local prefill and decode unchanged |
| 2026-07-26 | rejected worktree | T=1 decode attention may benefit from Tensor-Core QK/PV despite its one-row query geometry | Linux RTX 5080 Laptop, CUDA 13.3, physical FP8 KV; 8K context/256 generated, BF16 and TF32 WMMA prototypes | Scalar online split GQA: 33.349 tok/s | TF32 local+global 28.797 tok/s (-13.65%); global-only split-16 32.232 (-3.35%); global-only split-8 32.292 (-3.17%); restored scalar 33.236 tok/s | BF16 rejected numerically; TF32 operators pass (long-global max abs 0.00000551, cosine 1); restored checksum `14820510372112584179`; CTest passes | Prototype added no persistent allocation; all prototype code removed | Reject: at M=1, tile padding, staging, synchronization, and lower CTA residency cost more than Tensor-Core arithmetic saves |
| 2026-07-26 | working tree | Aligned 16-byte loads, E4M3x4 conversion, and paired BF16 stores can remove scalar global-attention K/V staging overhead without changing its MMA or online-softmax order | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV; 8K prefill, adjacent Nsight and 3 warm-ups/10 measured | 3,539.55 tok/s, 2,314.42 ms TTFT; global kernel 1.2867 s/two prefills | 3,683.18 tok/s (+4.06%), 2,224.17 ms TTFT (-3.90%), 95% throughput CI `[3,680.21, 3,692.22]`; global kernel 1.0379 s (-19.34%) | Global operator max abs 0.000538, RMS 0.000126, cosine 0.999997; exact-blue and 129/257 vLLM Top-1 pass; teacher-forced 121/127 Top-1 and 127/127 Top-5/Top-20; three 8K decode runs deterministic | No arena, persistent allocation, or shared-memory change; 8K workspace remains 673,808,384 bytes | Promote vector FP8x4/BF16x2 global K/V staging; retain local attention and all attention arithmetic |
| 2026-07-26 | `72426a9` | Checkpoint-order FP8 weights can feed a regular CUTLASS SM120 GEMM directly, leaving only exact per-token/per-channel output scaling | Linux RTX 5080 Laptop, CUDA/CUTLASS 13.3/4.5.2, checkpoint FP8 KV; 8K prefill, adjacent 3 warm-ups/10 measured | 2,910.53 tok/s, 2,814.61 ms TTFT | 3,539.55 tok/s (+21.61%), 2,314.42 ms TTFT (-17.77%), 95% throughput CI `[3,528.28, 3,543.69]` | Real 128x4,096x3,840 fixture is bit-exact across 524,288 FP32 outputs; exact-blue and 129/257 vLLM Top-1 pass; teacher-forced 121/127 Top-1 and 127/127 Top-5/Top-20; three 8K decode runs deterministic | No arena change: 8K workspace 673,808,384 bytes; weight/KV arenas and persistent repack unchanged | Promote separate CUTLASS Q/K/V/O prefill GEMMs; retain grouped native FP8 only for decode |
| 2026-07-26 | `cf41e6f` | The real Down shape can amortize the same temporary CUTLASS SM120 layout conversion as Gate/Up, and its BF16 result can feed the fused residual/RMSNorm boundary directly | Linux RTX 5080 Laptop, CUDA/CUTLASS 13.3/4.5.2, checkpoint FP8 KV; 8K prefill, adjacent 3 warm-ups/10 measured | 2,594.28 tok/s, 3,157.72 ms TTFT | 2,910.53 tok/s (+12.19%), 2,814.61 ms TTFT (-10.87%), 95% throughput CI `[2,904.41, 2,915.00]` | Real 128x3,840x15,360 fixture has zero BF16 mismatches; exact-blue and 129/257 vLLM Top-1 pass; teacher-forced 121/127 Top-1 and 127/127 Top-5/Top-20; three 8K decode runs deterministic | 8K workspace 672,333,824 to 673,808,384 bytes (+1,474,560); weight/KV arenas and persistent repack unchanged | Promote CUTLASS Down prefill; retain native Row8/K64 only for decode |
| 2026-07-26 | working tree | A CUTLASS SM120 block-scaled persistent GEMM can repay an in-arena layout transform for the large Gate/Up prompt projections while preserving the decode-optimal persistent layout | Linux RTX 5080 Laptop, CUDA/CUTLASS 13.3/4.5.2, checkpoint FP8 KV; 8K and 2K prefill, 3 warm-ups/10 measured | 8K 2,135.93 tok/s, 3,835.33 ms TTFT; 2K approximately 2,428 tok/s | 8K 2,584.77 tok/s (+21.0%), 3,169.34 ms TTFT (-17.4%); 2K 2,984.77 tok/s (+22.9%) | Real 2,048x128x3,840 projection has zero BF16 mismatches; exact-blue, 129/257 vLLM Top-1, teacher-forced 118/127 Top-1 and 126/127 Top-5, CTest pass; five 8K decode runs deterministic | 8K workspace 630,276,096 to 672,333,824 bytes (+42,057,728); weight/KV arenas and persistent repack unchanged | Promote CUTLASS Gate/Up prefill; retain native Down and all decode projections |
| 2026-07-26 | working tree | Share staged K/V across GQA heads, widen FP8 projection M reuse, and halve 8K chunk groups without changing per-head MMA order | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV, 8K prefill; final 3 warm-ups/10 measured; adjacent Nsight | Row8/K64 start: 1,560.234 tok/s, 5,250.495 ms TTFT; final 1K plan: 2,107.039 tok/s | Grouped local/global heads + FP8 M128 + 2K chunk: 2,138.504 tok/s (+37.06% from start, +1.49% from final 1K), 3,830.718 ms TTFT (-27.04% from start), 95% throughput CI `[2,134.606, 2,144.236]` | Exact-blue `[9503,106]`; 8K first token `496`; 129/257 vLLM Top-1 rank 1; teacher-forced 118/127 Top-1 and 126/127 Top-5; CTest passes; 8K decode checksum unchanged at 33.676 tok/s | 8K workspace 322,457,600 to 630,276,096 bytes; weight/KV and persistent repack unchanged; hot kernels have zero stack/local memory | Promote 2/4-head local/global attention CTAs, FP8 M128xN64xK64, and 2K checkpoint-FP8 chunks; projections are now 65.0% of profiled kernel time |
| 2026-07-26 | working tree | Exact Row8/K64 tiling of packed E2M1 weights at load time coalesces each warp's eight row fragments for both T=1 and batch SM120 kernels | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV, 8K context/64 decode tokens, 1 warm-up/3 measured | Source-row weights plus tiled scales: 31.604 tok/s median, p50 31.498 ms | Tiled weights and scales: 33.143 tok/s (+4.87%), p50/p95/p99 30.184/32.498/33.234 ms; 8K prefill 1,560.234 tok/s | Native/source-reference CUDA projection tests pass; complete Layer-0 MLP native/reference max abs 0 and cosine 1; exact-blue `[9503,106]`; 129/257 prefill Top-1 rank 1; teacher-forced 118/127 Top-1 and 126/127 Top-5 unchanged; CTest passes | Weight arena unchanged at 9,200,135,680 bytes; no raw GPU copy; 4 MiB maximum host staging; `persistent_repack_bytes=0` | Promote Row8/K64 packed weights as the sole native runtime layout |
| 2026-07-26 | working tree | Local Q/K/V and global Q/K can share one T=1 binding-dimension launch without changing any projection CTA or MMA ordering | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV, 8K context/64 decode tokens; adjacent stabilized 1/1 A/B plus candidate 1/3 | Three launches/layer: 30.511 tok/s, checksum `17504476492555856403` | Grouped launch: 32.034 tok/s in adjacent A/B (+4.99%); 31.604 tok/s median over 1/3, p50/p95/p99 31.498/33.622/35.451 ms | Direct grouped Q/K/V operator outputs bit-identical to the independent direct kernel; exact-blue `[9503,106]`; CTest passes | No weight, KV, or workspace arena change; graph-associated device bytes remain 18,874,368 | Promote grouped T=1 Q/K/V as the sole decode projection schedule |
| 2026-07-26 | rejected worktree | Applying required BF16 rounding in the FP8/NVFP4 projection stores and GELU product kernel can remove decode round launches | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV, 8K context/64 decode tokens, adjacent stabilized 1/1 | Separate round kernels: 30.446 tok/s | Rounded stores: 30.261 tok/s (-0.61%) | Bit-identical projection/product fixtures and output checksum; exact-blue and CTest pass | Graph bytes fell 18,874,368 to 16,777,216; no arena change | Reject and delete: extra MMA-epilogue conversion does not repay the removed graph nodes |
| 2026-07-25 | `011411d` worktree | Gate/Up projections can write their required BF16 boundary directly, avoiding FP32 workspace traffic before fused GELU/NVFP4 quantization | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV; 128/512 use adjacent 3/30, 2,048 uses 3/10; Nsight 512 | 1,789.06/2,195.19/1,990.27 tok/s median; 195.29 ms NVFP4 and 10.98 ms GELU across two profiled prefills | 1,811.43/2,284.63/2,038.77 tok/s (+1.25%/+4.07%/+2.44%); 186.14 ms NVFP4 (-4.68%) and 9.02 ms GELU (-17.9%) | Exact-blue `[9503,106]`; unchanged 129/257 vLLM boundaries and teacher-forced 118/127 Top-1, 126/127 Top-5, 127/127 Top-20; CTest passes | 2K workspace 438,422,528 to 312,593,408 bytes (-125,829,120); measured peak process 9,466 MiB; 128 registers, 9,216 shared bytes, zero stack/local | Promote direct BF16 Gate/Up output as the sole production path and delete the dead FP32 product arena |
| 2026-07-25 | `f76d478` rejected worktree | Separate 8-warp Gate and Up groups can share the proven M128 activation stage in one 16-warp CTA without changing either projection's arithmetic | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV, contexts 128/512, adjacent 3/10 | 1,922.27/2,272.96 tok/s; 66.588/225.291 ms median TTFT | 1,897.51/2,196.93 tok/s (-1.29%/-3.34%); 67.460/233.057 ms TTFT | 129-token operator boundary exactly matches both reference projections; exact-blue, vLLM 129/257 boundaries, and teacher-forced 118/127 Top-1, 126/127 Top-5, 127/127 Top-20 are unchanged | No arena or persistent-memory change; 126 registers, 9,216 shared bytes, zero stack/local | Reject after two required sizes: saved activation traffic and one launch do not repay 512-thread CTA scheduling; delete implementation and retain separate Gate/Up projections |
| 2026-07-25 | `f76d478` rejected worktree | Two independent M128 token-warp groups can share one staged N64xK64 weight tile and form a spill-free M256xN64 CTA | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV, contexts 128/512, adjacent 3/10 | 1,872.94/2,272.73 tok/s | 1,801.04/2,201.07 tok/s (-3.84%/-3.15%) | Exact-blue remains `[9503,106]`; extended 257-token CUDA projection fixture passes | No arena or persistent-memory change; 124 registers, 24,064 shared bytes, zero stack/local | Reject after two required sizes: shared weight staging and the 512-thread scheduling unit are slower; delete implementation |
| 2026-07-25 | `f76d478` rejected worktree | The post-attention BF16 rounding, dynamic FP8 quantization, and safe K/V ring commit can share one token CTA | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV, context 512, 3/10 and adjacent Nsight | 2,285.60 tok/s; 964 launches/prefill; affected boundaries about 2.12 ms/prefill | 2,303.73 tok/s (+0.79%); 868 launches; fused boundary about 1.30 ms | Bit-identical CUDA payloads/scales/wrapped cache; exact same vLLM boundary and 118/127 teacher-forced metrics; exact-blue and CTest pass | No arena or persistent-memory change | Reject: end-to-end mean intervals overlap strongly; remove the complete implementation and retain no variant |
| 2026-07-25 | `ccbe4ed` worktree | Projection BF16 rounding, Q/K RMSNorm, and RoPE can share one exact kernel, while one max-context trigonometric table removes identical work repeated across 48 layers | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV; 128/512 use adjacent 3/30 due short-run outliers, 2,048 uses 3/10; Nsight 512 | 1,590.17/1,895.52/1,716.56 tok/s; 1,300 launches and 250.03 ms GPU kernels/prefill | 1,831.33/2,182.51/2,004.23 tok/s (+15.17%/+15.14%/+16.76%); 964 launches (-25.85%); 208.01 ms (-16.80%) | Local D256/global partial-D512 outputs bit-identical to eight-kernel oracle; exact same 129/257 metrics and 118/127 teacher-forced Top-1; exact-blue and CTest pass | Peak 9,586 MiB (+4 MiB); 2K workspace +3,147,264 bytes; hot kernel 35 registers/3,072 shared/zero stack-local | Promote fused Q/K RMSNorm/RoPE and persistent exact tables as the sole path; complete K/V-write fusion, then return to projections |
| 2026-07-25 | `bdb1294` worktree | Exact normalization, residual, GELU, and activation-quantization boundaries can share launches without changing their prescribed BF16/E4M3/E2M1 values | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV; prefill 128/512/2,048, 3/10; Nsight 512 | 1,540.69/1,748.61/1,563.23 tok/s; 2,165 launches/prefill | 1,664.23/1,914.76/1,722.95 tok/s (+8.0%/+9.5%/+10.2%); 1,300 launches (-40.0%); GPU kernel time -8.73% | Fused/unfused CUDA payloads and scales bit-identical; exact same 129/257 boundary metrics and 118/127 teacher-forced Top-1; exact-blue and CTest pass | Peak process 9,582 MiB; weight/KV/workspace arenas unchanged; fused kernels have zero stack/local memory | Promote all exact boundary fusions as the sole prefill path; next fuse Q/K norm, RoPE, and K/V write without sharing arithmetic |
| 2026-07-25 | `6005921` worktree | Eight warps can share exact source-layout FP8 operands through a two-stage M64xN64xK64 CTA, and Q/K/V can share one binding-dimension launch | Linux RTX 5080 Laptop, CUDA 13.3, checkpoint FP8 KV; prefill 128/512/2,048, 3/10; Nsight 512 | Direct two-tile warps: 1,348.97/1,460.83/1,358.11 tok/s; FP8 114.09 ms and 184 launches/prefill | Pipelined/grouped CTA: 1,583.23/1,769.04/1,562.05 tok/s (+17.4%/+21.1%/+15.0%); FP8 62.21 ms (-45.5%) and 96 launches | Grouped Q/K/V exactly equals CUDA reference; exact same 129/257 boundary metrics and 118/127 teacher-forced Top-1; exact-blue and CTest pass | Weight/KV/workspace arenas unchanged; 60 registers, 17,408 shared bytes, zero stack/local | Promote M64xN64xK64 and grouped Q/K/V as the sole FP8 prefill path; proceed to profile-proven fusions |
| 2026-07-25 | `e17049b` worktree | Tiling exact local-scale bytes by output row8 and K64 coalesces native SM120 scale vectors and removes strided decode addressing | Linux RTX 5080 Laptop, CUDA 13.3, FP8 KV; prefill 128/512/2,048 with 3/10, 3/30, 3/10; short decode 1/3; Nsight 512 | Direct scales: 1,317.91/1,433.07/1,312.48 tok/s; decode 25.54 tok/s; NVFP4 198.72 ms/2 executions | Tiled scales: 1,332.45/1,453.52/1,354.31 tok/s (+1.10%/+1.43%/+3.19%); decode 31.63 tok/s (+23.85%); NVFP4 189.55 ms (-4.61%) | Exact same boundary metrics and 118/127 teacher-forced Top-1; Layer-0 activation bytes unchanged; exact-blue and CTest pass | Weight arena unchanged at 9,200,135,680 bytes; max 3,686,400 host staging bytes; 128 prefill/40 decode registers, zero stack/local | Promote exact row8/K64 scale tiling as the sole native runtime layout; retain source order only in reference probes |
| 2026-07-25 | `d8b73ce` worktree | The next exact activation K64 slice can overlap the current MMA stack through ping-pong `cp.async` staging | Linux RTX 5080 Laptop, CUDA 13.3, FP8 KV; context 128/2,048 3/10, context 512 adjacent 3/30; Nsight context 512 | Synchronous N64 CTA: 1,413.45 tok/s median at 512; NVFP4 119.04 ms/execution | Two-stage async CTA: 1,437.55 tok/s (+1.71%); NVFP4 98.94 ms (-16.9%); 2K 1,329.51 tok/s (+1.67%) | Exact same boundary metrics and 118/127 teacher-forced Top-1 aggregate; exact-blue and CTest pass | No arena/persistent change; shared 5,632 to 10,240 bytes; 124 registers, zero stack/local memory | Promote the two-stage `cp.async` CTA as the sole NVFP4 prefill pipeline |
| 2026-07-25 | `abb430c` worktree | One M128 activation K64 slice can be staged once and reused across eight N8 output warps without changing native MMA arithmetic | Linux RTX 5080 Laptop, CUDA 13.3, FP8 KV; context 128/512/2,048, adjacent 3/10; Nsight context 512 | M128xN32: 1,000.70/1,099.00/1,031.94 tok/s; NVFP4 234.71 ms/execution | M128xN64 CTA: 1,260.67/1,427.00/1,307.64 tok/s (+26.0%/+29.8%/+26.7%); NVFP4 114.58 ms (-51.2%) | Exact same boundary metrics and 118/127 teacher-forced Top-1 aggregate; exact-blue and CTest pass | No arena or persistent-memory change; 123 registers, 5,632 shared bytes, zero stack/local memory | Promote M128xN64 shared-activation CTA as the sole NVFP4 prefill kernel; continue asynchronous pipeline work |
| 2026-07-25 | `a375583` worktree | Retaining each packed NVFP4 weight/scale fragment across a larger M tile reduces repeated source-layout traffic without changing per-tile arithmetic | Linux RTX 5080 Laptop, CUDA 13.3, FP8 KV; context 128/512/2,048, 3/10; Nsight context 512 | M32: 948.73/973.15/915.24 tok/s; NVFP4 286.97 ms/execution | M128: 1,024.10/1,095.56/1,032.91 tok/s (+7.9%/+12.6%/+12.9%); NVFP4 233.33 ms (-18.7%) | Exact same 118/127 teacher-forced Top-1 aggregate and boundary/full-generation gates; CTest passes | No arena or persistent-memory change; 128 registers, zero stack/local memory | Promote M128 as the sole production NVFP4 batch tile; reject M256 because it creates a 248-byte stack frame; continue CTA pipeline work |
| 2026-07-25 | `c0f42de` plus qualification worktree | Shape-specific Tensor-Core online attention and a full 1,024-token prompt tile remove score traffic, scalar QK/PV, and repeated layer launches | Linux RTX 5080 Laptop, CUDA 13.3, FP8 KV; context 128/512/2,048, 3/10; Nsight context 512 | Score-matrix path: 698.25 tok/s at 512; attention 199.77 ms/execution; ~9,235 GPU ops | Online path: 973.15 tok/s at 512 (+39.4%); attention 24.41 ms (8.18x faster); ~2,311 GPU ops; chunk 1,024 gives 915.24 tok/s at 2K vs 893.60 for chunk 512 | CUDA operator max abs <=0.001013 and cosine >=0.999993; vLLM Top-1 rank 1 at 129/257; exact-blue; 118/127 teacher-forced Top-1 | Score arena removed; context-2K workspace 435,275,264 bytes, +216,823,808 vs chunk 512; zero hot-kernel stack/local memory | Retain online local/global attention and 1,024 as the sole checkpoint-FP8 production plan; optimize NVFP4 projections next |
| 2026-07-25 | `c0c9b42` worktree | Wide FP8 key loads can remove inefficient byte transactions without changing attention arithmetic | Linux RTX 5080 Laptop, CUDA 13.3, FP8 KV, context 512, 3/10; separately built reference | Scalar byte loads: 603.42 tok/s, 848.50 ms TTFT | Aligned 16-byte loads: 698.25 tok/s (+15.72%), 733.27 ms TTFT (-13.58%) | Bit-identical 32-D FP8 fused/reference operator output; exact 129/257 sequences; exact-blue and CUDA/unit gates pass | No arena or persistent-memory change | Promote wide loads as the only checkpoint-FP8 fused prefill attention path; Nsight measures -43.37% attention time |
| 2026-07-25 | `b032e6f` worktree | Adjacent FP8 token tiles can share each attention-projection weight fragment exactly as the NVFP4 winner does | Linux RTX 5080 Laptop, CUDA 13.3, FP8 KV, context 512, 3/10; separately built reference | One token tile/warp: 587.87 tok/s, 870.95 ms TTFT | Two token tiles/warp: 605.33 tok/s (+2.97%), 845.82 ms TTFT (-2.89%) | Exact 129/257-token eight-step sequences; exact-blue and CUDA/unit gates pass; no spill | No arena or persistent-memory change | Promote as the sole FP8 batch projection; Nsight measures -12.30% FP8 projection time |
| 2026-07-25 | `8f05333` worktree | One warp can amortize each NVFP4 weight-fragment load over two 16-token MMA tiles without changing either tile's accumulation order | Linux RTX 5080 Laptop, CUDA 13.3, FP8 KV, context 512, 3/10; separately built reference immediately followed by candidate | One token tile/warp: 542.58 tok/s, 943.64 ms TTFT | Two token tiles/warp: 587.68 tok/s (+8.31%), 871.23 ms TTFT (-7.67%) | Exact 129/257-token eight-step sequences; exact-blue gate; CUDA/unit tests pass; no local-memory spill | No arena or persistent-memory change | Promote the two-tile kernel as the only production NVFP4 batch projection; Nsight measures -18.33% NVFP4 projection time |
| 2026-07-25 | `960528d` plus consolidation worktree | One fixed production plan prevents known-slower paths while Linux resolves the last Gate/Up ambiguity | RTX 5080 Laptop GPU, Linux, CUDA 13.3, FP8 KV; Prefill 3/10 at 128 and 512; Decode context 128, 64 tokens, 1/3 | Fused Gate/Up: 527.53/410.16 Prefill tok/s and 25.86 Decode tok/s | Separate Gate/Up/GELU: 573.32/441.73 Prefill tok/s and 26.08 Decode tok/s | Deterministic output checksums retained; existing operator gates cover both implementations | No arena change | Select separate Gate/Up/GELU; remove all six production optimization switch families while retaining references in tests/probes |
| 2026-07-25 | `36c5041` worktree | Larger prompt tiles reduce launch overhead and improve M-dimensional projection occupancy without changing arithmetic | Linux RTX 5080 Laptop, CUDA 13.3, FP8 KV, context-budgeted 128-token chunk, 3/10 | 32-token chunks: 577.15/442.68 Prefill tok/s at 128/512; 15.32/16.13 MB workspace | 128-token chunks: 725.48/545.25 tok/s; 56.77/59.94 MB workspace | Exact 129/257-token eight-step sequences; exact-blue gate; CUDA/unit tests pass | +41.44/+43.81 MB workspace; long plans lower chunk size to retain a 512 MiB score budget | Promote 128-token default; retain only deterministic memory-bounded selection for long contexts |
| 2026-07-25 | working tree | Projection work and launch count explain the large prefill/decode gap | Nsight Systems 2026.1.3; native chunked prefill and persistent decode; context 128; FP8 KV; fused Gate+Up | Prefill trace: 21,846 launches; decode trace: 85,116 launches over 34 forwards | Prefill GPU time: Gate+Up 53%, FP8 projections 20%, NVFP4 Down 17%; decode GPU time: 45%, 21%, and 18% respectively | Existing native/serial checksum gate remains unchanged; this entry only adds instrumentation | Nsight reports remain in the untracked build profile directory | Build a true M-dimensional projection path first; follow with CUDA Graph decode |
| 2026-07-25 | working tree | One warp can reuse each FP8/NVFP4 weight fragment across 16 prompt rows | Native 32-token chunks, real `M16N8` MMA tiles, FP8 KV, fused Gate+Up | 128: 87.72 tok/s; 512: 82.42 tok/s | 128: 516.40 tok/s (3/10); 512: 392.61 tok/s (1/3) | Native/serial first-token identity at 8, 32, and 128 tokens; FP8/NVFP4 CUDA batch tests cover 17-token full-plus-tail geometry | No arena growth and no persistent repack | Retain; profile now attributes 55% of GPU time to projection tiles and 20% to causal attention/rotary work |
| 2026-07-25 | working tree | Combining causal score, softmax, and value phases removes launch overhead without numerical reordering | Native 32-token chunks, context 512, FP8 KV, fused Gate+Up, alternating retained-path A/B, 3/10 each | Unfused: 395.52 tok/s, 1294.49 ms median TTFT | Fused: 417.68 tok/s, 1225.83 ms median TTFT (+5.6%) | Identical first token (`236772`) in all 20 A/B runs; CUDA test requires bit-identical fused/reference FP32 output across a wrapped local ring; 32/128 first tokens unchanged | No arena growth; score workspace retained for exact arithmetic | Promote fused attention; later consolidation removes the public A/B switch; Nsight shows 44 rather than 46 GPU operations per prefill layer |
| 2026-07-25 | working tree | Capturing position-independent decode work will remove CPU/WDDM launch overhead while leaving context arithmetic unchanged | Context 128, 64 generated tokens, FP8 KV, fused Gate+Up, retained-path A/B, 1/3 each | Direct launches: 20.59 tok/s; p50 48.63 ms | Partial graphs: 25.88 tok/s (+25.7%); p50 38.81 ms (-20.2%) | All six measured runs have checksum `9292451356040114682`; 8/4 smoke A/B also matches | 12,582,912 measured graph-associated device bytes; graph capture and instantiation occur before cache reset/token timing | Promote graphs; later consolidation removes the public A/B switch; Nsight reduces average operations per layer from 44 to 8 and per forward from 2,118 to 390 |
| 2026-07-25 | working tree | Moving position and context control onto the device will let ordinary greedy decode replay the complete forward pass as one graph | Context 128, 64 generated tokens, FP8 KV, fused Gate+Up, 1/3; prior committed partial-graph result compared with full-graph result | Partial graphs: 25.88 tok/s; p50 38.81 ms | Full graph: 26.95 tok/s (+4.2%); p50 37.21 ms (-4.1%); same-run direct path 20.84 tok/s | Every measured full/direct run has checksum `9292451356040114682`; wrapped-ring context-2048 smoke also matches exactly (`4411138876731454963`) | Graph-associated device bytes rise from 12,582,912 to 20,971,520; pinned/device control and all graph executables are prepared before timing | Retain full graph; Nsight records eight `cudaGraphLaunch` calls for eight decode forwards, or one host graph launch per forward |
| 2026-07-25 | working tree | Evaluating one tied-BF16 vocabulary row per warp and reducing block candidates can remove the full-logit traffic and separate greedy scan | Context 128, 64 generated tokens, FP8 KV, fused Gate+Up, full graph, 3/10 retained-path A/B | Full-logit head: 27.02 tok/s; p50 36.94 ms | Warp-row fused head: 27.22 tok/s (+0.75%); p50 36.62 ms (-0.87%) | All 20 measured runs have checksum `9292451356040114682`; fused/unfused 31-token sky and 32-token integer sequences match exactly; 31-step sky logits have max abs `7.6294e-6`, RMS `1.0358e-6`, cosine `0.9999999999999948`, Top-1 31/31, mean Top-20 overlap 20/20 | 32 KiB candidate array; graph-associated device bytes remain 20,971,520 | Promote warp-row reduction; later consolidation removes the public A/B switch; Nsight direct-launch projection reduces output-head plus argmax from about 3.315 ms to 3.031 ms (-8.6%) |
| 2026-07-23 | working tree | Direct source-layout SM120 MMA can consume the checkpoint without persistent repack | Gate `[15360,3840]`, W4A4 NVFP4, 3 warm-ups/10 iterations | CUDA scalar reference 0.2785 ms | SM120 direct 0.0334 ms | max abs `1.1920929e-7`; cosine `0.9999999999999999` | 0 persistent repack bytes | Retain direct route; continue qualification |
| 2026-07-23 | working tree | The same mapping is valid for Up | Up `[15360,3840]`, W4A4 NVFP4, 3/10 | CUDA scalar reference 0.2784 ms | SM120 direct 0.0288 ms | max abs `5.9604645e-8`; cosine `1.0` | 0 persistent repack bytes | Retain direct route |
| 2026-07-23 | working tree | The direct route also covers the transposed logical Down shape | Down `[3840,15360]`, W4A4 NVFP4, 3/10 | CUDA scalar reference 0.6604 ms | SM120 direct 0.0412 ms | max abs `0`; cosine `1.0` | 0 persistent repack bytes | Retain direct route |
| 2026-07-23 | working tree | The exact operators compose without a numerical discontinuity at Down requantization | Complete Layer-0 MLP plus residual, W4A4 NVFP4, 10 warm-ups/100 iterations | CUDA scalar-reference chain 1.2648 ms | SM120 direct chain 0.4951 ms | zero differing Down-input bytes; final max abs `0`; oracle max abs `6.7374888e-9` | 99,774,000 probe device bytes; 0 persistent repack bytes | Proceed to trusted hidden-state comparison and fusion |
| 2026-07-23 | working tree | Per-token E4M3 and per-channel BF16 scales can use source rows directly | Layer-0 Q/K/V/O, W8A8 E4M3, isolated 3/10 hot-cache characterizations | CUDA scalar reference 0.424–0.450 ms | SM120 direct 0.015–0.037 ms | max abs at most `8.9406967e-7`; cosine at least `0.999999999999978` | 0 persistent repack bytes | Retain direct route; next measure combined Q/K/V working set |
| 2026-07-23 | working tree | The exact FP8 projections compose with Gemma local-attention semantics | Layer-0 local attention, decode position 31, deterministic 32-token FP32 K/V cache | Independent CUDA scalar projection chain | Direct-source SM120 projection chain | final max abs `4.8398972e-5`; RMS `4.2503101e-6`; cosine `0.9999999999984577` | 48,577,804 owned probe device bytes; 0 persistent repack bytes | Proceed to full-attention K=V and trusted hidden-state golden; no timing claim |
| 2026-07-23 | working tree | Full attention can reuse the raw K projection for V while retaining distinct post-processing and cache states | Layer-5 full attention, decode position 31, deterministic 32-token FP32 K/V cache, proportional RoPE | Independent CUDA scalar projection chain | Direct-source SM120 projection chain | final max abs `4.5299530e-6`; RMS `5.5268314e-7`; cosine `0.9999999999999085` | 65,545,484 owned probe device bytes; 0 persistent repack bytes | Retain projection reuse, require separate final K/V cache, proceed to trusted layer golden; no timing claim |
| 2026-07-23 | working tree | Validated FP8 attention and NVFP4 MLP operators compose into one device-resident decoder layer | Complete Layer-0, decode position 31, deterministic 32-token FP32 K/V cache | Independent CUDA scalar-projection chain | Direct-source SM120 projection chain | zero differing bytes at both NVFP4 activation boundaries; final max abs `4.7683716e-6`; RMS `2.8454761e-7`; cosine `0.9999999999999643` | 148,639,086 comparison-probe device bytes; 0 persistent repack bytes | Proceed to prompt-derived trusted Layer-0 fixture before fusion or timing claims |
| 2026-07-24 | working tree | The unfused 48-layer path is ready for performance qualification | Batch-one greedy, exact-blue 20-token prompt, 2 output tokens | No prior full-engine result | 28.58 output tok/s for the single measured decode transition | Exact two-token fixture passes, but longer fixture diverges at generated step 2 | 9,200,135,680-byte weight arena; 44,040,192-byte float32 K/V cache at context 64; 1,467,904-byte workspace | Reject as benchmark evidence; diagnose full logits and prompt-derived states first |
| 2026-07-24 | working tree | The sky step-2 divergence was caused by mismatched K/V precision | Sky prompt, greedy, 3 output tokens, native and scalar projection paths, vLLM auto-FP8 and explicit BF16 controls | BF16 gem16/vLLM `[818,7217,7412]` | Checkpoint-FP8 gem16/vLLM `[818,7217,563]` | Exact token agreement within each K/V mode | Initial evidence used dequantized FP8 values in a float cache | Retain as precision-control evidence |
| 2026-07-24 | working tree | Physical one-byte FP8 cache preserves the established FP8 behavior | Sky prompt, greedy, 12 output tokens, context allocation 64 | Dequantized-FP8 float cache: 44,040,192 bytes | Physical E4M3FN cache: 11,010,048 bytes | Identical 12-token gem16 sequence; dedicated exactly-representable FP8 append/read CUDA test passes | Unfused correctness attention; not benchmark-qualified | Retain physical storage; optimize only after longer correctness gate |
| 2026-07-24 | working tree | The remaining first greedy divergence originates after the first Layer-0 MLP boundary, not the FP8 cache or direct projection path | Sky prompt position 0 and generated decision position 33; gem16 native/reference, vLLM auto-FP8, llama.cpp token fixture | vLLM and llama.cpp agree through output token 12 (`3730`) | gem16 agrees through token 11, then selects `57583` | Position 0 is bit-exact through pre-feedforward norm; Layer-0 MLP output RMS `5.793e-2`, max `2.0`, cosine `0.9999995`; native/reference bit-identical | vLLM and llama.cpp later diverge at output token 19 | Capture Gate/Up/GELU/Down boundaries and inspect the step-12 logit margin |
| 2026-07-24 | working tree | Exact vLLM GELU and NVFP4 activation arithmetic remove the prior Layer-0 MLP discrepancy but expose an earlier FP8-attention mismatch | Sky prompt, positions 0, 1, and 24; physical FP8 gem16; vLLM FP8/BF16 controls | Prior compensating revision matched 11 FP8-reference tokens | Corrected revision selects `[818,7217,7412]`; FP8-vLLM/llama.cpp select `[818,7217,563]` | Position 0 exact through Layer 29 and final state exact; position-1 Layer-0 attention context RMS `3.846e-3`, max `6.25e-2`; position-24 RMS `6.640e-3`, max `1.875e-1` | Physical FP8 cache remains one byte/value; no performance claim | Retain corrected GELU/quantization, localize FP8 cache-write and attention reduction arithmetic |
| 2026-07-24 | working tree | The single sky divergence may overstate broad model drift | 12 deterministic chats, 127 teacher-forced positions, gem16 FP8 and llama.cpp F16 compared with vLLM FP8 top-20 | Original gate covered 3 prompts/65 autoregressive positions | gem16 Top-1 118/127; llama.cpp 119/127; reference Top-1 in both Top-5 at 127/127 | Mean selected-token logprob absolute delta: gem16 `0.1064`, llama.cpp `0.1195`; mean Top-20 overlap: `14.606` and `15.646` | Diagnostic full-logit host capture only; no performance claim | Retain as broad characterization; do not adopt a tolerance yet |
| 2026-07-24 | working tree | Matching higher-precision cache modes will reduce residual distribution drift | Same 12 chats, 131 teacher-forced positions, gem16 BF16 and llama.cpp F16 compared with vLLM BF16 | gem16 FP8/vLLM FP8 Top-1 92.9%, selected-logprob delta `0.1064` | gem16 BF16/vLLM BF16 Top-1 96.9%, delta `0.0580`; llama.cpp F16/vLLM BF16 Top-1 98.5%, delta `0.0419` | Ten of 12 prompts have complete Top-1 agreement for both candidates; reference Top-1 always in Top-5 | BF16 is correctness mode; no speed claim | FP8 cache explains part but not all drift; keep FP8 default and BF16 diagnostic control |
| 2026-07-24 | working tree | Coalesced SIMT/GEMV loads may beat tensor-core MMA at batch one | Layer-0 Gate/Up/Down, direct packed source layout, 10 warm-ups/100 iterations | SM120 MMA: `0.03217`/`0.02594`/`0.04832` ms | SIMT GEMV: `0.15193`/`0.16113`/`0.15005` ms | Reference/SIMT max abs at most `5.9605e-8`; Down uses the same explicitly reported CUDA quantization bytes after a CPU boundary mismatch | One additional probe output only; 0 persistent repack bytes | Reject for runtime dispatch; MMA is 3.1x–6.2x faster |
| 2026-07-24 | working tree | Closing Gate, Up, BF16 boundaries, and GELU product into one kernel can reduce launch overhead | Real Layer-0 Gate/Up tensors, 10/100 kernel timing; alternating 32-token full-engine A/B on Windows | Unfused Gate/Up/product `0.28665` ms | Fused without diagnostic stores `0.28190` ms | CUDA test pins exact BF16 Gate/Up/product; 12-prompt FP8 suite remains 118/127 Top-1 and 127/127 Top-5 | No arena growth; five recurring launches removed | Retain only as a characterization probe; Linux end-to-end measurements later select the separate production path |
