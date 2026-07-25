# Performance ledger

Kernel characterizations below are development evidence, not accepted end-to-end benchmark claims. They use one
deterministic activation and the pinned Layer-0 checkpoint tensors on the current Windows Blackwell machine.
Repeated isolated projection measurements keep one 33.3 MB tensor family hot in cache; do not add their times to
estimate a layer. The complete MLP row cycles through the 99.5 MB three-projection working set and is the more useful
decode characterization.

## Native prefill and long-context decode characterization

The active work program, required gates, and promotion policy are fixed in
[`docs/PREFILL_OPTIMIZATION_PLAN.md`](PREFILL_OPTIMIZATION_PLAN.md). At `1bc942b`, the Linux 512-token median is
698.25 tok/s versus the retained vLLM orientation point of 6,146.50 tok/s. A direct Nsight comparison attributes
approximately 289.78/199.77/131.13 ms of gem16gb GPU time per execution to NVFP4 projections, attention, and FP8
projections, versus 24.23/13.11/27.15 ms for vLLM. The plan therefore prioritizes online Tensor-Core attention,
then larger deterministic prompt chunks, large pipelined NVFP4 CTA tiles, large/grouped FP8 projection tiles, and
only then residual profile-proven fusions. These values identify work; they are not parity benchmark claims.

The first consolidated Linux run at base commit `960528d` fixes the production plan to native SM120 projections,
chunked/fused prefill, separate Gate/Up/GELU, complete decode graphs, and fused output reduction. It uses checkpoint
FP8 KV and the standard 3 warm-up/10 measured policy. These ratios are orientation only: retained llama.cpp and
vLLM artifacts use BF16 KV, llama.cpp maps attention weights to BF16, and prefill timing boundaries differ.

| Workload | gem16gb median | llama.cpp | vLLM | gem/llama | gem/vLLM |
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

The following console characterizations were collected on the same Windows Blackwell development machine after
commits `0d2065e` and `914aba1`, using direct checkpoint loading, checkpoint FP8 KV, native SM120 projections, and
the opt-in fused Gate/Up path. They are not accepted benchmark artifacts: 128 and 512 prefill use the full 3 warm-up/
10 measured policy, while the expensive 2K and 8K scaling points deliberately use fewer repetitions. The existing
llama.cpp and vLLM values come from their separately retained development runs, so ratios are orientation rather
than parity claims.

| Prompt | gem16gb prefill tok/s | Repetitions | llama.cpp | vLLM | gem/llama | gem/vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 87.72 | 3/10 | 2,215 | 4,679 | 0.040x | 0.019x |
| 512 | 82.42 | 3/10 | 2,628 | 6,146 | 0.031x | 0.013x |
| 2,048 | 69.92 | 1/3 | 2,539 | 4,913 | 0.028x | 0.014x |
| 8,192 | 55.83 | 1/1 | 2,362 | 3,929 | 0.024x | 0.014x |

| Existing context | gem16gb decode tok/s | gem p50/p95/p99 ms | Repetitions | llama.cpp | vLLM | gem/llama | gem/vLLM |
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
| 2026-07-24 | working tree | The sky step-2 divergence was caused by mismatched K/V precision | Sky prompt, greedy, 3 output tokens, native and scalar projection paths, vLLM auto-FP8 and explicit BF16 controls | BF16 gem16gb/vLLM `[818,7217,7412]` | Checkpoint-FP8 gem16gb/vLLM `[818,7217,563]` | Exact token agreement within each K/V mode | Initial evidence used dequantized FP8 values in a float cache | Retain as precision-control evidence |
| 2026-07-24 | working tree | Physical one-byte FP8 cache preserves the established FP8 behavior | Sky prompt, greedy, 12 output tokens, context allocation 64 | Dequantized-FP8 float cache: 44,040,192 bytes | Physical E4M3FN cache: 11,010,048 bytes | Identical 12-token gem16gb sequence; dedicated exactly-representable FP8 append/read CUDA test passes | Unfused correctness attention; not benchmark-qualified | Retain physical storage; optimize only after longer correctness gate |
| 2026-07-24 | working tree | The remaining first greedy divergence originates after the first Layer-0 MLP boundary, not the FP8 cache or direct projection path | Sky prompt position 0 and generated decision position 33; gem16gb native/reference, vLLM auto-FP8, llama.cpp token fixture | vLLM and llama.cpp agree through output token 12 (`3730`) | gem16gb agrees through token 11, then selects `57583` | Position 0 is bit-exact through pre-feedforward norm; Layer-0 MLP output RMS `5.793e-2`, max `2.0`, cosine `0.9999995`; native/reference bit-identical | vLLM and llama.cpp later diverge at output token 19 | Capture Gate/Up/GELU/Down boundaries and inspect the step-12 logit margin |
| 2026-07-24 | working tree | Exact vLLM GELU and NVFP4 activation arithmetic remove the prior Layer-0 MLP discrepancy but expose an earlier FP8-attention mismatch | Sky prompt, positions 0, 1, and 24; physical FP8 gem16gb; vLLM FP8/BF16 controls | Prior compensating revision matched 11 FP8-reference tokens | Corrected revision selects `[818,7217,7412]`; FP8-vLLM/llama.cpp select `[818,7217,563]` | Position 0 exact through Layer 29 and final state exact; position-1 Layer-0 attention context RMS `3.846e-3`, max `6.25e-2`; position-24 RMS `6.640e-3`, max `1.875e-1` | Physical FP8 cache remains one byte/value; no performance claim | Retain corrected GELU/quantization, localize FP8 cache-write and attention reduction arithmetic |
| 2026-07-24 | working tree | The single sky divergence may overstate broad model drift | 12 deterministic chats, 127 teacher-forced positions, gem16gb FP8 and llama.cpp F16 compared with vLLM FP8 top-20 | Original gate covered 3 prompts/65 autoregressive positions | gem16gb Top-1 118/127; llama.cpp 119/127; reference Top-1 in both Top-5 at 127/127 | Mean selected-token logprob absolute delta: gem16gb `0.1064`, llama.cpp `0.1195`; mean Top-20 overlap: `14.606` and `15.646` | Diagnostic full-logit host capture only; no performance claim | Retain as broad characterization; do not adopt a tolerance yet |
| 2026-07-24 | working tree | Matching higher-precision cache modes will reduce residual distribution drift | Same 12 chats, 131 teacher-forced positions, gem16gb BF16 and llama.cpp F16 compared with vLLM BF16 | gem16gb FP8/vLLM FP8 Top-1 92.9%, selected-logprob delta `0.1064` | gem16gb BF16/vLLM BF16 Top-1 96.9%, delta `0.0580`; llama.cpp F16/vLLM BF16 Top-1 98.5%, delta `0.0419` | Ten of 12 prompts have complete Top-1 agreement for both candidates; reference Top-1 always in Top-5 | BF16 is correctness mode; no speed claim | FP8 cache explains part but not all drift; keep FP8 default and BF16 diagnostic control |
| 2026-07-24 | working tree | Coalesced SIMT/GEMV loads may beat tensor-core MMA at batch one | Layer-0 Gate/Up/Down, direct packed source layout, 10 warm-ups/100 iterations | SM120 MMA: `0.03217`/`0.02594`/`0.04832` ms | SIMT GEMV: `0.15193`/`0.16113`/`0.15005` ms | Reference/SIMT max abs at most `5.9605e-8`; Down uses the same explicitly reported CUDA quantization bytes after a CPU boundary mismatch | One additional probe output only; 0 persistent repack bytes | Reject for runtime dispatch; MMA is 3.1x–6.2x faster |
| 2026-07-24 | working tree | Closing Gate, Up, BF16 boundaries, and GELU product into one kernel can reduce launch overhead | Real Layer-0 Gate/Up tensors, 10/100 kernel timing; alternating 32-token full-engine A/B on Windows | Unfused Gate/Up/product `0.28665` ms | Fused without diagnostic stores `0.28190` ms | CUDA test pins exact BF16 Gate/Up/product; 12-prompt FP8 suite remains 118/127 Top-1 and 127/127 Top-5 | No arena growth; five recurring launches removed | Retain only as a characterization probe; Linux end-to-end measurements later select the separate production path |
