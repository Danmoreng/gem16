# Performance ledger

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
