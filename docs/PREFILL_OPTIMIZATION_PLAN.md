# Prefill optimization plan

Status: active execution goal

Target machine: Linux, NVIDIA GeForce RTX 5080 Laptop GPU, compute capability 12.0

Target workload: direct `unsloth/gemma-4-12b-it-NVFP4` loading, text-only, batch one

## Objective

Bring correct end-to-end prompt processing to the performance of the retained direct-load vLLM reference, or
exceed it, without weakening checkpoint semantics, output quality, benchmark boundaries, the 16 GB memory budget,
or the no-allocation token-loop contract. Performance changes are developed directly on `main`. A candidate is
promoted only when it wins the prescribed repeated benchmark and passes all applicable correctness gates. The
winner becomes the sole production implementation; rejected and superseded implementations do not remain as
user-selectable optimization modes.

The initial Linux reference point is commit `1bc942b`. Online attention, larger prompt chunks, projection
pipelines, boundary fusion, GQA grouping, CUTLASS FP8/NVFP4 prefill GEMMs, and global-attention staging have since
superseded that implementation. The former 512-token “current” row is intentionally removed because it predates
the CUTLASS promotions and had become misleading. The latest retained Linux long-prompt point before the current
baseline refresh is 3,704.64 tok/s at 8K with 2,211.28 ms TTFT; it is development evidence, not a parity claim.
The 2026-07-27 Linux refresh at `304a113` measures 2,575.72/4,277.03/4,318.31/3,674.54 tok/s at
`128`/`512`/`2,048`/`8,192`; short-prompt samples are clock-bimodal, while the 8K 95% interval is stable. The first
post-refresh promotion vectorizes local-attention FP8 staging and raises the same 8K median to 3,815.49 tok/s.
The bounded global-attention follow-up removes redundant modulo from contiguous-cache staging and reaches a
combined adjacent median of 3,827.47 tok/s versus its 3,801.98 tok/s parent (+0.67%). The max-power 16K refresh at
`0c0ec10` establishes 5,034.80 tok/s and 3,254.15 ms TTFT for the 2,048-token parent. Promoting an 8,192-token
checkpoint-FP8 chunk raises the adjacent 3/10 median to 5,172.75 tok/s (+2.74%) and lowers TTFT to 3,167.37 ms
(-2.67%) while retaining the output hash; peak sampled VRAM with the MTP assistant resident is 12,704 MiB. The
next promotion closes CUTLASS FP8 scaling at the already-required BF16 boundary, reaching 5,271.29 tok/s and
3,108.15 ms without changing the hash or memory plan. Asynchronous local FP8 staging then reaches 5,379.58 tok/s
and 3,045.59 ms while reducing the profiled local family by 20.34%. Cross-engine claims must reconcile timing boundaries and
cache precision under `docs/BENCHMARKING.md` and `AGENTS.md`.

## Profile-derived diagnosis

A direct Linux Nsight Systems characterization of both engines at 512 tokens gives the following approximate GPU
cost per prefill execution. The current column is the online-attention path; the initial column is the profile that
established this program:

| Phase | Initial gem16 | Current gem16 | vLLM | Current gap |
|---|---:|---:|---:|---:|
| NVFP4 MLP projections | 289.78 ms | 97.68 ms | 24.23 ms | 4.03x |
| Attention | 199.77 ms | 23.10 ms | 13.11 ms | 1.76x |
| FP8 attention projections | 131.13 ms | 64.60 ms | 27.15 ms | 2.38x |
| Other GPU work | 115.98 ms | 22.63 ms | 9.98 ms | 2.27x |
| Total GPU time | 736.66 ms | 208.01 ms | 74.47 ms | 2.79x |

The initial gem16 path launched approximately 9,235 GPU operations per execution, versus 747 for vLLM. Online
attention, larger prompt chunks, grouped FP8 projections, and exact boundary/RoPE fusion removed most of that
launch pressure. The former 65.0% projection/26.6% attention split predates the subsequent CUTLASS projection and
global-attention staging promotions. The refreshed Linux 8K trace instead attributes 21.9% to global attention, 20.9% to local
attention, 20.2% to FP8 projection GEMMs, and 17.9% to NVFP4 projection GEMMs. Vectorized local FP8 staging then
reduces local attention by 26.88% and total profiled kernel time by 4.39%. Direct contiguous indexing then reduces
global attention by 7.24%. Global attention and FP8 projection GEMMs each account for approximately 21.9% of the
latest profiled kernel time; the bounded attention-staging sprint is closed pending a fresh controlled
cross-engine comparison.

The neighboring Apache-2.0 NInfer implementation supplies useful implementation concepts, not a compatible
runtime path: shape-specific plans, BF16 Tensor-Core QK/PV, FP32 online softmax, swizzled shared-memory staging,
and pipelined K/V tiles. gem16 must adapt those concepts to direct mixed FP8/NVFP4 checkpoint storage, the hybrid
local/global Gemma attention geometry, circular local cache addressing, and its existing correctness contract.

## Ordered implementation program

### 1. Online Tensor-Core prefill attention

Status: implemented and model-qualified at `c0f42de`; the local path now pipelines raw-FP8 K/V staging in its
existing 64 KiB shared allocation, reducing its current 16K profile from 416.12 to 331.47 ms without changing the
softmax order. Global 32-key and approximate-exponential candidates failed correctness or end-to-end gates and are
absent.

Implement Gemma-specific attention without a global score matrix:

- use Tensor Cores for both QK and probability-times-V;
- retain FP32 row maxima, normalization sums, and output accumulation;
- apply exact causal and 1,024-token local-window masks from absolute positions;
- read keys and values belonging to the current chunk directly and older local positions from the circular cache,
  preventing overwrite hazards;
- support the 40 local layers (`16` query heads, `8` KV heads, dimension `256`) and eight global layers (`16`
  query heads, one KV head, dimension `512`) with shape-specific plans;
- eliminate the prompt-length-times-context score arena from the production plan;
- preserve the unfused scalar implementation as a test oracle, not a selectable production mode.

Qualification begins with the local D256 path, because it represents five sixths of the layers. It may be committed
as an internal verified milestone only when production behavior is not regressed. The phase is complete only when
both attention geometries are qualified and the online implementation is the sole production path.

### 2. Promote the largest deterministic winning prompt chunk

Status: 8,192 selected as the sole checkpoint-FP8 standard after the current max-power 16K comparison. Against the
2,048-token parent's 5,034.80 tok/s and 3,254.15 ms TTFT, the adjacent 3/10 candidate reaches 5,172.75 tok/s
(+2.74%) and 3,167.37 ms (-2.67%). Total MTP-resident workspace rises from 726,836,224 to 2,611,102,720 bytes; the complete
MTP-resident process peaks at 12,704 MiB and remains within the 15.3 GB limit. A 4,096-token screen reached
5,139.29 tok/s, while an M256 CUTLASS NVFP4 tile and a single N=30,720 Gate/Up GEMM were independently exact but
slower or neutral and were removed. Local attention reads the complete current chunk directly, then commits only
its newest 1,024-token suffix to the ring so positions one ring apart never race on a modulo address. BF16
correctness prefill remains capped at 1,024.

After the score arena is removed, measure complete-prompt chunks of `512` and `1,024` tokens against the current
context-budgeted plan. Select the fastest size that is deterministic, fits the measured arena budget across context
profiles, and passes generation/logit gates. Encode that selection as the standard plan; do not add a public chunk
or legacy-path switch. Long-context tiers may select a smaller compile-time plan only when their documented memory
geometry requires it.

### 3. Rebuild NVFP4 prefill projections around large SM120 CTA tiles

Status: implemented and qualified. The first promoted reuse step retains each packed weight/scale fragment across eight
independent `m16n8k64` token tiles (M128). Eight warps then form an M128xN64 CTA, and its two `cp.async` stages
overlap exact K64 activation transfers with the current MMA stack. Relative to `2366c03`, CTA reuse reduces
NVFP4 profile time by 51.2%; the asynchronous stage removes another 16.9%. Exact load-time scale tiling removes a
further 4.61%. The current kernel uses 128 registers and 10,240 shared bytes with zero stack/local memory. Tested
per-warp M256 and N128 extensions remain rejected for spills and an end-to-end loss respectively. A later
spill-free M256xN64 CTA made two M128 warp groups share a weight stage, but lost 3.84%/3.15% end to end at
128/512 tokens; it was removed as well.

Replace the current warp-level token tiling with a shape-specific block pipeline that:

- reuses packed E2M1 weights and E4M3 block scales across a substantially larger token tile;
- overlaps global-memory loads, source-layout preparation, and `m16n8k64` MMA work;
- consumes the source checkpoint values exactly, including local and global scaling;
- uses an exact load-time scale/layout swizzle into the final GPU allocation only if profiling proves direct source
  consumption is slower;
- never creates a persistent second weight copy and records load-time and peak-memory impact of any swizzle;
- has no local-memory spills in the selected hot kernel.

Gate, Up, and Down are measured individually and end to end. A layout transformation is accepted only when its
end-to-end benefit, exact value preservation, and memory cost are recorded in `docs/WEIGHT_LAYOUT.md`.

#### Phase-3 state after the M128xN64 asynchronous CTA promotion

The current production geometry is one warp per N8 output slice and eight M16 token tiles per retained packed
weight fragment. Eight warps cover M128xN64 and double-buffer the M128xK64 activation tile plus scales with
`cp.async`, while each warp still consumes direct-layout N8 weights and preserves the original K64 accumulation order.
Increasing the per-warp accumulator footprint to M256 is not viable because it spills.

Inspection of NInfer's Apache-2.0 Q4 row-split GEMM identifies the next transferable mechanisms: shape-specific
M64xN64/M64xN128 CTA schedules, K64 iteration, two- or three-stage `cp.async` pipelines, XOR-swizzled shared
memory, CTA-wide reuse, and optional accumulator-fragment ping-pong. Its Q4 values are decoded to BF16 shared
memory before BF16 MMA, so that numerical/storage path is not reusable for this engine's native block-scaled
E2M1 MMA and must not be copied.

The two-stage activation pipeline and exact final-allocation scale layout are qualified. N128 is rejected because
its 3/10 context-512 median loses about 2.5% despite remaining spillfree. The loader now transforms only local scale
byte order to `[row8][K64][row][4 scales]`; packed weights and FP32 accumulation order remain unchanged, and no
second device copy exists. Phase 3 is closed. Proceed to the ordered large/grouped FP8 projection phase.

### 4. Rebuild and group the FP8 attention projections

Status: implemented and model-qualified; the original M64 winner from `6005921` was superseded on 2026-07-26 by
the spill-free M128 extension. Production scaling now closes each FP8 projection at its required BF16 boundary,
removing redundant V/O round passes and raising the current 16K median from 5,172.75 to 5,271.29 tok/s.

Apply the same large-token-tile and pipeline discipline to Q, K/V, and O while preserving per-token dynamic FP8
activation quantization and per-channel weight scaling. Evaluate shape-specific combined Q/K/V scheduling and the
full-attention K-projection reuse already required by model semantics. Promote grouping only when it reduces
end-to-end time; do not retain separate grouped/ungrouped user modes.

The production kernel uses one 256-thread M128xN64xK64 CTA. Eight warps cooperate on two ping-pong stages
containing exact source-layout E4M3 activation and weight bytes; each staged weight fragment feeds eight M16 MMA
tiles. Local Q/K/V and global Q/K occupy one binding-dimension launch, while O uses the same kernel independently.
The M64 promotion originally cut context-512 FP8 projection time by 45.5%; the later M128 extension cuts a further
1.66% in adjacent 8K profiles and improves the immediate 8K end-to-end median by 2.13%. The current kernel uses
96 registers, 25,600 bytes static shared memory, and zero stack/local memory. Operator output is bit-identical to
the CUDA reference, boundary logits and all 12-prompt teacher-forced metrics are unchanged, and no alternate
production selector remains.

### 5. Fuse only profile-proven bandwidth and launch boundaries

Status: closed. Exact RMSNorm/quantization and MLP activation boundaries are the sole production prefill path.
Projection BF16 rounding, Q/K RMSNorm, RoPE, and post-RoPE BF16 rounding are also one exact standard kernel backed
by persistent max-context RoPE tables. The final post-attention K/V-write fusion was correct and locally faster,
but its `+0.79%` context-512 end-to-end result had strongly overlapping mean intervals; the complete candidate was
removed. No phase-5 variant remains. Further work returns to the dominant NVFP4 and FP8 projection pipelines.

After attention and projection kernels are no longer the old bottlenecks, profile again and consider, in order:

1. RMSNorm plus activation quantization;
2. Q/K normalization plus RoPE plus K/V write;
3. Gate/Up epilogue plus GELU-tanh product;
4. residual/norm boundaries whose intermediate values have no other consumer.

Each fusion must retain a test oracle, report numerical reordering, and improve repeated end-to-end prefill. An
isolated kernel win is insufficient, as shown by the previously rejected Gate/Up fusion.

The first accepted set preserves every observable BF16 and E4M3/E2M1 boundary while combining RMSNorm with FP8 or
NVFP4 token quantization, combining post-projection norm/residual/optional layer scale, and combining separate
Gate/Up outputs with Gemma GELU-tanh and the Down-input NVFP4 quantizer. Dedicated CUDA fixtures require
bit-identical payloads and scales versus the former sequence. Against detached `bdb1294`, 3/10 medians rise from
1,540.69/1,748.61/1,563.23 to 1,664.23/1,914.76/1,722.95 tok/s at 128/512/2,048 tokens
(`+8.0%/+9.5%/+10.2%`). Nsight reduces launches from 2,165 to 1,300 per prefill (`-40.0%`) and total GPU kernel
time by 8.73% at 512. Exact-blue, both vLLM boundary checks, and the 12-prompt teacher-forced metrics are unchanged;
peak process VRAM is 9,582 MiB with no arena growth.

The second promotion removes redundant trigonometry as well as launches. Initialization computes local D256 and
global partial-D512 cosine/sine tables with the same double-precision expressions as the former RoPE kernel and
stores their float results for every planned position. Every layer then consumes those exact values while one CTA
preserves projection BF16 rounding, the original 256-thread RMSNorm reduction tree, normalized BF16 rounding,
RoPE, and the final BF16 boundary. Local and global CUDA fixtures are bit-identical to all eight former kernels.
Against detached `ccbe4ed`, noisy 128/512 cases use 3/30 medians and improve from 1,590.17/1,895.52 to
1,831.33/2,182.51 tok/s (`+15.17%/+15.14%`) with non-overlapping mean 95% intervals; the required 2,048-token 3/10
median improves from 1,716.56 to 2,004.23 tok/s (`+16.76%`). Nsight reports 1,300 to 964 launches and 250.03 to
208.01 ms kernel time per 512-token prefill. The hot kernel uses 35 registers, 3,072 shared bytes, zero stack/local;
peak process VRAM is 9,586 MiB. All model-quality metrics remain unchanged. This left the post-attention K/V
append as the final unmeasured boundary before returning to the projection profile.

The K/V-append experiment then combined attention BF16 rounding, FP8 token quantization, and the safe
post-attention ring commit. It reduced the production launch count from 964 to 868 per 512-token prefill and the
affected GPU boundary from about 2.12 to 1.30 ms, while preserving bit-identical FP8 bytes, scales, and wrapped
cache contents. Its 3/10 end-to-end medians were 2,285.60 versus 2,303.73 tok/s, and the mean 95% intervals
overlapped strongly. Per promotion policy the entire candidate was deleted rather than retained behind a switch.
Phase 5 is therefore closed on `f76d478` and projection work resumes from that sole standard path.

A subsequent projection-grouping experiment assigned Gate and Up to independent eight-warp groups inside one
16-warp CTA and shared only the already-qualified M128 activation stage. It was arithmetically exact, used 126
registers with zero stack/local memory, and removed one launch per layer, but reduced adjacent 3/10 medians by
1.29% at 128 tokens and 3.34% at 512. The 512-thread scheduling unit outweighed the saved activation traffic, so
the complete implementation was deleted. Separate M128xN64 Gate and Up launches remain the sole standard.

The accepted follow-up instead changes only the Gate/Up epilogue representation: both projections write the
model-required BF16 boundary directly, and the fused GELU/NVFP4 quantizer consumes that BF16 storage. This removes
two FP32 round trips and the dead FP32 product arena without changing the numerical boundary. Relative to detached
`f76d478`, adjacent medians improve by 1.25%/4.07%/2.44% at 128/512/2,048 tokens; the 2,048-token confidence
intervals do not overlap. The 512-token profile reduces NVFP4 time by 4.68% and GELU time by 17.9%. At the 2,048
context the reusable workspace falls by 125,829,120 bytes to 312,593,408 bytes, with a measured 9,466 MiB process
peak. Exact-blue, both vLLM boundaries, all teacher-forced aggregates, and CTest are unchanged. Direct BF16
Gate/Up output is therefore the sole production path; no FP32 or user-selectable variant remains.

## Mandatory correctness gates

Every promoted milestone must pass:

- host unit tests and CUDA operator tests;
- exact checkpoint-format, scale, packing, cache-addressing, causal-mask, and local-window fixtures relevant to the
  changed code;
- comparison with the retained unfused/reference operator, reporting maximum absolute error, RMS error, cosine
  similarity, and row-sum/finite checks where applicable;
- exact-blue generation and vLLM Top-1/full-logit comparison at the fixed 129- and 257-token prefill boundaries;
- the committed teacher-forced suite and full-logit comparison when the changed arithmetic can affect logits;
- checks at chunk boundaries, a wrapped local cache, and at least one global-attention layer;
- unchanged tokenizer, chat template, sampling configuration, checkpoint revision, and prompt token IDs.

Tensor-Core online softmax deliberately changes floating-point reduction order, so bit identity with the serial
attention oracle is not a valid universal requirement. Any tolerance used for promotion must be derived from the
observed distribution, recorded in `tests/tolerances.yaml`, and supported by model-logit and generation evidence.
No tolerance may be relaxed only to accept a speedup.

The former eight-token boundary sequences were generated by the old gem16 path itself and are not a trustworthy
reference: direct vLLM generation differs from them. The committed boundary fixture is produced offline by vLLM
0.25.1 from the pinned checkpoint and checks the only position computed by batch prefill. Later positions belong
to the separate teacher-forced decode suite.

## Mandatory performance and resource gates

For every production promotion, collect on a thermally stable machine:

- prefill at `128`, `512`, and `2,048` prompt tokens, batch one;
- three warm-up and ten measured runs, retaining raw samples and reporting median, mean, standard deviation, and a
  95% confidence interval;
- prompt tokens/s and TTFT with the same timing boundaries before and after;
- Nsight Systems path confirmation, phase/kernel time, and launch count;
- register count, stack frame, spill stores/loads, and local-memory use for changed hot kernels;
- peak and steady-state VRAM plus named arena changes;
- confirmation of no recurring allocation or filesystem access in the execution loop;
- deterministic output checksums across repeated runs.

The main comparison is against a separately built parent commit, executed immediately adjacent to the candidate
under the same clocks and thermal conditions. The current direct vLLM profile is retained as an optimization
target. A parity or superiority claim requires a freshly controlled cross-engine run with identical prompt IDs,
cache precision where supported, warm-up policy, and explicitly reconciled timing boundaries.

## Promotion, commits, and rollback

Work proceeds directly on `main`. Stable intermediate commits are expected after a self-contained milestone has
passed its applicable correctness gates and either:

- improves the complete prefill benchmark with statistical support; or
- adds a necessary internal implementation/test foundation without changing or slowing the production path.

Before each performance promotion, retain benchmark and profile evidence under
`benchmarks/results/<date>/<git-sha>/<machine-id>/`, update `docs/PERFORMANCE_LEDGER.md`, build and run the full
available test suite, then commit and push to `origin/main`. A losing candidate is removed rather than hidden behind
an option. If a newly promoted path later fails a broader correctness or performance gate, fix it immediately or
revert the complete promotion with an explicit ledger entry.

## Completion criteria

This goal is complete only when all of the following hold:

1. Online Tensor-Core attention is standard for local and global prefill and no global score matrix is allocated.
2. The best qualified prompt chunk is the deterministic standard.
3. NVFP4 and FP8 prefill projections use the qualified large-tile plans.
4. Remaining fusions are exhausted based on current profiles, not assumed benefit.
5. The mandatory correctness, quality, memory, Nsight, and 3/10 benchmark evidence is retained.
6. Controlled batch-one prefill reaches or exceeds the current direct-load vLLM reference across the required
   `128`, `512`, and `2,048` points, or a documented hardware/resource lower bound demonstrates the remaining gap
   and the project records an explicit decision before changing the objective.
