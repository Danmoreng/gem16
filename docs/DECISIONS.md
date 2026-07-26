# Decisions

## 2026-07-26: Use CUTLASS SM120 for FP8 prefill projections

Date: 2026-07-26
Decision: Run prompt Q, K, optional V, and O projections through CUTLASS 4.5.2 SM120 128x128x64
warp-specialized FP8 GEMMs with FP32 accumulation/output. Consume checkpoint `[N,K]` weight bytes directly as
CUTLASS column-major B, then apply each token's FP32 activation scale and each output channel's BF16 weight scale
in a separate device kernel using the original left-to-right multiplication order. Retain the grouped native
direct-source path for token-at-a-time decode.
Context: After CUTLASS NVFP4 covered the full MLP, FP8 Q/K/V/O was the largest remaining projection family.
The source-layout M128xN64xK64 kernel improved substantially over the original direct path, but still managed
tiling, staging, and scheduling manually. The checkpoint's FP8 weight ordering already matches a regular GEMM B
operand, so CUTLASS requires no per-layer transform.
Alternatives: Continue tuning the native CTA; keep grouped Q/K/V in one launch; fuse per-row/per-column scaling
into a custom CUTLASS epilogue; or retain a raw unscaled FP32 buffer. The first left confirmed end-to-end
throughput unused, the grouped launch was slower than independently scheduled GEMMs, and epilogue fusion is
deferred until it proves a further gain without changing scaling order. The existing projection buffers hold the
raw accumulators transiently, so no new arena is necessary.
Consequences: Runtime metadata reports `cutlass_m128n128k64`, `cutlass_auto`, and no grouped prefill Q/K/V.
Persistent weights, KV storage, reusable workspace (673,808,384 bytes at 8K), and
`persistent_repack_bytes=0` remain unchanged.
Evidence: A real 128x4,096x3,840 fixture is bit-exact across 524,288 FP32 outputs versus the prior kernel. Under
3 warm-ups and 10 measured 8K runs, median prefill improves from 2,910.53 to 3,539.55 tok/s (+21.61%) and TTFT
from 2,814.61 to 2,314.42 ms (-17.77%), with a 95% throughput CI of `[3,528.28, 3,543.69]`. CUDA tests,
exact-blue, vLLM boundary Top-1 at 129/257, and teacher-forced 121/127 Top-1 plus 127/127 Top-5/Top-20 pass.
Three 8K decode runs retain one checksum and a 33.073 tok/s median.

## 2026-07-26: Extend CUTLASS block-scaled GEMM to Down prefill

Date: 2026-07-26
Decision: Run Down through the same CUTLASS 4.5.2 SM120 128x128x128 block-scaled GEMM used by Gate and Up.
Interleave the newly quantized Down-input scales for its 15,360-element contracting dimension, transform the
active Down weight into the existing preallocated CUTLASS scratch, write BF16 directly into the projection
buffer, and consume that buffer in the fused residual/RMSNorm boundary. Keep the native Row8/K64 path for
token-at-a-time decode.
Context: After promoting Gate/Up, adjacent Nsight attribution placed native Down at 16.1% of 8K profiled kernel
time, versus 7.8% for both CUTLASS Gate and Up together. The real Down shape therefore had enough work to amortize
the in-arena layout conversion, while the decode-optimal persistent layout still could not be replaced globally.
Alternatives: Retain native Down; keep a persistent second CUTLASS layout; or convert Down back through FP32 before
the next boundary. The first leaves a measured bottleneck, the second violates the single-copy contract, and the
third adds traffic with no numerical benefit.
Consequences: The weight, weight-scale, and CUTLASS workspace scratch already fit Down. Only the padded
activation-scale view grows by 1,474,560 bytes, taking the 8K reusable workspace from 672,333,824 to 673,808,384
bytes. Persistent weights remain 9,200,135,680 bytes, `persistent_repack_bytes` remains zero, and decode is
unchanged.
Evidence: The real 128x3,840x15,360 Down fixture has zero BF16 mismatches across 491,520 outputs. Under 3 warm-ups
and 10 measured 8K runs, the adjacent median improves from 2,594.28 to 2,910.53 tok/s (+12.19%) and TTFT from
3,157.72 to 2,814.61 ms (-10.87%), with a 95% throughput CI of `[2,904.41, 2,915.00]`. CUDA tests, exact-blue,
vLLM boundary Top-1 at 129/257, and teacher-forced 121/127 Top-1 plus 127/127 Top-5/Top-20 pass. Three 8K decode
runs retain one checksum and a 32.661 tok/s median.

## 2026-07-26: Use CUTLASS SM120 block-scaled GEMM for Gate/Up prefill

Date: 2026-07-26
Decision: Run the large Gate and Up prompt projections through the pinned CUTLASS 4.5.2 SM120
128x128x128 block-scaled persistent GEMM with BF16 output. Interleave activation scales once per layer and
transform one active projection at a time from the sole persistent Row8/K64 allocation into preallocated prompt
scratch. Reuse that scratch for Up immediately after Gate. Retain the native Row8/K64 path for Down and all
token-at-a-time decode.
Context: After grouped online attention, 2K chunks, and M128 FP8 projections, projections accounted for 65% of
the 8K prefill profile. Isolated CUTLASS runs at the real Gate/Up and Down geometries showed substantially more
headroom than further raster-order or L2-persistence changes to the native CTA. The persistent Row8/K64 layout is
still the measured decode winner, so replacing it globally would trade away a higher-priority path.
Alternatives: Retain the native prefill CTA; keep a second persistent CUTLASS weight layout; transform every
projection including Down; or restore row-major weights globally. The first leaves confirmed prompt throughput
unused, the second violates the single-copy memory contract, Down did not yet have an end-to-end promotion result,
and the fourth regresses the qualified decode layout.
Consequences: Gate and Up each pay an in-timing device layout conversion but use a TMA/warp-specialized
block-scaled Tensor-Core GEMM afterward. One 29,491,200-byte weight scratch, 3,686,400-byte weight-scale scratch,
padded activation scales, and an 8 MiB CUTLASS workspace increase the 8K reusable workspace from 630,276,096 to
672,333,824 bytes. The 9,200,135,680-byte persistent weight arena, KV cache, decode implementation, checkpoint
bytes, and `persistent_repack_bytes=0` remain unchanged. Runtime JSON records the distinct Gate/Up and Down plans.
Evidence: Under 3 warm-ups and 10 measured runs, 8K median prefill improves from 2,135.93 to 2,584.77 tok/s
(+21.0%) and TTFT from 3,835.33 to 3,169.34 ms (-17.4%); 2K improves from approximately 2,428 to 2,984.77 tok/s
(+22.9%). A real-geometry 2,048x128x3,840 CUDA fixture has zero differing BF16 outputs against the native kernel.
CTest, exact-blue `[9503,106]`, vLLM boundary Top-1 at 129/257, and teacher-forced 118/127 Top-1,
126/127 Top-5, and 127/127 Top-20 pass unchanged. Five 8K decode runs remain internally deterministic and median
decode throughput changes by -0.63% in a short regression run, consistent with an untouched decode path and
measurement noise.

## 2026-07-26: Fence shared online-decode reduction results before reuse

Date: 2026-07-26
Decision: In both online-decode block reductions, copy the final shared maximum or sum into a thread-local register,
then execute a second block barrier before returning. Keep the selected split/merge algorithm, FP8 KV format,
fixed graph geometry, and accumulation order unchanged. Add repeated local/global/16K operator output checks and a
fresh-process 512+256 greedy determinism gate.
Context: The shared 16K Wikipedia benchmark produced ten different nominally greedy gem16gb outputs while vLLM
and llama.cpp were stable. A 512+256 prompt was stable, but a 16K+64 reproducer diverged as early as output step 22.
Prefill logits were bit-identical and the controlled reference-attention path was deterministic, isolating the
problem to online decode attention. Racecheck then reported Read/Write hazards in both Split and Merge.
Alternatives: Accept numerically plausible greedy drift; disable CUDA Graphs; clear the partial workspace on every
layer; or fall back to the score-matrix attention path. Drift violates the correctness contract, the direct
controlled path reproduced the issue without graphs, workspace clearing did not help, and the reference path
would discard the promoted long-context performance.
Consequences: Two additional `__syncthreads()` operations protect each shared reduction result before the same
array is reused. There is no allocation, format, kernel-grid, cache, or arithmetic-order change. The new tool can
force full-length generation without EOS and exits non-zero when `--require-deterministic` observes multiple
hashes.
Evidence: Targeted Racecheck changes from reported Split/Merge hazards and corrupted instrumented output to zero
hazards. Repeated operator outputs are bit-identical; the 16K global fixture retains maximum absolute error
`1.04308e-7`, RMS `2.19933e-8`, and cosine 1.0. Five fresh-process 16K+64 production-graph runs share hash
`0b373ccd...e43d52`; five 512+256 runs share `8cc1cc48...6fce2`. CTest and exact-blue `[9503,106]` pass.

## 2026-07-26: Group prefill GQA heads, widen FP8 M, and use 2K FP8 chunks

Date: 2026-07-26
Decision: Make each local prefill-attention CTA process the two query heads sharing one KV head and each global
CTA process four query heads sharing the sole KV head. Stage K/V once for that group while retaining independent
per-head FP32 online-softmax state and the existing MMA order. Widen the two-stage FP8 projection CTA from
M64xN64xK64 to M128xN64xK64 so each weight fragment serves eight M16 tiles. Use 2,048-token chunks for
checkpoint-FP8 prefill; after local attention, commit only the newest 1,024-token suffix to its ring. Keep BF16
correctness prefill capped at 1,024 tokens.
Context: The 8K Nsight baseline attributed 52.6% of GPU time to attention. Its one-head CTAs redundantly staged
the same local K/V twice and the same global K/V sixteen times. After grouping, projections became the dominant
cost, while a 1,024-token chunk still repeated every layer launch group eight times at 8K.
Alternatives: Group two instead of four global heads; leave local heads independent; retain M64 FP8 projection;
or make a local chunk wider than its ring and commit all positions modulo the ring. Two global heads left measured
performance unused. Independent local heads repeat K/V traffic. M64 loses the adjacent A/B. Committing more than
one ring concurrently creates modulo-aliasing writes and is rejected.
Consequences: Local/global attention CTAs use 128/256 threads and group 2/4 query heads. Their static shared-memory
allocations including toolchain overhead are 66,560/99,328 bytes; both use 254 registers with zero stack/local
memory. The FP8 projection uses 96 registers and 25,600 bytes shared with zero stack/local memory. The 8K prefill
workspace grows from 322,457,600 to 630,276,096 bytes, remaining below the 1 GiB activation-arena target. No
persistent allocation or checkpoint representation changes. Runtime JSON and validators require the M128 tile,
head grouping, and 2K checkpoint-FP8 chunk.
Evidence: Under 3 warm-ups and 10 measured runs, 8K reaches 2,138.50 tok/s with 95% CI
`[2,134.61, 2,144.24]` and 3,830.72 ms median TTFT. The preceding 1K plan measured 2,107.04 tok/s under the same
policy; the Row8/K64 starting point was 1,560.23 tok/s and 5,250.50 ms TTFT. Final improvement over that starting
point is +37.1% throughput and -27.0% TTFT. Nsight reduces global/local attention from 4.440/1.242 seconds to
1.207/0.892 seconds across two 8K prefills; projections are now 65.0% of kernel time. CTest, exact-blue
`[9503,106]`, vLLM boundary rank 1 at 129/257, and teacher-forced 118/127 Top-1 plus 126/127 Top-5 all pass.
The 8K decode regression retains checksum `17504476492555856403` in all runs and reaches 33.676 tok/s median.

## 2026-07-26: Tile packed NVFP4 weights into the sole final SM120 allocation

Date: 2026-07-26
Decision: Transform every manifest-classified `NVFP4_PACKED` tensor at load time from checkpoint row-major order
to `[row tile 8][K64 block][row][32 packed E2M1 bytes]`. Use the existing matching Row8/K64 scale order for both
T=1 and batch SM120 kernels. Stream bounded transformed windows directly into the final arena, retain no raw GPU
copy, expose no runtime selector, and preserve source order only in reference/SIMT probes.
Context: After attention and grouped Q/K/V improvements, NVFP4 Gate/Up/Down still consumed about 10.56 ms of an
8K decode forward. Each output warp loaded eight rows whose K64 fragments were separated by a full source-row
stride even though the corresponding scale words had already been tiled contiguously.
Alternatives: Keep source-row weights; retain source plus a decode-only tiled copy; tile only Gate/Up; or perform
layout conversion inside each kernel. These respectively leave measured decode speed unused, violate the 16 GB
single-copy contract, split the MLP layout, or repeat address/data movement in the hot path.
Consequences: Every packed nibble, E4M3 scale byte, global divisor, and MMA accumulation order is unchanged.
Decode and prefill share one runtime layout. A reusable host staging vector is bounded at 4 MiB, the persistent
weight arena stays 9,200,135,680 bytes, and `persistent_repack_bytes` remains zero. Model-load timing includes the
CPU transformation and direct-to-final-allocation transfers.
Evidence: Host mapping tests cover K blocks, row tiles, and tail rows. CUDA native projections match the
source-layout reference, and the complete Layer-0 MLP has maximum absolute difference 0 and cosine 1. Exact-blue
remains `[9503,106]`; 129/257-token prefill retains vLLM Top-1 rank 1; the teacher-forced suite remains 118/127
Top-1 and 126/127 Top-5; and CTest passes. FP8-KV 8K decode improves from 31.604 to 33.143 tok/s median (+4.87%)
under the same 1-warm-up/3-run policy.

## 2026-07-26: Split and merge long-context FP8 decode attention

Date: 2026-07-26
Decision: For checkpoint-FP8 plans above 512 positions, replace the materialized score matrix and separate
softmax/value kernels with shape-specific split attention inside the complete decode graph. Group the two local
queries per KV head and four global queries at a time, compute normalized partial outputs with FP32 log-sum-exp
state, and merge token splits in a second kernel. Use 256-token local splits and 512-token global splits. Retain the
prior score/softmax/value path for plans through 512 positions and for explicit BF16 K/V correctness mode.
Context: The prior controlled decode path serially reduced D256/D512 QK inside one thread, wrote every score to
global memory, reread it for softmax, and scanned the cache again for PV. Its cost grew sharply at 8K even though
vLLM remained nearly context-flat. A single CTA per query with token-serial online softmax would remove score
traffic but expose too little parallelism, especially for the eight global layers.
Alternatives: Use one CTA per query; retain separate score and value kernels while only parallelizing QK; use one
fixed split size for both Gemma geometries; or select a plan dynamically in the token loop. These respectively
underfill the GPU, retain avoidable traffic, ignore distinct D256/D512 work, or weaken deterministic CUDA Graph
execution.
Consequences: Decode workspace stores normalized split outputs and LSE values rather than only scores, but remains
approximately the previous global score size plus small LSE storage. Kernel grids and addresses are fixed during
graph capture; there are no token-loop allocations or host decisions. JSON reports `fp8_online_split_gqa` versus
`score_softmax_value_reference`. The remaining 8K gap is now primarily projection and output-head work.
Evidence: Product-shape CUDA comparisons have maximum absolute error `3.73e-8` local and `1.86e-7` global with
cosine 1.0. All host/CUDA tests and exact-blue generation pass. On the Linux RTX 5080 Laptop, context-8K decode is
30.02 tok/s with 33.21/35.19/36.65 ms p50/p95/p99 over 3 warm-ups and 10 measured runs; all runs share checksum
`17504476492555856403`. The node-level Nsight trace measures about 5.1 ms/token for all 48 decode-attention layers
and merges. The short-path regression check is 32.13 tok/s at context 128.

## 2026-07-26: Compose Unsloth weights with Google's current tokenizer metadata

Date: 2026-07-26
Decision: Keep every weight, quantization, vocabulary, generation, and chat-template artifact at the locked
Unsloth revision, but source `tokenizer_config.json` from official Google Gemma 4 12B IT commit
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. Represent this as a per-file immutable source in lock schema v2.
Parse and validate Google's token roles and response template at engine startup, use its thinking/content
delimiters for visible response extraction, and retain `generation_config.json` as the authoritative stop-ID list.
Context: The Unsloth snapshot predates Google's response template and aliases `eos_token` to the turn-end token.
Google now distinguishes `<eos>` from `<turn|>` and publishes structured response boundaries. Merely copying the
new file into one local checkpoint would make provenance irreproducible, while merely locking it without reading
it would leave engine behavior unchanged.
Alternatives: Continue using Unsloth metadata; follow Google's `main`; silently overlay a bundled runtime file; or
replace the complete checkpoint with Google's BF16 source. Those choices respectively retain stale semantics,
lose reproducibility, hide the effective checkpoint composition, or discard the selected NVFP4/FP8 weights.
Consequences: `tools/fetch_model.py` resolves file-specific sources and resumes only partial files whose URL, size,
and digest identity matches. A downloaded checkpoint directory remains directly loadable and contains one
canonical `tokenizer_config.json`. Google's approximately 1e30 tokenizer-length sentinel is accepted as generic
JSON metadata but never replaces the 262,144-position model contract. Generated tool calls still fail visibly
until the phase-one runtime deliberately supports them.
Evidence: The official file is locked at 3,089 bytes with SHA-256
`a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9`. Host C++ tests cover the large JSON number,
Google schema validation, response extraction, and rejection of the old EOS alias. Python tests cover per-file
source selection, safe replacement, and identity-bound resume. The fully materialized local snapshot passes lock
verification, `gem16gb-inspect --validate`, and native chat render-only loading.

## 2026-07-26: Make interactive chat a resident exact-token session

Date: 2026-07-26
Decision: Create one engine for the lifetime of `gem16gb-chat`, retain its weights, arenas, CUDA Graphs, and hybrid
KV cache across turns, and batch-prefill only tokens appended after the materialized cache prefix. Preserve the
original generated token IDs at the CLI boundary; do not reconstruct prior assistant output through text
decode/re-encode. Require every submitted prompt to extend the cached token prefix exactly and fail visibly on a
mismatch. This is the sole interactive path and has no cache/reload selector.
Context: The former loop called `RunGreedyInference` for every user message. It reloaded roughly 9 GB of weights,
cleared the cache, rendered the entire history, and recomputed every prior token. A first resident prototype proved
why text is not a valid cache identity: decoded `blau` re-encoded without the generated channel tokens and failed
the exact prefix gate. Continuing from the actual autoregressive token sequence is both cheaper and faithful to
the state that produced the answer.
Alternatives: Reload and prefill the complete conversation; compare only decoded text; silently reset when token
prefixes differ; or add a user-selectable session mode. Full replay discards the available KV state, decoded text
does not uniquely identify BPE tokens, silent reset hides a performance and semantic change, and a selector would
retain an inferior interactive path.
Consequences: Initial prompt processing is unchanged. Each later turn pre-fills the preserved final assistant token
when a turn ended at the length limit, followed by the newly tokenized exact Gemma turn delimiter, user content,
and generation header. Existing conversation K/V remains in the local rings and global contiguous cache. The
session pre-reserves host token bookkeeping through `--max-context`; generation retains the no-allocation token
loop. A failed inference poisons the session because partially written KV state cannot be rolled back safely.
Evidence: The release and host builds pass, CTest passes both host and CUDA suites, and a real resident GPU smoke
test completed three dependent turns (`blau`, recall `blau`, translate `blue`) after a single model load. Both
continuations passed the exact token-prefix check and returned without another weight load or full-history prefill.
A separate two-turn run with `--max-tokens 1` validates the pending, not-yet-materialized final assistant token;
a two-turn thinking-template run validates the same prefix continuation with generated channel tokens.

## 2026-07-25: Precompute exact RoPE and fuse the full Q/K normalization boundary

Date: 2026-07-25
Decision: Generate local D256 and proportional global D512 cosine/sine tables once during engine initialization for
every position in the planned context. Make one CTA per token/head preserve projection BF16 rounding, the original
256-thread RMSNorm reduction, normalized BF16 rounding, RoPE, and post-RoPE BF16 rounding. Group Q and K blocks in
one launch without sharing reduction state. Make this the sole prefill path and expose no selector.
Context: After the first boundary promotion, the old RoPE kernels still consumed 50.58 ms per 512-token prefill
and repeated identical double-precision `pow`/`cos`/`sin` work for every head in every layer. Together with Q/K
rounding and normalization they required eight launches per layer.
Alternatives: Merely group Q/K launches; share trigonometry inside each layer; use approximate float intrinsics;
retain runtime variants; or store no table. The first fused prototype was exact but only won long prompts and kept
the expensive trig stackframe in the hot kernel. An earlier approximate sharing probe changed logits. Persistent
tables preserve the exact float values, remove hot-path trigonometry, and cost only 1,536 bytes per planned token.
Consequences: Prefill executes one 35-register, 3,072-byte-shared, zero-stack/local kernel instead of eight launches
per layer. The initialization-only table kernel retains the exact double expressions and is included in model-load
time. At context 2,048 the workspace grows by 3,147,264 bytes and measured process peak grows 4 MiB to 9,586 MiB.
Runtime JSON and validators require the fused kernel and `precomputed_exact_max_context` table.
Evidence: Against detached `ccbe4ed`, adjacent 3/30 medians improve 15.17%/15.14% at 128/512 with non-overlapping
mean 95% intervals; 2,048 improves 16.76% in the required 3/10 run. Nsight reduces launches from 1,300 to 964 and
GPU kernel time from 250.03 to 208.01 ms per 512-token prefill. Local/global CUDA outputs are bit-identical to the
eight-kernel oracle. CTest, exact-blue, vLLM boundaries 129/257, and every aggregate metric in the
12-prompt/127-position teacher-forced suite remain unchanged.

## 2026-07-25: Fuse exact prefill normalization and MLP quantization boundaries

Date: 2026-07-25
Decision: Replace the production prefill sequences at RMSNorm/FP8 quantization, RMSNorm/NVFP4 quantization,
post-projection norm/residual/optional layer scale, and Gate/Up/GELU-tanh/NVFP4 quantization with shape-specific
fused kernels. Preserve every prescribed BF16 cast and quantized payload exactly. Make the fused implementation
the sole production path; retain the former sequence only as a CUDA test oracle and expose no selector.
Context: After the projection phases, context-512 prefill still launched 2,165 kernels. Repeated standalone
rounding, RMSNorm, GELU, and quantization kernels dominated launch-heavy residual work even though their producer
and consumer share the same token geometry.
Alternatives: Keep the separate sequence; combine Gate and Up projections; share Q/K RoPE trigonometric arithmetic;
or retain runtime A/B modes. Separate launches leave a measured end-to-end gain unused. Combined projections had
lost the prior Linux A/B. Sharing RoPE arithmetic was re-tested and rejected because it changed boundary logits.
Runtime variants violate the single-winner policy.
Consequences: Gate and Up projections remain separate, but their exact BF16 outputs feed one GELU/product/NVFP4
kernel. Normalization fusions retain BF16 rounding before residual and optional layer scaling. Runtime JSON and
validators require all four fused families. Launches fall to 1,300 per context-512 prefill; arenas and persistent
checkpoint storage are unchanged. The selected kernels use at most 40 registers and 3,072 bytes shared memory and
have zero stack/local memory.
Evidence: Against detached `bdb1294`, final 3/10 medians improve from 1,540.69/1,748.61/1,563.23 to
1,664.23/1,914.76/1,722.95 tok/s at 128/512/2,048 (`+8.0%/+9.5%/+10.2%`). Nsight measures 40.0% fewer launches
and 8.73% less total kernel time at 512. CTest, exact-blue, vLLM boundaries 129/257, and all aggregate metrics in
the 12-prompt/127-position teacher-forced suite remain unchanged. Peak process VRAM is 9,582 MiB.

## 2026-07-25: Pipeline source-layout FP8 prefill and group Q/K/V

Date: 2026-07-25
Decision: Replace the 32-token-per-warp FP8 batch projection with one 256-thread M64xN64xK64 CTA. Double-buffer
exact source-layout activation and weight bytes in shared memory with `cp.async`, reuse every weight fragment over
four M16 MMA tiles, and group local Q/K/V or global Q/K through one binding-dimension launch. Make this the only
FP8 prefill projection path; keep the latency-oriented T=1 decode kernel separate and expose no selector.
Context: At context 512, the prior direct matrix kernel consumed 114.09 ms per prefill and launched 184 times.
Increasing per-warp M reuse alone was neutral or regressive because it did not address redundant, weakly
coalesced operand traffic across warps. The CTA tile lets eight warps share staged operands and overlap the next
K64 slice, while grouping removes otherwise independent launches without changing any projection arithmetic.
Alternatives: Retain two M16 tiles per warp; use four/eight tiles without CTA staging; preserve grouped and
ungrouped runtime variants; repack FP8 checkpoint weights. The M64/M128 warp-only candidates lost at short context,
parallel variants violate the single-winner policy, and persistent repacking is unnecessary.
Consequences: FP8 checkpoint bytes, per-token scales, per-channel BF16 scales, and FP32 K order are unchanged.
The kernel uses 60 registers and 17,408 bytes static shared memory with zero stack/local memory. FP8 launches fall
from 184 to 96 per context-512 prefill; no persistent allocation, arena size, or token-loop allocation changes.
Runtime JSON and validators require `m64n64k64`, two pipeline stages, and grouped Q/K/V.
Evidence: Against exact commit `6005921`, final 3/10 medians rise from 1,348.97/1,460.83/1,358.11 to
1,583.23/1,769.04/1,562.05 tok/s at 128/512/2,048 (`+17.4%/+21.1%/+15.0%`). Nsight reduces FP8 projection time
from 114.09 to 62.21 ms per context-512 prefill (`-45.5%`). CTest, the exact grouped-Q/K/V operator fixture,
exact-blue, vLLM boundaries 129/257, and all 127 teacher-forced positions preserve their prior results.

## 2026-07-25: Tile exact NVFP4 weight scales into the final allocation

Date: 2026-07-25
Decision: Keep packed E2M1 weights in checkpoint row-major order, but reorder every manifest-classified
`NVFP4_LOCAL_SCALE_E4M3` tensor at load time to `[row tile 8][K64 block][row][4 scale bytes]`. Stream each bounded
host tensor directly into its final arena address. Make this the only native SM120 scale layout for decode and
prefill; expose no selector and retain source order only in correctness/SIMT probes.
Context: Each output warp needs one four-byte scale vector for each of eight rows. Source row-major order places
those words at large row strides. The tiled order places them in one 32-byte region and lets decode use a constant
32-byte K-step instead of repeated strided 64-bit address construction.
Alternatives: Keep direct scale order; repack packed weights too; retain both device layouts; expose an opt-in.
Direct scales leave measured performance unused. Packed-weight repacking has no evidence, while duplicate layouts
and switches violate the memory and single-winner contracts.
Consequences: Every quantized value, scale byte, global divisor, and FP32 K accumulation remains unchanged. The
weight arena remains 9,200,135,680 bytes; the largest transient host vector is 3,686,400 bytes and exists only
during model load. Prefill uses 128 registers/10,240 shared bytes and decode uses 40 registers; both report zero
stack/local memory. Runtime JSON distinguishes direct packed weights from the mandatory scale layout.
Evidence: Against detached `e17049b`, 128/512/2,048-token prefill medians improve by 1.10%/1.43%/3.19%. Nsight
reduces NVFP4 time by 4.61% and all GPU-operation time by 2.22% at 512. A short context-128 decode rises from
25.54 to 31.63 tok/s with identical checksum; the complete Layer-0 MLP falls from 0.480 to 0.260 ms. CTest,
exact-blue, 129/257 vLLM boundaries, Layer-0, and all 127 teacher-forced positions preserve their prior metrics.
Evidence is retained under
`benchmarks/results/2026-07-25/e17049b-worktree/blackwell16gb-linux-nvfp4-scale-tile/`.

## 2026-07-25: Pipeline NVFP4 activation staging with two cp.async buffers

Date: 2026-07-25
Decision: Make the production M128xN64 NVFP4 prefill CTA double-buffer its exact packed activation bytes and E4M3
activation scales. Use 16-byte and 4-byte `cp.async` transfers with zero fill for token tails, and issue the next
K64 stage while the current stage feeds eight native block-scaled OMMAs. Replace the synchronous stage directly.
Context: CTA-wide activation reuse removed more than half of NVFP4 time, but every K64 iteration still synchronously
loaded and stored its shared tile before MMA work could begin. The independent K64 accumulation provides a natural
ping-pong boundary without changing arithmetic.
Alternatives: Retain synchronous staging; add a runtime pipeline selector; use an N128 CTA; prefetch weights only
in registers. N128 loses about 2.5% end to end and was removed. Register-only weight prefetch was neutral. A public
selector conflicts with the single-winner plan.
Consequences: Static shared memory rises from 5,632 to 10,240 bytes and registers from 123 to 124; stack/local
memory remain zero. SASS contains the expected `LDGSTS` asynchronous copies. Arena, checkpoint layout, launch
count, weight loads, token-tail values, and FP32 K accumulation are unchanged.
Evidence: A neighboring 30-run context-512 comparison raises mean/median throughput by 1.92%/1.71%. Paired
throughput differences have a 95% interval of +10.68 to +43.69 tok/s (`t=3.37`, 29 degrees of freedom). At 2,048
tokens the 3/10 median rises from 1,307.64 to 1,329.51 tok/s with non-overlapping confidence intervals. Nsight
reduces NVFP4 time by 16.9% and projected total GPU time by 7.5%. All fixed correctness metrics remain identical.
Evidence is retained under
`benchmarks/results/2026-07-25/d8b73ce-worktree/blackwell16gb-linux-nvfp4-async-pipeline/`.

## 2026-07-25: Reuse each NVFP4 activation K64 slice across an M128xN64 CTA

Date: 2026-07-25
Decision: Make the sole production NVFP4 prefill kernel use eight-warp, 256-thread M128xN64xK64 CTAs. Retain the
existing M128 per-warp accumulator stack and direct N8 packed-weight fragments, but cooperatively stage the exact
M128xK64 packed activation bytes and E4M3 activation-scale words once in shared memory for all eight output warps.
Context: After the M128 register-reuse promotion, Nsight still attributed about 47% of projected prefill GPU time
to NVFP4. Every output warp independently issued the same activation loads and address arithmetic while only its
N8 weight fragment differed.
Alternatives: Prefetch the next weight fragment in registers; extend each warp to M256; retain four-warp N32
CTAs; add a selectable CTA size; decode or repack weights. Adjacent 3/10 measurement found register prefetch
neutral/slightly slower and it was removed. M256 spills. Permanent geometry choices conflict with the one-winner
plan, and weight conversion is unnecessary before exact activation reuse is exhausted.
Consequences: The kernel uses 123 registers, 5,632 static shared bytes, zero stack/local memory, and no new arena
or persistent allocation. Packed checkpoint weights and scales remain unchanged. K64 and FP32 accumulation order
within every M16N8 result is identical. No runtime selector or fallback is added.
Evidence: Against a separately built `2366c03` reference, 3/10 median prefill improves by 26.0%/29.8%/26.7% at
128/512/2,048 tokens. The 512 and 2,048 confidence intervals do not overlap. Nsight reduces Gate+Up, Down, and
total NVFP4 time by 49.4%, 54.4%, and 51.2%. Release CTest, exact-blue, vLLM 129/257 boundary logits, and every
teacher-forced aggregate remain unchanged. Evidence is retained under
`benchmarks/results/2026-07-25/abb430c-worktree/blackwell16gb-linux-nvfp4-cta-m128n64/`.

## 2026-07-25: Vectorize checkpoint-FP8 key reads without reordering QK

Date: 2026-07-25
Decision: Make fused checkpoint-FP8 prefill attention load aligned key rows in 16-byte vectors, extract their FP8
bytes in increasing dimension order, and retain the existing serial FP32 FMA accumulation. Use the scalar loop only
for internal geometries whose row address or extent is not 16-byte aligned; product model shapes always satisfy the
wide-load invariant. This is deterministic geometry handling, not a runtime performance option.
Context: Linux Nsight Systems attributed 41.3% of context-512 projected GPU time to the fused attention kernel.
Each score thread consumed a contiguous FP8 key row one byte at a time; the compiler could not combine those loads
across the loop-carried FMA dependency.
Alternatives: Parallelize each QK dot product across a warp; introduce an approximate or tensor-core attention
route; retain scalar loads. The warp prototype was faster but changed the first generated token on 129/257-token
synthetic prompts and was discarded. Wider loads obtain a larger gain without changing arithmetic.
Consequences: The production FP8 kernel uses 48 registers, 3 KiB shared memory, and no stack/local memory. Score
storage, softmax, value accumulation, cache semantics, arena sizes, and launch count are unchanged. A true online
FlashAttention design remains the next architectural opportunity but must establish its own numerical evidence.
Evidence: Against a separately built `c0c9b42` reference at context 512 with 3 warm-ups and 10 runs, throughput
improves from 603.42 to 698.25 tok/s (+15.72%) and TTFT falls from 848.50 to 733.27 ms (-13.58%). Nsight measures
705.49 to 399.53 ms (-43.37%) fused-attention time. A 32-dimensional FP8 fused/reference fixture is bit-identical;
exact-blue, exact 129/257-token eight-step sequences, and release unit/CUDA tests pass.

## 2026-07-25: Reuse FP8 weights across two prefill token tiles

Date: 2026-07-25
Decision: Assign two consecutive 16-token MMA tiles to each production FP8 batch-projection warp, reusing the
loaded Q/K/V/O weight fragment for both. Replace the one-tile mapping directly without a selector.
Context: After the NVFP4 promotion, FP8 attention projections still consumed 16.0% of context-512 projected GPU
time. Their source-layout `m16n8k32` mapping had the same adjacent-token weight reload as the old NVFP4 kernel.
Alternatives: Retain one tile per warp; add a configurable tile count; defer all projection work until a complete
asynchronous pipeline exists. The first wastes measured bandwidth, the second violates the single-winner plan, and
the third leaves an independently validated gain unused.
Consequences: Accumulator count grows while each tile retains its original K-order. `cuobjdump` reports 56
registers and zero stack/local memory. Source layout, scales, arena sizes, and tail masking are unchanged.
Evidence: Against a separately built `b032e6f` binary at context 512 with 3 warm-ups and 10 measured runs, median
throughput improves from 587.87 to 605.33 tok/s (+2.97%) and TTFT falls from 870.95 to 845.82 ms (-2.89%). Nsight
Systems measures 280.18 to 245.73 ms (-12.30%) for FP8 projection kernels across two prefill executions. Exact-blue,
exact 129/257-token eight-step sequences, and release unit/CUDA tests pass.

## 2026-07-25: Reuse NVFP4 weights across two prefill token tiles

Date: 2026-07-25
Decision: Make the production NVFP4 batch projection assign two consecutive 16-token MMA tiles to each warp. Load
each source-layout Gate, Up, or Down weight fragment and scale once per contracting block, then issue one MMA for
each token tile with independent activation fragments, scales, and accumulators. This replaces the prior one-tile
kernel directly and adds no runtime selector.
Context: After widening prefill chunks, Nsight Systems attributed 35.5% of projected context-512 GPU time to NVFP4
projections. The old mapping made separate warps reread identical weight fragments for adjacent token tiles even
though batch-one prefill has abundant token-parallel work.
Alternatives: Keep one tile per warp; add a user-selectable tile count; change the K-dimension accumulation order;
stage a larger weight tile in shared memory. A permanent selector conflicts with the single-winner execution plan,
changing accumulation order weakens the numerical comparison, and shared staging is not justified before measuring
this register-only reuse.
Consequences: Each warp carries twice as many output accumulators. The unfused and fused forms use 72 and 80
registers respectively, but `cuobjdump` reports zero stack and local memory. Arena sizes, source weight layout,
global/local scaling, FP32 accumulation order within each tile, and tail masking are unchanged.
Evidence: A separately built `8f05333` reference and the promoted kernel were measured with 3 warm-ups and 10 runs
at context 512. Median throughput changes from 542.58 to 587.68 tok/s (+8.31%); TTFT changes from 943.64 to
871.23 ms (-7.67%). Nsight Systems reports 669.99 to 547.20 ms (-18.33%) for NVFP4 projection kernels across two
prefill executions. Exact eight-token sequences match at prompt lengths 129 and 257, the exact-blue fixture passes,
and all release unit/CUDA tests pass.

## 2026-07-25: Promote measured winners and remove production optimization switches

Date: 2026-07-25
Decision: Expose one production execution plan: native SM120 projections, context-budgeted chunked prefill (128 tokens
by default), fused causal
prefill attention, separate Gate/Up/GELU, complete decode graphs, and fused warp-row output reduction. Remove the
six public projection/prefill/fusion/graph A/B switches and their option fields. Keep slower and reference
implementations callable only from dedicated tests and characterization probes.
Context: The CLI had accumulated six optimization switch families and 68 source/documentation occurrences. This
made a validated fast path look optional and allowed ordinary runs to select known-slower plans. Gate/Up was the
only unresolved choice because a 1.7% isolated kernel gain had not survived Windows end-to-end measurement.
Alternatives: Keep every A/B switch indefinitely; enable every fusion; delete reference kernels as well as runtime
dispatch; introduce an automatic tuner. Persistent switches weaken the product contract, while deleting probes
would weaken correctness evidence and an automatic tuner would make plans non-deterministic.
Consequences: Product CLIs have no optimization opt-ins or opt-outs. A diagnostic logits/state request may still
use its required capture mechanics, and explicit BF16 K/V remains a labeled numerical correctness mode rather than
a performance implementation choice. New implementations must beat the current plan with correctness evidence
before replacing it; they are characterized through tests/probes rather than shipped as permanent toggles.
Evidence: Existing gates establish fused prefill attention (+5.6%), complete decode graphs (+31% over direct in the
same Windows characterization), and fused output reduction (+0.75%) with matching token checks. Linux Gate/Up A/B
at commit `960528d` selected separate operations: Prefill 128 was 573.32 versus 527.53 tok/s, Prefill 512 was 441.73
versus 410.16 tok/s, and context-128 Decode was 26.08 versus 25.86 tok/s. All compared runs retained deterministic
checksums.

## 2026-07-25: Use warp-row candidate reduction for greedy decode output

Date: 2026-07-25
Decision: Make the decode-only tied-BF16 output path evaluate one vocabulary row per warp, execute eight rows per
block, apply the configured softcap to every row, and retain one candidate per block before a final GPU reduction.
Keep the original full-logit output head and separate argmax for internal arithmetic and performance probes. The
later production-path consolidation removes its public runtime switch.
Context: The original head launched one 256-thread block for each of 262,144 rows, wrote every float32 logit, and
then scanned that array in a separate argmax kernel. An initially fused implementation preserved the exact
256-thread addition order but serialized multiple rows per block and was slower. Row-per-warp execution exposes
enough parallel rows while avoiding the full-logit write/read in ordinary greedy decode.
Alternatives: Retain only the full-logit path; accept the slower exact-order fused implementation; skip the monotonic
softcap during argmax; change the BF16 tied weights to a lower precision. The latter two violate the checkpoint and
benchmark contracts and were rejected without measurement.
Consequences: Decode dot products use a 32-lane rather than 256-thread addition tree, so diagnostic logits can
differ by a few float32 ULPs. Full-logit capture uses the same warp arithmetic when fusion is enabled, allowing
explicit comparison. The softcap, suppression behavior, lowest-token tie break, and no-allocation token loop remain
unchanged. The candidate array costs 32 KiB and the full-logit implementation remains available.
Evidence: At context 128 with 64 generated tokens and 3/10 runs, median throughput improves from 27.02 to
27.22 tok/s and median inter-token latency falls from 36.94 to 36.62 ms. Nsight measures the isolated output stage
at about 3.031 ms instead of 3.315 ms. Across 31 sky positions, fused versus retained logits have max absolute error
`7.6294e-6`, RMS error `1.0358e-6`, cosine similarity `0.9999999999999948`, Top-1 agreement 31/31, and mean Top-20
overlap 20/20. The 31-token sky and 32-token integer autoregressive sequences are identical.

## 2026-07-24: Match checkpoint FP8 K/V semantics by default and label BF16 explicitly

Date: 2026-07-24
Decision: When the checkpoint declares its static FP8 K/V scheme and per-layer scales, apply those E4M3
quantize/dequantize semantics by default and store each cached value as one physical E4M3FN byte. Retain
`--kv-cache bf16` as an explicit correctness mode. Label the routes and physical storage in every inference result,
and reject the initial unfused cache kernels as benchmark evidence.
Context: The first sky-prompt divergence was deterministic under greedy decoding: vLLM and llama.cpp selected token
`563`, while the original BF16-cache engine selected `7412`. Layerwise prompt-derived dumps found exact V before
the cache but a difference in the first attention context after cache reuse.
Alternatives: Treat the token difference as an acceptable low-precision variation; force every reference to BF16;
claim that float storage containing dequantized FP8 values is a production FP8 cache.
Consequences: Normal chat follows the checkpoint. BF16 remains useful for isolating operator error, but results
from different cache modes must never be presented as parity comparisons.
The one-byte cache gives valid allocator accounting; optimized cache/attention kernels remain required for
performance qualification.
Evidence: Explicit BF16 vLLM and gem16gb both generate `[818,7217,7412]`. The current physical-FP8 gem16gb path
also generates `[818,7217,7412]`, while FP8-vLLM and llama.cpp generate `[818,7217,563]`; the FP8 attention
difference remains open. At context 64 the physical FP8 allocation is 11,010,048 bytes versus 44,040,192 bytes for
the float32 BF16-semantics diagnostic cache.

## 2026-07-24: Expose checkpoint chat semantics through a native C++ boundary

Date: 2026-07-24
Decision: Implement checkpoint byte-fallback BPE, text chat rendering, decoding, and generation controls in C++.
Read and identity-check the actual `chat_template.jinja`, implement its supported text branches natively, and reject
unknown revisions or unsupported roles. Keep the processor independent of terminal I/O for later Chat Completions
reuse.
Context: The user-facing chat executable must not depend on Python or Transformers. A generic Jinja runtime would
add a broad dependency and still require careful model-specific semantics, while silently hard-coding a template
without reading the artifact would violate the checkpoint contract.
Alternatives: Retain the Python bridge; embed Python; vendor a general Jinja interpreter; accept only manual token
IDs.
Consequences: `gem16gb-chat` is a self-contained C++ process and the tokenizer/processor can later serve HTTP
requests. The supported template revision is explicit. Tool-call and multimodal branches remain unsupported until
implemented and tested. The engine still reloads weights per turn until a persistent session API is introduced.
Evidence: Native C++ rendering and BPE reproduce the committed 20-, 23-, and 27-token prompts exactly, and the
CUDA one-shot path produces and decodes `[9503, 106]` as `blue`.

## 2026-07-24: Use cross-engine distributions and quality, not bit identity

Date: 2026-07-24
Decision: Do not require generated tokens or logits to be bit-identical to vLLM or llama.cpp. Require unexplained
large or early deviations to be investigated with full logits, hidden states, quality tasks, and independent
references before setting measured tolerances.
Context: vLLM consumes the mixed FP8/NVFP4 source directly, while the closest-parity llama.cpp candidate maps FP8
attention to BF16 and uses different kernels. They nevertheless agree for most current tokens but eventually
diverge, as expected from autoregressive sensitivity.
Alternatives: Require exact token equality indefinitely; accept any coherent-looking text; select one runtime as
infallible.
Consequences: Product correctness is based on operator contracts, distribution metrics, generation stability, and
task quality. Early disagreement still blocks acceptance until configuration differences are excluded; tolerances
are not invented merely to accept it.
Evidence: The sky step-2 disagreement was investigated rather than waived. FP8-versus-BF16 K/V semantics determine
vLLM's result, but the current precision-matched FP8 gem16gb path still differs and remains an active correctness
gate.

## 2026-07-23: Qualify unfused full-layer composition before fusion

Date: 2026-07-23
Decision: Compose the validated FP8 local-attention and NVFP4 MLP routes into a complete Layer-0 device path before
introducing fused Q/K/V, Gate/Up, residual, or CUDA Graph implementations. Keep independent CUDA scalar-projection
and direct SM120 paths alive through the final layer output and expose their quantization-boundary differences.
Context: Individual operators and sublayers were numerically close, but a quantization boundary can amplify small
attention differences. A full layer is the smallest executable unit that proves the residual, norm, mixed-format,
and `layer_scalar` ordering together.
Alternatives: Begin fusion from isolated kernel results; join sublayers through host memory; wait for tokenizer and
embedding support before testing full-layer composition.
Consequences: The characterization deliberately owns two copies of execution buffers and is not a production
memory plan. It establishes a no-host-roundtrip correctness path and a stable orchestration gate while preserving
the requirement for a later prompt-derived trusted hidden-state comparison.
Evidence: The real Layer-0 path produces zero differing bytes at both NVFP4 activation boundaries. Its final
CUDA-reference/direct-SM120 comparison has maximum absolute error `4.7683716e-6`, RMS error `2.8454761e-7`, and
cosine similarity `0.9999999999999643`.

## 2026-07-23: Store final K and V cache states separately

Date: 2026-07-23
Decision: Reuse the single full-attention K projection output as the input to both K and V post-processing, but
always allocate and append separate final K and V cache states. Reject `--kv-storage shared` rather than accepting
an invalid memory optimization. Continue reporting a one-state byte count only as a diagnostic lower bound.
Context: The executable Layer-5 path resolves the earlier ambiguity around `attention_k_eq_v=true`. The raw K
projection is shared, but K then receives its learned per-head RMSNorm and proportional RoPE while V receives a
scale-free RMSNorm and no RoPE. Their stored values are therefore distinct.
Alternatives: Physically share the cache because the projection tensor is shared; recompute one state during every
attention read; leave the option selectable until end-to-end assembly.
Consequences: The one-byte FP8 cache budget at 64K is 672 MiB rather than 336 MiB. The memory plan remains below the
16 GB target and now matches the implemented model semantics. Projection reuse still avoids a separate `v_proj`
weight read and launch on full-attention layers.
Evidence: The real Layer-5 checkpoint probe binds the absent `v_proj` as a reused raw K projection, applies the two
distinct post-processing paths, appends both states, and matches the independent CUDA scalar route with maximum
absolute error `4.5299530e-6`, RMS error `5.5268314e-7`, and cosine similarity
`0.9999999999999085`.

## 2026-07-23: Bring up NVFP4 from an exact oracle into separate decode and prefill plans

Date: 2026-07-23
Decision: Implement the E2M1/E4M3FN and compressed-tensors divisor contract first, followed by an explicit
correctness CUDA route, direct source-layout SM120 fragment views, and independently measured packed-GEMV and native-MMA decode
candidates. Fuse Gate/Up only after the common input/global-scale invariant is validated; build prefill as a
separate plan and keep FP8 attention as a separate precision backend.
Context: NInfer demonstrates the value of closed, shape-specific plan catalogs, arenas, graph-stable addresses, and
Gate/Up fusion, but its integer Q4 format and offline `.ninfer` artifact are incompatible with this checkpoint.
A neighboring SM120 prototype demonstrates the native block-scaled instruction and operand-fragment mapping, but it
retains multiple device layouts and previously exposed an input-global-scale semantic error. The pinned Gemma
checkpoint stores compressed-tensors global divisors, has exact SM120-friendly MLP dimensions, and uses mixed FP8
attention plus NVFP4 MLP projections.
Alternatives: Start with a complete unfused model and debug quantization indirectly; copy the neighboring loader and
retain raw plus multiple repacked device tensors; assume native MMA is fastest at `T=1`; use one GEMM plan for decode
and prefill.
Consequences: Kernel work begins later but every route shares one independent oracle. The 16 GB memory contract is
preserved, silent precision fallback remains impossible, and the project obtains direct evidence for the actual
batch-one winner. Gate/Up can reuse one activation quantization and later fuse the GELU-tanh epilogue. The first
native candidate adds no persistent repacked weight or expanded scale copy; a streamed transformation remains a
measured fallback. Loader and kernel layouts remain architecture-specific implementation details behind the
manifest contract.
Evidence: All 48 Gate/Up pairs have identical stored input and weight divisors. The real Gate/Up and Down shapes are
divisible by the intended 128/64 outer/contracting geometry. The 144 local-scale tensors contain 530,841,600
positive, nonzero E4M3FN bytes with no NaN encoding.

## 2026-07-23: Keep the first memory plan explicit and evidence-bounded

Date: 2026-07-23
Decision: Build a deterministic 256-byte-aligned base arena from the parsed text-only tensor inventory and context
metadata. Calculate both one-state and separate K/V payloads, require an explicit selection, and leave execution
workspaces visibly unplanned until kernel shapes define them. The later Layer-5 decision above resolves the storage
selection to separate K and V.
Context: At the time, the checkpoint proved `attention_k_eq_v=true`, but physical shared-cache semantics and
workspace sizes had not yet been validated by an executable model path. A 16 GB budget cannot tolerate hidden or
guessed allocations.
Alternatives: Assume shared K/V immediately; reserve budget-table maxima as real allocations; defer all memory work
until CUDA kernels exist.
Consequences: Weight, scale, and KV offsets are deterministic and overflow-checked now. Memory reports remain useful
without claiming peak VRAM. The plan is deliberately incomplete until activations, logits, sampling, graph, kernel,
and prefill workspaces are derived and measured.
Evidence: The locked 1,389-tensor manifest yields 9,200,026,528 text-only bytes. At 64K, parsed layer metadata yields
336 MiB shared or 672 MiB separate one-byte K/V payloads, matching the independently documented formula.

## 2026-07-23: Support Linux and Windows in the repository foundation

Date: 2026-07-23
Decision: Keep one Ninja-based preset layout for both operating systems, isolate file mapping behind POSIX and Win32
implementations, and provide native Bash and PowerShell build entry points. Add Windows host CI while retaining the
Linux sanitizer path.
Context: Development moved from Linux to Windows on the same Blackwell machine. Loader and build work must remain
reproducible on both systems without weakening the Linux reference path.
Alternatives: Develop only through WSL; maintain unrelated Windows CMake targets; replace memory mapping with full
file reads.
Consequences: Host and SM120a capability builds share target names and their internal `bin`/`lib` layout, while
OS-named build roots prevent incompatible CMake caches from colliding. Windows uses Unicode-aware Win32 file
mapping and self-discovers MSVC through Visual Studio Build Tools. ASan/UBSan remains Linux-only until a
Windows sanitizer configuration provides comparable signal. Linux remains the production platform required by the
phase-one contract, while Windows is now a supported development and validation host.
Evidence: On the reference Windows installation, MSVC 19.44 and CUDA 13.3 configure and build both presets with
warnings as errors; host and CUDA CTest runs pass.

## 2026-07-21: Keep CUDA opt-in during repository initialization

Date: 2026-07-21  
Decision: Provide separate host-debug and Blackwell CUDA presets. Do not label the CUDA runtime probe as a native
kernel path.  
Context: Parser and manifest work must build on machines without CUDA, while performance builds must remain
architecture-specific.  
Alternatives: Require CUDA for every build; silently build host-only when CUDA is absent.  
Consequences: CPU CI stays useful; `GEM16GB_ENABLE_CUDA=ON` fails if CUDA is missing; native capability remains false
until implemented.  
Evidence: The neighboring `qwen35x` repository successfully uses optional CUDA language enablement, but its
silent CPU fallback was tightened here to a fatal error when CUDA is explicitly requested.

## 2026-07-21: Implement a strict in-repository JSON parser

Date: 2026-07-21  
Decision: Use a small C++ parser with duplicate-key rejection, resource limits, Unicode validation, and checked
integer parsing for initial config and Safetensors work.  
Context: Runtime dependency count should remain small and model files are untrusted input.  
Alternatives: Vendor a JSON library immediately; use string searching.  
Consequences: The parser is narrowly testable and dependency-free, but it carries maintenance responsibility and
must be fuzzed before the loader is considered production-ready.  
Evidence: Neighboring ad-hoc string-search Safetensors code does not meet this repository's schema and security
requirements.

## 2026-07-21: Target the 16 GB CUDA hardware class, Blackwell first

Date: 2026-07-21  
Decision: Define the product target as NVIDIA CUDA GPUs with approximately 16 GB VRAM. Optimize and validate the
first backend on the available Blackwell compute-capability-12.0 GPU.
Context: The engine should become useful across the 16 GB CUDA class; retail board form factors do not belong in the
architecture or project identity.
Alternatives: Bind the project to one retail board; attempt multi-architecture kernels before the first backend is
correct and competitive.
Consequences: Blackwell remains the immediate kernel and benchmark target. Later GPU backends must preserve the same
correctness, memory, and benchmark contracts, and exact board details remain benchmark metadata rather than product
scope.
