# Decisions

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
