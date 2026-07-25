# Decisions

## 2026-07-25: Use warp-row candidate reduction for greedy decode output

Date: 2026-07-25
Decision: Make the decode-only tied-BF16 output path evaluate one vocabulary row per warp, execute eight rows per
block, apply the configured softcap to every row, and retain one candidate per block before a final GPU reduction.
Keep the original full-logit output head and separate argmax behind `--disable-fused-output-head`, and use it for
arithmetic and performance A/B checks.
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
