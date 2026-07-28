# Architecture

## Current loader path

`gem16-inspect` validates required checkpoint metadata, parses `config.json`, compiles its quantization target regexes
once, memory-maps Safetensors files, and builds a deterministic tensor manifest. No model payload is copied into
host RAM by the inspector.

The manifest is the only planned source for tensor names, shapes, offsets, dtype classes, scale relationships,
text-only residency, and tied-weight aliasing. Execution code must consume it rather than infer tensor inventory.

## Hardware backend boundary

`gem16` targets the approximately 16 GB NVIDIA CUDA GPU class. Architecture-specific kernels and dispatch live
behind explicit capability checks; the first implementation is Blackwell SM120/SM120a. Model execution plans,
allocator contracts, tensor manifests, and correctness fixtures must not encode a retail board name. A later CUDA
architecture backend should reuse those contracts while supplying its own kernels and measured dispatch choices.

## Planned execution split

Prefill and decode use separate execution plans. Both draw from named preallocated device arenas. Decode uses fixed
addresses and captures a complete greedy forward graph during initialization. A preallocated pinned-host control
record is copied into a device control record at graph launch and supplies the current token, position, and
suppression count to device-side RoPE, KV append, ring selection, attention, and argmax logic. Two
position-independent segments per layer are also retained for diagnostic forwards that request hidden states or
full logits. Decode may not allocate, access files, capture graphs, or compile code in the token loop.

The ordinary greedy output stage assigns one vocabulary row to each warp, evaluates eight rows concurrently per
block, applies the checkpoint's softcap to every logit, and writes one `(value, token)` candidate per block. A small
second reduction chooses the token with the same lowest-token tie break as the reference full-logit head. This
changes only the dot-product addition tree. Diagnostic logit capture writes the warp-row logits without changing
selection; the reference head is restricted to tests and characterization probes. The output-head kernels, their
candidate type, and all launch/error handling live in `src/cuda/output_head.{h,cu}`; `inference.cu` retains only
fixed-arena ownership and execution-plan dispatch. This boundary is deliberately format- and model-specific rather
than a generic graph abstraction.

Sampling is an explicit, separate output plan; disabled sampling preserves the fused greedy graph and workspace.
The exact plan lives in `src/cuda/sampling/`, materializes softcapped logits, applies full-history repetition
penalty and suppression, divides by temperature, and performs a descending CUB radix sort in preallocated
workspace. Filtered probabilities are accumulated by a preallocated in-place CUB double-precision inclusive scan;
a constant-work final kernel uses binary searches to apply top-k, min-p relative to the maximum probability, and
top-p in that order before drawing from a SplitMix64 stream keyed by seed and output step. The sampled whole-model decode graph reads
the changing step from its copied device control record, updates repetition history, and returns only the selected
token. An atomic bitset tracks repetition history without duplicate-token write races. Sampling adds about 7.1 MiB
to the context-128 workspace and performs no token-loop allocation. The same bounded parallel plan handles top-k
64 and the unfiltered full vocabulary at measured greedy-performance parity.

The model-specific sequence is attention normalization and FP8 projections, specialized local/global attention,
then NVFP4 MLP projections and residual updates. This sequence now exists both as an independent Layer-0 comparison
probe and as a fused, batch-one 48-layer greedy characterization. The latter loads the complete text-only model
into one aligned arena, keeps separate K/V state and reusable workspace allocations fixed for the run, applies the
tied BF16 embedding/output matrix, exact logit softcap, and GPU candidate reduction, and performs no token-loop allocation. It
uses one full graph replay for ordinary greedy decode but is not yet benchmark-qualified.

The first full-model path intentionally accepts token IDs and uses a hybrid cache through the checkpoint's 262,144
position contract. Its 40 local-attention layers use fixed 1,024-token rings; its eight full-attention layers use
absolute, growing storage. Checkpoint-FP8 prefill uses one fixed 2,048-token chunk. Attention projections run as
CUTLASS SM120 128x128x64 warp-specialized FP8 GEMMs directly over checkpoint-order activation and weight bytes,
followed by explicit per-token/per-channel scaling in FP32. Q/K/V are separate prompt GEMMs. Decode uses the
binding-dimension grouping around the
latency-oriented T=1 direct-source kernel, reducing three independent graph nodes to one while retaining each
projection's original CTAs and MMA ordering. Gate, Up, and Down prefill use CUTLASS SM120 block-scaled GEMM over a
temporary arena view. Packed E2M1 weights and their
E4M3 scales are transformed byte-exactly into the sole final Row8/K64 device layout at model load; no raw GPU copy
survives. This replaces strided per-row decode weight loads without changing any nibble, scale, global
divisor, or FP32 K accumulation. It evaluates local D256 and global D512
causal attention with shape-specific online Tensor-Core kernels. Those
kernels stage current-chunk K/V directly, read older positions from the circular or growing cache, retain row max,
normalization sum, and output accumulators in FP32, and never materialize a global score matrix. K/V is committed
only after the layer's attention finishes. The serial path remains a test/probe oracle and is not a runtime option. The pure C++
`GemmaChatProcessor` loads the checkpoint vocabulary, merge ranks, byte fallback, generation controls, and exact
pinned Jinja artifact. It implements the supported text-only behavior of that template natively and rejects a
different template revision rather than silently approximating it. This makes real chat flows testable now while
preserving a narrow execution contract. Interactive chat owns one `ConversationSession` for its lifetime. The
session keeps weights, arenas, CUDA Graphs, and hybrid KV storage resident, records exactly which token IDs have
materialized cache entries, and requires every later prompt to extend that prefix exactly. The CLI preserves the
original generated IDs instead of decode/re-encode round trips, then tokenizes only the continuation delimiter,
new user message, and generation header. The new suffix uses batch prefill at the existing absolute cache position;
generated tokens continue through a whole-model decode graph for both greedy and sampled generation. Diagnostic
state or full-logit capture retains the direct layer-segment path so observability does not complicate the ordinary
graph.

The reusable `ChatMessage`, `Tokenizer`, and `GemmaChatProcessor` interfaces are deliberately independent of
terminal I/O. A future OpenAI-compatible Chat Completions server can reuse this request-to-token boundary; HTTP,
JSON request schemas, streaming, and session scheduling are not part of the current CLI milestone.

The greedy plan copies checkpoint `suppress_tokens` into fixed workspace before prompt processing and stops on any
checkpoint EOS token. Optional full-logit capture preallocates host storage before the token loop and writes raw
little-endian float32 only after generation; it is a correctness diagnostic and invalidates timing comparisons.

## MTP assistant boundary

The target checkpoint's direct upload, SM120 Row8/K64 layout transformation, tensor validation, and fixed
`LayerBinding` construction are isolated in `src/cuda/engine/target_model.{h,cu}`. The execution engine consumes
those immutable bindings and owns only cache/workspace addresses. Its private Pimpl boundary is declared in
`cuda/engine/inference_engine.h`: CUDA arenas, immutable plans, graph capture, prefill/decode execution, and MTP
verification remain in one CUDA translation unit rooted at `cuda/engine/inference_engine.cu`, while
`cuda/inference.cu` contains host session, run, and benchmark orchestration. Private class-body fragments group
initialization, memory planning, ordinary forward, prefill layers, decode graphs, and MTP independently without
introducing a cross-translation-unit launch boundary or changing kernel code generation.

The optional official MTP assistant is a separate model owner, not part of the target weight arena. It memory-maps
its source checkpoint, streams all 48 BF16 tensors into one independently allocated 256-byte-aligned device arena,
and binds its tied embedding, pre/post projections, final norm, and four exact Q-only layer families at fixed
addresses. No converted or second device layout exists. Load-time device prefix/suffix probes cover every tensor.
The target continues to own all KV storage. Three Q-only sliding assistant layers read the target Layer-46 local
ring, while the final Q-only full layer reads the target Layer-47 contiguous cache. All iterations in one draft
group retain the target's last processed position and cache view; each projected 3,840-dimensional assistant state
is paired with the next scaled target embedding. `--mtp-draft-tokens 1|2|4` activates BF16 assistant execution and
batched exact target verification. Each batch retains tentative K/V rows in a dedicated fixed arena and restores
local-ring writes after attention. Drafts remain device-resident; a GPU acceptance kernel applies stop IDs and
selects the exact prefix, fixed kernels commit tentative K/V and the selected hidden row, and one compact pinned
result is returned to the scheduler. Target verification uses decode-order direct FP8 Q/K/V because the CUTLASS
Q/K/V batch changes long-context greedy output; the T≤5 O projection uses the same direct K32 accumulation rather
than a short CUTLASS tile. T≤5 NVFP4 Down uses one unstaged token tile and four warps. Both retain K accumulation
order. The MTP transaction kernels and their fixed result layout live in `src/cuda/mtp/verify.{h,cu}`, while the
engine retains the fixed arenas and execution ordering. At FP8 capacities above 512, assistant attention reuses
split-online decode attention. The SM120 attention implementation is divided by execution geometry into
`attention/decode_sm120.cu`, `attention/prefill_local_sm120.cu`, and `attention/prefill_global_sm120.cu`; each
translation unit keeps its complete kernel schedule and exact device primitives together. Correctness and boundary
operators are likewise grouped by responsibility under `attention/reference.cu`, `kv_cache/reference.cu`,
`norm/reference.cu`, and `rope/reference.cu`; `layer/reference.h` remains the narrow compatibility declaration
surface while callers migrate to operator-specific headers. `--mtp-adaptive` optionally
selects D4/D2/D1 and ordinary fallback from context and 16-group acceptance windows. This scheduler emits only
target-verified tokens and therefore retains ordinary greedy output.

The next execution boundary before multimodal is a fully GPU-controlled greedy MTP decode graph. It does not alter
the target or assistant math. An arena-backed `MtpDeviceControl` will own the current input token, processed
position, remaining length, stop state, output index, and graph mode. Development first mirrors the existing host
transition and asserts parity after each compact result. A complete fixed-D2 graph then captures both assistant
steps, all 48 target layers, acceptance, and transactional commit before conditional graph execution chains groups
without a host dependency. Stop and final-tail handling move to device control only after fixed-D2 chaining passes.

Streaming is a separate boundary from graph scheduling. A preallocated GPU-producer/host-consumer ring contains
only target-verified token IDs and monotonically published indices. A host poller may decode text and invoke the
existing callback, but normal compute progress does not wait for callback completion. Ring capacity, system-visible
memory ordering, shutdown, and backpressure are explicit plan properties; no growing host container, per-token
allocation, or pageable transfer is introduced. Adaptive D1/D2/ordinary graph branches follow fixed-D2 streaming,
not precede it. An optional later `ngram-mod` child branch may look up fixed D2/D4 proposals from a deterministic,
fixed-capacity device hash table before MTP; a hit skips assistant execution and a miss falls through to MTP, while
both paths share the exact verifier and transaction nodes. This is proposal-source routing, not token-level merging.
It is retained only if per-source hit/acceptance telemetry and representative end-to-end benchmarks win. The
previously removed verifier-suffix graph remains evidence that graph capture alone is not a performance result:
every phase requires an Nsight timeline, memory accounting, and exact ordinary/MTP identity.
The detailed order and gates are binding in [MTP.md](MTP.md).

## Planned multimodal boundary

The pinned Gemma 4 12B Unified checkpoint contains an encoder-free vision embedder and a direct audio-waveform
projection in addition to the currently resident text tensor set. The multimodal expansion will compile ordered
text/image/audio content into one immutable prompt plan, project media rows into the existing 3,840-wide input
embedding sequence, and reuse the current 48-layer transformer and generated-token decode graph.

Vision is not only an input-projection change. Image and video spans require same-block bidirectional visibility in
sliding-attention layers while full-attention layers remain causal. Prefill chunk boundaries must therefore keep a
complete vision block available to the online-attention kernel. Audio remains causal. Resident session identity
must also include canonical media digests because token placeholders alone do not identify the media-derived K/V
state.

The complete tensor inventory, processor contracts, mask formula, arena additions, public-boundary changes,
correctness/benchmark matrices, and ordered delivery gates are binding in
[MULTIMODAL.md](MULTIMODAL.md). Until those gates pass, all production paths remain explicitly text-only.

## NVFP4 execution boundary

The NVFP4 MLP backend has three deliberately separate layers:

1. A platform-independent numeric contract and CPU oracle define E2M1, E4M3FN, compressed-tensors global-scale
   divisors, dynamic local activation quantization, and the observable output cast.
2. A loader-owned weight view keeps packed E2M1 values in source Safetensors order and transforms only local E4M3
   scale byte order to the measured SM120 row8/K64 access order. Each bounded host tensor is written directly to
   its final arena address; no raw device copy or second persistent layout exists.
3. Operator-owned decode and prefill plans select only explicitly qualified implementations for an exact shape and
   token extent. A correctness route, packed SIMT/GEMV route, and native SM120a MMA route are distinct capabilities;
   none may silently stand in for another.

For the pinned checkpoint, Gate and Up have identical input and weight global divisors in all 48 layers. The native
decode plan may therefore quantize their shared input once, contract both matrices, and apply Gemma's GELU-tanh
product in one closed operator. Down performs its own dynamic-local quantization and may fuse its residual epilogue.
The attention projections remain a separate dynamic-FP8/per-channel-FP8 path.

The production FP8 prefill projection is a CUTLASS 4.5.2 SM120 128x128x64 Tensor-Core GEMM with an automatic
warp-specialized schedule and FP32 output. Checkpoint `[N,K]` weights are already the column-major B memory order
expected by the GEMM, so no repack or second weight copy is needed. A 256-thread scale kernel then applies the
dynamic per-token activation scale followed by the per-output-channel BF16 checkpoint scale, matching the former
FP32 multiplication order exactly. Q, K, optional V, and O are separate prompt GEMMs and reuse the existing 8 MiB
CUTLASS workspace sequentially. The T=1 decode plan retains its grouped native direct-source projection.

Checkpoint-FP8 prefill attention uses BF16 Tensor-Core QK and probability-times-V operations with FP32 online
softmax state. This deliberately changes the scalar FP32 reference's reduction tree and rounds MMA operands, so it
is qualified with local/global operator error distributions plus vLLM model-logit gates rather than bit identity.
The local kernel uses 32-query by 32-key tiles and processes the two query heads sharing a KV head in one
128-thread CTA. The global kernel uses 16-query by 16-key tiles and processes four query heads in one 256-thread
CTA. K/V staging is shared within each group while softmax and output state remain per head. Both consume the
physical E4M3 K/V bytes and checkpoint BF16 scales without a persistent conversion or fallback. A 2,048-token
current chunk may exceed the local 1,024-token ring because current K/V is read directly; only the newest
1,024-token suffix is committed after attention, avoiding concurrent modulo-aliasing writes. Local K/V staging
loads aligned 16-byte E4M3 vectors, converts four values per instruction, and writes paired BF16 values without
changing the attention arithmetic or 64 KiB operand allocation. Global K/V staging uses two raw-FP8 ping-pong
tiles and one overlaid BF16 operand tile within the existing 96 KiB shared allocation.
Aligned 16-byte asynchronous copies overlap current-V traffic with QK/online softmax and next-K traffic with PV;
vector E4M3x4 conversion and paired BF16 stores complete before each operand is consumed. Older global K/V uses
the absolute token position directly because the cache is contiguous and launch validation proves it is in bounds;
only the local circular cache applies modulo addressing. The attention MMA and online-softmax reduction order are
unchanged.

Gate, Up, and Down prefill use CUTLASS 4.5.2 SM120 block-scaled Tensor-Core GEMMs with a 128x128x128 thread-block tile,
automatic TMA/warp-specialized schedule, and BF16 output. CUTLASS consumes a different operand layout from the
decode-optimized persistent allocation: compact activation scales are converted once per layer into its padded
128x4 interleave, and one active projection is converted from Row8/K64 into preallocated row-major packed-weight
and interleaved-scale scratch. All three projections reuse the same scratch sequentially; Down's larger contracting
dimension also reuses an enlarged activation-scale view. The conversion and GEMM are both inside measured prompt
processing; there is no allocation, persistent second weight copy, or disk conversion. Down writes BF16 directly
into the projection buffer, which the fused residual/RMSNorm boundary consumes without an intervening FP32 round
kernel.

All decode projections retain the native T=1 implementation over the sole persistent
`[8 output rows][K64 block][row]` weight and scale allocation. The decode kernel uses 40 registers and zero
stack/local memory. All layout transformations preserve every packed value and scale byte, leave the
9,200,135,680-byte weight arena unchanged, and are mandatory rather than selectable.

Prefill and ordinary decode materialize no redundant normalized or MLP-product tensor solely to cross a
quantization boundary. Shape-specific kernels combine RMSNorm with the exact BF16 cast and dynamic FP8/NVFP4 token
quantizer, and combine the separate Gate/Up projection outputs with the exact BF16/GELU-tanh/product boundaries and
Down-input NVFP4 packing. Prefill additionally combines post-projection normalization with residual and optional
BF16 layer scaling. These kernels preserve the former operation order and bytes exactly; the unfused sequence
remains only as the CUDA test oracle and diagnostic hidden-state path. There is no fused/unfused runtime selector.

Q/K prefill and whole-model decode use another exact closed boundary. One CTA per token/head performs the projection-output BF16 cast,
the original 256-thread RMSNorm reduction, normalized BF16 cast, RoPE, and the post-RoPE BF16 cast. Q and K share
the launch but never share reduction state. Local D256 and proportional global D512 cosine/sine tables use the
former double-precision `pow`/`cos`/`sin` expressions and are generated once during initialization for every
position in the planned context. Decode reads its dynamic position from the graph control record and indexes the
same immutable tables. The tables cost 1,536 bytes per context token and eliminate per-layer trigonometry. The
prefill kernel uses 35 registers and 3,072 bytes shared memory; the decode kernel uses 24 registers and 2,048 bytes
shared memory. Both have zero stack/local memory. The initialization-only table kernel is outside prompt and decode
timing. The unfused sequence remains available for diagnostics and CUDA tests.

## Memory-plan boundary

The first runtime component now converts parsed model metadata and the authoritative text-only manifest into a
deterministic 256-byte-aligned base arena. It places immutable weights/model state, scales, and the selected KV
payload in named regions with checked offsets. The required separate K/V size and a diagnostic one-state lower
bound are retained in every result; shared physical cache selection is rejected.

The greedy characterization uses an execution workspace containing hidden-state ping-pong, quantized activations
and scales, projection intermediates, retained full logits, a 32 KiB fused-output candidate array, and GPU argmax
state. Sampling conditionally adds adjusted/sorted logits, token-index pairs, a vocabulary repetition mask, and
CUB radix-sort scratch; none of these regions is allocated for greedy generation. The checkpoint-FP8 prefill arena contains no attention-score region; only the explicit BF16 correctness
mode retains the scalar attention score workspace. Exact sizes are reported per run.
Its default hybrid cache stores physical E4M3FN bytes with checkpoint BF16 scales; an explicit float32
BF16-semantics diagnostic allocation remains available. The general planner remains conservative until production
prefill, graph, and sampling shapes are defined.
