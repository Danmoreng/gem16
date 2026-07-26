# Architecture

## Current loader path

`gem16gb-inspect` validates required checkpoint metadata, parses `config.json`, compiles its quantization target regexes
once, memory-maps Safetensors files, and builds a deterministic tensor manifest. No model payload is copied into
host RAM by the inspector.

The manifest is the only planned source for tensor names, shapes, offsets, dtype classes, scale relationships,
text-only residency, and tied-weight aliasing. Execution code must consume it rather than infer tensor inventory.

## Hardware backend boundary

`gem16gb` targets the approximately 16 GB NVIDIA CUDA GPU class. Architecture-specific kernels and dispatch live
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
selection; the reference head is restricted to tests and characterization probes.

The model-specific sequence is attention normalization and FP8 projections, specialized local/global attention,
then NVFP4 MLP projections and residual updates. This sequence now exists both as an independent Layer-0 comparison
probe and as an unfused, batch-one 48-layer greedy characterization. The latter loads the complete text-only model
into one aligned arena, keeps separate K/V state and reusable workspace allocations fixed for the run, applies the
tied BF16 embedding/output matrix, exact logit softcap, and GPU candidate reduction, and performs no token-loop allocation. It
uses one full graph replay for ordinary greedy decode but is not yet benchmark-qualified.

The first full-model path intentionally accepts token IDs and uses a hybrid cache through the checkpoint's 262,144
position contract. Its 40 local-attention layers use fixed 1,024-token rings; its eight full-attention layers use
absolute, growing storage. Checkpoint-FP8 prefill uses one fixed 1,024-token chunk. Attention projections run in
256-thread M64xN64xK64 FP8 CTAs with two exact `cp.async` stages for source-layout activations and weights; local
Q/K/V and global Q/K share one grouped launch. Decode uses the same binding-dimension grouping around the
latency-oriented T=1 direct-source kernel, reducing three independent graph nodes to one while retaining each
projection's original CTAs and MMA ordering. NVFP4 SM120 MMA is batched across tokens and reuses each NVFP4
Row8/K64 weight fragment across eight consecutive 16-token MMA tiles,
and groups eight NVFP4 warps into an M128xN64 CTA. Two shared-memory stages cooperatively transfer the CTA's exact
packed activation bytes and E4M3 scale words with `cp.async`, overlapping the next K64 slice with the current MMA
stack. Packed E2M1 weights and their E4M3 scales are transformed byte-exactly into the sole final Row8/K64 device
layout at model load; no raw GPU copy survives. This replaces strided per-row weight loads without changing any
nibble, scale, global divisor, or FP32 K accumulation. It evaluates local D256 and global D512
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
generated tokens continue through the ordinary decode graph. A separate parallel prefill graph and sampling plans
remain required production components.

The reusable `ChatMessage`, `Tokenizer`, and `GemmaChatProcessor` interfaces are deliberately independent of
terminal I/O. A future OpenAI-compatible Chat Completions server can reuse this request-to-token boundary; HTTP,
JSON request schemas, streaming, and session scheduling are not part of the current CLI milestone.

The greedy plan copies checkpoint `suppress_tokens` into fixed workspace before prompt processing and stops on any
checkpoint EOS token. Optional full-logit capture preallocates host storage before the token loop and writes raw
little-endian float32 only after generation; it is a correctness diagnostic and invalidates timing comparisons.

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

The production FP8 prefill projection is a 256-thread M64xN64xK64 CTA. It double-buffers two K32 fragments at a
time, cooperatively copying 64 activation rows and 64 source-layout weight rows into 17,408 bytes of static shared
memory before issuing FP32-accumulating E4M3 MMA. Four M16 tiles reuse each staged weight fragment. The kernel uses
60 registers with zero stack/local memory. Local Q/K/V and global Q/K are grouped through the kernel's binding
dimension, reducing FP8 projection launches from 184 to 96 per context-512 prefill without changing any output;
there is no grouped/ungrouped runtime selector.

Checkpoint-FP8 prefill attention uses BF16 Tensor-Core QK and probability-times-V operations with FP32 online
softmax state. This deliberately changes the scalar FP32 reference's reduction tree and rounds MMA operands, so it
is qualified with local/global operator error distributions plus vLLM model-logit gates rather than bit identity.
The local kernel uses 32-query by 32-key tiles; the global kernel uses 16-query by 16-key tiles. Both consume the
physical E4M3 K/V bytes and checkpoint BF16 scales without a persistent conversion or fallback.

The production NVFP4 prefill projection uses 256-thread M128xN64xK64 CTAs. Its two shared activation stages are a
temporary 9,216-byte payload within the kernel allocation, not an arena or checkpoint-layout copy. Gate, Up, and
Down read packed E2M1 weights directly from source order and read local scales from
`[8 output rows][K64 block][row][4 E4M3 scales]`. The selected prefill kernel uses 128 registers and 10,240 total
static shared bytes including toolchain overhead; the decode kernel uses 40 registers. Both have zero stack/local
memory. The scale transformation preserves all bytes, leaves the 9,200,135,680-byte weight arena unchanged, and
is mandatory rather than selectable.

Prefill materializes no redundant normalized or MLP-product tensor solely to cross a quantization boundary.
Shape-specific kernels combine RMSNorm with the exact BF16 cast and dynamic FP8/NVFP4 token quantizer, combine
post-projection normalization with residual and optional BF16 layer scaling, and combine the separate Gate/Up
projection outputs with the exact BF16/GELU-tanh/product boundaries and Down-input NVFP4 packing. These kernels
preserve the former operation order and bytes exactly; the unfused sequence remains only as the CUDA test oracle.
There is no fused/unfused runtime selector. Diagnostic decode can still request the normalized intermediate when
capturing hidden states, while ordinary prefill omits that store.

Q/K prefill uses another exact closed boundary. One CTA per token/head performs the projection-output BF16 cast,
the original 256-thread RMSNorm reduction, normalized BF16 cast, RoPE, and the post-RoPE BF16 cast. Q and K share
the launch but never share reduction state. Local D256 and proportional global D512 cosine/sine tables use the
former double-precision `pow`/`cos`/`sin` expressions and are generated once during initialization for every
position in the planned context. The tables cost 1,536 bytes per context token and eliminate per-layer
trigonometry. The hot fused kernel uses 35 registers, 3,072 bytes shared memory, and zero stack/local memory; the
initialization-only table kernel is outside prompt timing. The unfused sequence remains only in CUDA tests.

## Memory-plan boundary

The first runtime component now converts parsed model metadata and the authoritative text-only manifest into a
deterministic 256-byte-aligned base arena. It places immutable weights/model state, scales, and the selected KV
payload in named regions with checked offsets. The required separate K/V size and a diagnostic one-state lower
bound are retained in every result; shared physical cache selection is rejected.

The greedy characterization uses an execution workspace containing hidden-state ping-pong, quantized activations
and scales, projection intermediates, retained full logits, a 32 KiB fused-output candidate array, and GPU argmax
state. The checkpoint-FP8 prefill arena contains no attention-score region; only the explicit BF16 correctness
mode retains the scalar attention score workspace. Exact sizes are reported per run.
Its default hybrid cache stores physical E4M3FN bytes with checkpoint BF16 scales; an explicit float32
BF16-semantics diagnostic allocation remains available. The general planner remains conservative until production
prefill, graph, and sampling shapes are defined.
