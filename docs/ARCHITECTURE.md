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
absolute, growing storage. Checkpoint-FP8 prefill uses one fixed 1,024-token chunk, batches direct-source FP8 and
NVFP4 SM120 MMA across tokens, reuses each projection weight fragment across two consecutive 16-token MMA tiles,
and evaluates local D256 and global D512 causal attention with shape-specific online Tensor-Core kernels. Those
kernels stage current-chunk K/V directly, read older positions from the circular or growing cache, retain row max,
normalization sum, and output accumulators in FP32, and never materialize a global score matrix. K/V is committed
only after the layer's attention finishes. The serial path remains a test/probe oracle and is not a runtime option. The pure C++
`GemmaChatProcessor` loads the checkpoint vocabulary, merge ranks, byte fallback, generation controls, and exact
pinned Jinja artifact. It implements the supported text-only behavior of that template natively and rejects a
different template revision rather than silently approximating it. This makes real chat flows testable now while
preserving a narrow execution contract. A separate parallel prefill graph, circular local cache, growing global
cache storage and sampling plans remain required production components.

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
2. A loader-owned weight view prefers the source Safetensors layout directly. If measurement proves a final
   architecture-specific layout necessary, a streamed transformation may replace it without changing any
   quantized code or scale and without retaining a second device copy.
3. Operator-owned decode and prefill plans select only explicitly qualified implementations for an exact shape and
   token extent. A correctness route, packed SIMT/GEMV route, and native SM120a MMA route are distinct capabilities;
   none may silently stand in for another.

For the pinned checkpoint, Gate and Up have identical input and weight global divisors in all 48 layers. The native
decode plan may therefore quantize their shared input once, contract both matrices, and apply Gemma's GELU-tanh
product in one closed operator. Down performs its own dynamic-local quantization and may fuse its residual epilogue.
The attention projections remain a separate dynamic-FP8/per-channel-FP8 path.

Checkpoint-FP8 prefill attention uses BF16 Tensor-Core QK and probability-times-V operations with FP32 online
softmax state. This deliberately changes the scalar FP32 reference's reduction tree and rounds MMA operands, so it
is qualified with local/global operator error distributions plus vLLM model-logit gates rather than bit identity.
The local kernel uses 32-query by 32-key tiles; the global kernel uses 16-query by 16-key tiles. Both consume the
physical E4M3 K/V bytes and checkpoint BF16 scales without a persistent conversion or fallback.

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
