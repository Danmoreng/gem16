# Memory

The deterministic base-arena planner is implemented. The verified checkpoint
contains 9,304,786,336 tensor payload bytes. The original text-only inventory
classified 9,200,026,528 bytes and 104,759,808 bytes of audio/vision projection
and embedding tensors. The runtime now keeps both classes resident. The planner
separates 8,772,780,320 bytes of weights/model state from 532,006,016 bytes of
scales and aligns every named region to 256 bytes. This is manifest payload
accounting; aligned arenas, CUDA-private allocations, and sampled process use
are reported separately.

For the parsed 48-layer architecture and one-byte FP8 cache, the formula after the local window is full is:

```text
local one-state lower bound = 40 * min(tokens, 1024) * 8 * 256
global one-state lower bound = 8 * tokens * 1 * 512
required separate K and V   = 2 * one-state lower bound
```

At 64K, the one-state lower bound is 336 MiB and the required separate K/V payload is 672 MiB. Although
`attention_k_eq_v=true` reuses the raw full-attention K projection for V, learned K normalization plus RoPE and
scale-free V normalization produce distinct final cache states. Shared physical storage is therefore rejected.
These are formulas, not allocator measurements. Metadata, scale storage, alignment, CUDA context, workspaces, and
graph pools are additional named or measured costs and are not included in the
cache formula.

The one-byte FP8 payload plans are:

| Profile | Context | One-state lower bound | Required separate K/V | Invalid shared arena | Selected separate arena |
|---|---:|---:|---:|---:|---:|
| `interactive` | 8,192 | 112 MiB | 224 MiB | 8,885.83 MiB | 8,997.83 MiB |
| `standard` | 32,768 | 208 MiB | 416 MiB | 8,981.83 MiB | 9,189.83 MiB |
| `long` | 65,536 | 336 MiB | 672 MiB | 9,109.83 MiB | 9,445.83 MiB |
| `xlong` | 131,072 | 592 MiB | 1,184 MiB | 9,365.83 MiB | 9,957.83 MiB |
| `max` | 262,144 | 1,104 MiB | 2,208 MiB | 9,877.83 MiB | 10,981.83 MiB |

Every plan reports both byte formulas for auditability, requires an explicit layout, and accepts only `separate`.
Checked multiplication, addition, and alignment reject integer overflow. The base-arena table intentionally
excludes activation, logits, sampling, CUDA Graph, kernel, media, server-slot, and prefill workspaces. Production
plan construction accounts those regions separately; `total_arena_bytes` in this table is not a peak-VRAM claim.

The current unified Target allocation is 9,304,895,488 bytes and includes all text, audio, and vision tensors in
one fixed-address arena. The optional official MTP assistant is held in a distinct fixed-address 845,714,944-byte
BF16 arena containing its exact 845,713,928-byte source payload and 1,016 bytes of 256-byte alignment padding. The
loader keeps no second device layout and probes the uploaded prefix and suffix of every one of its 48 tensors.
`cudaMemGetInfo` measured 847,249,408 additional bytes across assistant loading. At context 128,
sequential 50 ms `nvidia-smi` polling observed 9,660 MiB total GPU use for target-only and 10,468 MiB for target
plus assistant, an 808 MiB increment. The assistant proposal workspace is 289,024 bytes at context 128, including
a 16-head FP32 score view, recurrent 3,840-wide feedback, layer intermediates, and output candidates. Active
batched verification additionally reserves fixed tentative/backup target K/V rows and five-row output selection:
assistant-plus-verifier workspace measures 2,213,376 bytes with FP8 KV and 7,374,336 bytes with BF16 KV at context
128. FP8 mode reserves the larger of the reference-score and split-online attention requirements. The additional
fixed regions hold GPU acceptance, stop IDs, and one committed hidden row. The qualified 24,576-context D2 run
peaks at 10,838 MiB under 200 ms telemetry. The final corrected 16K/1,135-token cross-engine command at `a819d14c`
peaks at 11,746 MiB for gem16, leaving approximately 4,557 MiB relative to the reported 16,303 MiB device total.
That command includes resident Target and assistant weights, fixed-D2 graphs, unified model tensors, and the final
chunk plan, but uses greedy selection; it does not replace S08's pending direct all-regions sampling/media reserve
record. No assistant KV cache or second weight layout is allocated.

GPU-chained fixed D2 additionally reserves three `uint32_t` entries per context position: one verified-output slot
and two proposal slots, plus a small aggregate record. The same payload is allocated once in pinned host memory for
the post-chain transfer. This is 294,912 bytes on the Wikipedia benchmark's 24,576-position plan and 3,145,728 bytes
at the 262,144-position maximum, plus alignment and the aggregate. The conditional executable raises measured
graph-associated device memory from 14,680,064 to 20,971,520 bytes on the 32K profiling plan.

The asynchronous SPSC ring adds one fixed 1,088-byte mapped pinned allocation: four 64-bit producer/consumer,
cancellation, and backpressure counters plus 256 token IDs and alignment. It has no token-loop allocation. Adding
the dependent ordinary-tail conditional raises 32K graph-associated device memory to 23,068,672 bytes.

The runtime applies the qualified hybrid layout: 40 local layers allocate
at most 1,024 physical slots and reuse them as chronological rings, while eight global layers allocate the requested
context extent. Separate K/V storage is retained for both. Optional
full-logit diagnostics use host memory (`steps * 262144 * 4` bytes) allocated before generation and do not change
persistent device storage.

A real FP8 allocation at a 1,026-position execution plan measured 176,177,152 cache bytes: 167,772,160 bytes for the
40 local K/V rings plus 8,404,992 bytes for eight global K/V extents. This exactly matches the formula above and
crosses the first local-ring wrap in full-model execution.

The native prefill arena is allocated once during engine initialization. Checkpoint-FP8 execution holds one fixed
2,048-token hidden/projection/MLP tile and no causal score matrix: local and global SM120 attention maintain online
softmax state inside their CTAs. Gate, Up, and Down additionally share preallocated temporary storage for one row-major
packed weight (29,491,200 bytes), its CUTLASS-interleaved scales (3,686,400 bytes), padded activation scales, and
an 8 MiB CUTLASS workspace. Accommodating Down's 15,360-element contracting dimension enlarges the reusable
activation-scale view by 1,474,560 bytes. The 8K execution plan measures 673,808,384 reusable workspace bytes
versus 630,276,096 before CUTLASS prefill, an increase of 43,532,288 bytes. Persistent weights, KV storage, and
`persistent_repack_bytes=0` are unchanged. The explicit BF16 correctness mode still uses the scalar attention
oracle and therefore retains the context-budgeted score matrix and deterministic 512 MiB score-budget selector.
The selected chunk size is reported in every inference result and never changes inside prompt processing. Final
prompt RMSNorm reserves five FP32 hidden rows (76,800 bytes) rather than one row per 8,192-token chunk
(125,829,120 bytes). Normalizing only the last row of the final ordinary prompt chunk therefore removes exactly
125,752,320 bytes from the named arena; the independent MTP verifier retains its five-row contract.

Explicit sampling conditionally adds adjusted/sorted vocabulary logits, input/sorted token IDs, CUB radix-sort
scratch, an in-place double-precision probability scan, and a 32-bit atomic repetition bitset to the deterministic
workspace. At context 128 this is 7,408,128 bytes above greedy. Disabled sampling allocates none of those regions. Both modes keep their whole-model graph at
fixed addresses and allocate nothing in the token loop.

The decode boundary and controlled Q/K fusion reuse the existing activation, quantization, RoPE-table, and graph
regions; they add no arena bytes. Before unified media residency, the context-8,192 plan reported 9,200,135,680
weight bytes, 236,978,176 KV bytes, and 674,200,064 reusable workspace bytes; a 20 ms `nvidia-smi` probe observed
9,852 MiB peak process VRAM. These retained values are historical execution evidence, not current unified-loader
accounting. Nsight records all CUDA allocations during initialization and none in prefill or the token loop.

Fresh single-run Wikipedia QA characterizations now exercise both extended profiles with the current production
path. At 131,072 prompt plus 256 output positions, the runtime reports 1,243,611,136 KV bytes and 870,823,424
workspace bytes; sampled total GPU usage peaks at 11,022 MiB. At 261,888 prompt plus 256 output positions, exactly
filling the 262,144-position model limit, it reports 2,315,255,808 KV bytes and 1,080,129,024 workspace bytes;
sampled total GPU usage peaks at 12,244 MiB. Both runs complete with no fallback or token-loop allocation. Those
historical `nvidia-smi` samples were taken on a device reporting 16,303 MiB nominal capacity and imply 5,281 MiB
and 4,059 MiB sampled headroom. Runtime admission and the 26B product gate instead use directly measured
CUDA-visible capacity (approximately 15,881 MiB on the reference machine) and require at least 700 MiB unused;
nominal or sampled headroom is not an allocation promise.

The always-resident multimodal loader adds the checkpoint's exact 104,759,808
audio/vision tensor bytes. Vision execution reserves 24,086,720 reusable bytes:
two 280x6,912 float patch views, two 280x3,840 float hidden views, and 560
`int32` position coordinates. These regions are allocated with the prefill arena
and are never allocated or resized in the token loop.

Server residency now accounts for immutable and mutable memory separately.
`ModelRuntime::weight_bytes()` and `assistant_weight_bytes()` describe the one
process-wide allocation. Every `ExecutionSlot` reports its own KV, reusable
workspace, assistant workspace, and graph-private bytes. Creating a slot from a
runtime copies only pointer bindings; it performs no checkpoint I/O, weight
upload, or persistent repack. Multiple simultaneous slots therefore scale with
their context plans, not with model-weight size.

The bounded server scheduler makes that scaling an explicit admission policy:
`--max-sessions` is the maximum number of resident mutable slots. Before
listening, the server creates one temporary slot and selects the larger of its
named allocator accounting and observed `cudaMemGetInfo` delta. The full
configured slot count must fit in currently free VRAM with one probe resident
and leave at least 700 MiB unused; otherwise startup fails. The probe is then
released. Runtime metrics export planned, configured, and resident slot bytes,
device capacity, and the enforced margin. Admission never evicts an active slot
and returns resource exhaustion if all slots are busy.
In the Windows A11 gate with FP8 KV, context 1,024, target plus MTP assistant,
the shared runtime reported 9,304,895,488 target bytes and 845,714,944 assistant
bytes; each most-recent execution slot reported 877,867,520 bytes. Two slots
generated concurrently without another checkpoint load. These counters are
exported at `/metrics`; they are allocator accounting, not a sampled whole-
process VRAM peak.
