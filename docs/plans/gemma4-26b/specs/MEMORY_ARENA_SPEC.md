# Memory arenas and residency specification

## Ownership levels

### ModelRuntime: process-wide immutable

Owns:

- compiled target weights;
- scale tensors;
- runtime-tiled weight layout;
- immutable RoPE tables if shared;
- optional model metadata copied to device.

One allocation family per process. Sessions never duplicate weights.

### ExecutionSlot: concurrently active request

Owns:

- reusable prefill buffers;
- router/assignment/permutation buffers;
- decode activations;
- output candidates/logits/sampling;
- CUDA stream and graph-private allocations;
- mapped streaming ring;
- cancellation/control records.

### SessionState: conversation

Owns:

- KV cache;
- exact cached token/media prefix;
- sampling history/RNG;
- position and response-channel state.

The implementation may combine SessionState and ExecutionSlot physically if server ownership remains explicit. Metrics must still distinguish them.

## Immutable weight target

Core arithmetic from the planning estimate:

| Family | Bytes |
|---|---:|
| Routed experts NVFP4 | 12,846,366,720 |
| Shared MLP NVFP4 | 301,086,720 |
| Attention FP8 weights | 1,110,179,840 |
| Tied Q4_0/NVFP4 head | 415,236,096 |
| Router BF16 | 21,626,880 |
| Core subtotal | 14,694,496,256 |

Small scales, norms, layer scalars, metadata and alignment must be added from the real manifest.

Program thresholds:

- target: `≤ 14,100 MiB`;
- warning/review: `14,101–14,300 MiB`;
- hard stop: `> 14,300 MiB`.

These are engineering gates, not promises before M08/M09 measured accounting.

## Named regions

Minimum immutable regions:

```text
tied_embedding_head
final_norm_and_small_state
attention_fp8_weights
attention_fp8_scales
shared_mlp_nvfp4_weights
shared_mlp_nvfp4_scales
routed_expert_nvfp4_weights
routed_expert_nvfp4_scales
router_bf16
```

Minimum mutable regions:

```text
kv_local_k
kv_local_v
kv_global_k
kv_global_v
decode_hidden_a
decode_hidden_b
router_logits
router_top8
expert_w13_product
expert_w2_partial_or_reduction
shared_mlp_product
prefill_assignments
prefill_histogram_prefix
prefill_permutation
prefill_activation_values_scales
cutlass_workspace
output_candidates
sampling_workspace
graph_private
device_control
```

## Alignment

- checked power-of-two alignment;
- 256-byte default arena alignment unless kernels require more;
- per-region offsets reported;
- padding included in totals;
- integer overflow rejected.

## One final weight layout

For any tensor:

```text
source mapped file
→ bounded host staging
→ final device layout
```

No persistent raw device copy plus tiled device copy.

Temporary prefill conversion scratch is mutable and reusable. It is not a second persistent checkpoint layout.

## Allocation phase

All normal CUDA allocations occur during:

1. model runtime load;
2. slot/session creation;
3. graph preparation.

No allocations during prompt chunks or token decode.

Library calls must be traced because CUTLASS/CUB/CUDA Graph can allocate indirectly.

## Device admission

Before listening/serving:

- query total/free VRAM;
- account for current CUDA context;
- allocate/probe a representative slot if necessary;
- require configured slot count plus reserve;
- fail visibly.

Recommended release reserves:

- 32K: at least 700 MiB;
- qualified 64K: at least 500 MiB;
- startup reserve is based on measured process peak, not only named payload.

## Metrics

Expose:

```text
model_weight_bytes
model_scale_bytes
tied_head_bytes
kv_bytes
prefill_workspace_bytes
decode_workspace_bytes
graph_device_bytes
slot_reserved_bytes
device_total_bytes
device_free_before_load
device_free_after_load
admission_margin_bytes
persistent_repack_bytes
fallback_count
token_loop_allocations
```

## Host memory

Compiler and runtime host responsibilities differ.

Runtime may memory-map source files and use bounded staging. It must not retain all BF16 source weights after upload.

Record peak RSS during load. Failure to fit host RAM is a product issue even when VRAM fits.

## Tests

- arithmetic and overflow;
- allocation reconciliation;
- repeated load/unload;
- multiple slots;
- failed allocation cleanup;
- CUDA API allocation trace;
- process VRAM telemetry;
- 12B unchanged ownership.
