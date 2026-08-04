# Executive architecture

## System split

The implementation has three independently testable products:

1. **Source and evidence layer**
   Immutable source locks, tensor inventories, reference outputs and calibration/test corpora.

2. **Deterministic checkpoint compiler**
   Converts one exact BF16 source revision into one exact mixed-precision text-only Safetensors artifact. It is offline, versioned and reproducible. It does not run in the inference process.

3. **gem16 runtime**
   Validates and loads the compiled artifact into final device layouts, allocates KV/workspace/graphs once, and executes text inference without per-token allocation.

This split is mandatory. The runtime must not hide conversion, and the compiler must not claim runtime performance.

## Initialization flow

```text
locked source or compiled checkpoint directory
        │
        ├─ config/tokenizer/generation/template verification
        ├─ gem16 compilation manifest verification
        ├─ tensor manifest construction
        ├─ model variant selection
        ├─ text-only residency filtering
        ├─ exact arena planning
        ├─ streamed source-layout → final device-layout upload
        ├─ per-tensor prefix/suffix probes
        ├─ KV/workspace/graph allocation
        └─ capability and memory admission
```

Model selection occurs once. The 12B dense-unified and 26B MoE paths use separate statically specialized plans.

## 26B layer execution

The reference ordering must match the current Hugging Face Gemma 4 definition:

```text
residual_0 = hidden
x = input_norm(hidden)
a = attention(x)
a = post_attention_norm(a)
hidden = residual_0 + a

residual_1 = hidden

shared = pre_feedforward_norm(hidden)
shared = dense_shared_mlp(shared)
shared = post_feedforward_norm_1(shared)

router_input = router_norm_and_scale(residual_1)
p, top8_weight, top8_id = router(router_input)

expert = pre_feedforward_norm_2(residual_1)
expert = routed_experts(expert, top8_id, top8_weight)
expert = post_feedforward_norm_2(expert)

combined = shared + expert
combined = final_post_feedforward_norm(combined)
hidden = residual_1 + combined
hidden *= layer_scalar
```

The router operates on the residual before the shared MLP input norm. It performs FP32 softmax over all 128 experts, top-8 selection, renormalization of the selected probabilities, and multiplication by per-expert learned scales.

## Decode dataflow

For one token:

```text
embedding lookup
  → 30 × {
       attention projections and attention
       shared dense NVFP4 MLP
       BF16 router + deterministic top-8
       routed NVFP4 W13 for 8 experts
       activation
       routed NVFP4 W2 with weighted reduction
       residual/norm boundaries
     }
  → final norm
  → quantized tied output head
  → softcap
  → candidate reduction
  → sample/argmax
```

All routing remains on device. Expert IDs must not return to the CPU. A complete greedy decode step is captured in a whole-model CUDA Graph after addresses are fixed.

## Prefill dataflow

For a bounded prompt chunk:

```text
attention prefill
  → shared dense MLP
  → router for every token
  → [token, rank, expert, weight] assignment generation
  → expert histogram
  → prefix sum
  → stable token permutation
  → grouped W13
  → activation
  → grouped W2
  → weighted inverse scatter/reduction
  → residual/norm closure
```

The chunk size is a memory and performance tuning parameter. The implementation may not allocate an activation tensor for every token × every expert.

## Weight layouts

### Source artifact

The derived checkpoint should remain Safetensors-based and self-describing. Quantized values and scales are stored once. The default artifact should favor interoperability and auditability over an opaque GPU binary.

### Device artifact

During load, packed NVFP4 values and E4M3 scales are streamed into the final SM120 layout already used by `gem16` or a documented MoE extension of that layout. No source-order device copy survives.

Expert-major addressing should permit:

```text
[layer][expert][projection][row tile][K block]
```

Gate/up may be fused into W13 if validation proves an exact one-copy representation. Down remains W2.

## Embedding and output head

The tied matrix is a separate format decision:

- Q4_0 offers the closest path to Google's QAT target format.
- NVFP4 offers native block-scaled hardware and reuse of existing machinery.
- Input lookup does not need Tensor Cores.
- Decode output-head behavior is often memory-bound and must be measured, not inferred from peak FLOPS.

Only one representation is resident in a production profile. The loader must reject profiles that accidentally contain both.

## Memory ownership

```text
ModelRuntime
  ├─ immutable target weight arena
  ├─ immutable tables and tensor bindings
  └─ no MTP/vision arena in first release

ExecutionSlot
  ├─ separate FP8 K/V cache
  ├─ bounded prefill workspace
  ├─ decode activations
  ├─ routing/top-8 workspace
  ├─ output-head/sampling workspace
  └─ CUDA Graph-private state

SessionState
  ├─ exact token prefix
  ├─ RNG/repetition state
  └─ logical conversation metadata
```

The server's admission controller must use the measured larger of named allocator bytes and `cudaMemGetInfo` delta, not a constant inherited from the 12B path.

## Why not a generic framework

The hot path is shape- and model-specific:

- 30 fixed layers;
- 25 local and 5 global attention layers;
- 128 experts and top-8;
- hidden 2816;
- expert intermediate 704;
- shared intermediate 2112;
- vocabulary 262144;
- local/global head dimensions 256/512.

A small model-traits layer is useful. Runtime graph abstraction inside every layer is not. The preferred architecture is two independently compiled backends selected once at load.
