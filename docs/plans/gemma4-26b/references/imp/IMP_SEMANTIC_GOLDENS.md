# imp-derived semantic golden plan

## Normative hierarchy

1. Pinned official Google/Transformers Gemma 4 implementation and checkpoint metadata.
2. gem16 CPU oracle.
3. Independent runtime references, including pinned imp and a trusted framework runtime.

Imp is valuable because it contains documented failure modes. It is not allowed to override the official model contract merely because its output differs.

## Golden fixture families

### Router input transform

Capture at layers 0, 5, 15, 28 and 29:

- post-attention residual before either FFN branch;
- scale-free RMSNorm denominator/output;
- learned router input scale application;
- `1/sqrt(2816)` application;
- FP32 router input;
- 128 FP32 logits;
- full softmax probabilities;
- selected top-8 IDs and pre/post-renormalization weights;
- per-expert scale application.

Include adversarial synthetic vectors with near ties at positions 7–9 and exact ties.

### Shared and routed branch separation

Capture independently:

- shared pre-norm output;
- shared MLP gate/up/down output;
- shared post-norm output;
- expert pre-norm output;
- each selected expert gate/up/down output;
- weighted expert reduction in FP32;
- expert post-norm output;
- combined branch output;
- final post-FFN norm/residual/layer-scalar boundaries.

### Precision probes

For the same input, retain four controlled arms:

```text
FP32 residual + FP32 router
BF16 residual + FP32 router
BF16 residual + BF16 router
production quantized expert path
```

The purpose is to identify where expert IDs diverge, not to promote a high-precision production path by default.

### Quantization producer semantics

Create tiny and real-shape fixtures for:

```text
llm-compressor: real = fp4 * local_scale / global_divisor
ModelOpt:       real = fp4 * local_scale * tensor_multiplier
```

Use deliberately non-unit global scales so a reversed interpretation fails loudly.

### Tied head

Capture Q4_0, NVFP4 and BF16 reference logits for:

- ordinary hidden states;
- low-norm states;
- high-dynamic-range states;
- near-argmax ties;
- repeated tokens and stop-token boundaries.

## Acceptance

A semantic fixture is accepted only when:

- source revision, token IDs, dtype boundaries and tensor hashes are retained;
- official reference and local CPU oracle agree within frozen tolerance;
- imp either agrees or the discrepancy is understood and documented;
- the fixture fails under at least one intentional mutation of the relevant formula/order.
