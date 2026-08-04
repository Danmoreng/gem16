# Model variant traits and static dispatch specification

## Purpose

`gem16` must support Gemma 4 12B Unified and Gemma 4 26B A4B without turning the runtime into a generic graph framework and without scattering size checks through hot code.

The design separates:

1. **parsed checkpoint facts**;
2. **validated model contracts**;
3. **immutable execution traits**;
4. **architecture-specific kernel plans**.

A model is parsed dynamically and validated exactly once. After validation, the runtime selects a closed implementation family with compile-time constants where those constants improve code generation.

## Required variant identifiers

Suggested internal identifiers:

```cpp
enum class ModelVariant {
  kGemma4Unified12B,
  kGemma4A4B26B,
};

enum class WeightProfile {
  kDirectUnslothMixedFp8Nvfp4,
  kGem16QatHybridQ4Head,
  kGem16QatHybridNvfp4Head,
  kReferenceQ4_0,
};
```

Public model names must be stable and must not imply endorsement. Suggested shape:

```text
gem16-gemma4-12b-unified-nvfp4
gem16-gemma4-26b-a4b-qat-hybrid-q4head
gem16-gemma4-26b-a4b-qat-hybrid-nvfp4head
```

## Parsed facts

Extend `ModelConfig` so it can represent both variants without interpretation loss. At minimum:

```cpp
struct ModelConfig {
  // Existing fields...
  std::uint64_t shared_intermediate_size;
  std::uint64_t moe_intermediate_size;
  std::uint64_t expert_count;
  std::uint64_t top_k_experts;
  bool enable_moe_block;

  std::uint64_t router_projection_rows;
  std::uint64_t hidden_size_per_layer_input;
  std::uint64_t vocabulary_size_per_layer_input;

  std::vector<PerLayerAttentionConfig> per_layer_attention;
  std::vector<std::string> layer_types;
};
```

Do not rely on field names in this proposal. The pinned `config.json` is authoritative. If the actual config uses a different nesting or naming scheme, parse and preserve it explicitly.

## Validated traits

After contract validation, construct an immutable value:

```cpp
struct ModelTraits {
  ModelVariant variant;
  std::uint32_t layer_count;
  std::uint32_t hidden_size;
  std::uint32_t vocabulary_size;
  std::uint32_t sliding_window;
  std::uint32_t max_positions;

  std::uint32_t query_heads;
  std::uint32_t local_kv_heads;
  std::uint32_t global_kv_heads;
  std::uint32_t local_head_dim;
  std::uint32_t global_head_dim;

  std::uint32_t shared_intermediate_size;
  std::uint32_t expert_count;
  std::uint32_t top_k_experts;
  std::uint32_t expert_intermediate_size;

  bool attention_k_eq_v;
  bool tied_embeddings;
  bool has_moe;
  bool supports_media;
  bool supports_mtp;

  std::vector<LayerTraits> layers;
};
```

`LayerTraits` must name the attention type and cache ownership explicitly:

```cpp
enum class AttentionType { kSliding, kFull };
enum class KvSource { kOwnedProjection, kSharedFromLayer };

struct LayerTraits {
  AttentionType attention;
  std::uint32_t head_dim;
  std::uint32_t kv_heads;
  bool stores_v_projection;
  KvSource kv_source;
  std::int32_t kv_producer_layer;
};
```

## 12B invariants

The existing 12B path keeps its exact contract, including:

- 48 layers;
- hidden size 3840;
- dense intermediate size 15360;
- 16 query heads;
- 8 local KV heads and 1 global KV head;
- D256 local and D512 global;
- five sliding layers followed by one full layer;
- mixed FP8 attention and NVFP4 dense MLP;
- tied BF16 embedding/head in the existing checkpoint;
- current multimodal and MTP support.

Do not route 12B through the new 26B MoE kernels.

## 26B invariants

The pinned 26B model contract is expected to validate:

- 30 layers;
- hidden size 2816;
- vocabulary 262,144;
- shared dense intermediate 2112;
- 128 routed experts;
- top-8 experts;
- expert intermediate 704;
- 16 query heads;
- 8 local KV heads;
- 2 global KV heads;
- D256 local and D512 global;
- 25 sliding and 5 full layers;
- tied embedding/head;
- `attention_k_eq_v = true`;
- text-only production profile;
- no MTP in the first release.

These values must be validated against the immutable source lock. They are not permission to ignore a changed source.

## Static dispatch

Preferred pattern:

```cpp
template <typename Traits>
Status ExecuteDecode(ExecutionState&);

using Gemma4_12B = StaticTraits<...>;
using Gemma4_26B = StaticTraits<...>;

switch (runtime.traits.variant) {
  case ModelVariant::kGemma4Unified12B:
    return ExecuteDecode<Gemma4_12B>(state);
  case ModelVariant::kGemma4A4B26B:
    return ExecuteDecode<Gemma4_26B>(state);
}
```

The switch occurs at plan construction or a coarse execution entry point, not inside every layer or inner loop.

For layer-dependent local/global shapes, build immutable arrays of function pointers or graph nodes during initialization. Avoid virtual dispatch in CUDA hot paths.

## Contract validation

Validation must fail before device allocation on:

- unknown architecture/model type;
- wrong layer count or layer pattern;
- missing expert/router dimensions;
- inconsistent top-k;
- unsupported per-layer input features;
- unsupported quantization schema;
- missing or unexpected tensors;
- unsupported modality requirements;
- incompatible tied-head format;
- unrecognized compiled-artifact schema.

Errors must include:

```text
model variant
field or tensor
actual value
expected value
source lock/revision
```

## Capability separation

Model architecture and product profile are separate concepts.

A checkpoint can describe image/audio support while the text-only compiled artifact deliberately omits those tensors. `supports_media` in the execution profile must therefore be false even though the parent architecture is multimodal.

Similarly, a target architecture may theoretically support MTP while the selected profile has no compatible assistant.

## Tests

Required host tests:

- parse and validate both pinned configs;
- reject one-field mutations for every invariant;
- build exact per-layer trait tables;
- serialize traits into diagnostic JSON;
- ensure 12B and 26B choose distinct execution families;
- prove public capability output follows the selected profile, not architecture metadata alone.

## Prohibited design

Do not:

- replace all dimensions with runtime integers inside every kernel;
- copy 12B constants into 26B branches;
- infer the variant from model directory names;
- infer local/full layers from index modulo without validating `layer_types`;
- advertise media or MTP based only on architecture metadata;
- add a catch-all “Gemma4” fallback.
