# API, CLI and product-surface change specification

## Model selection

Add explicit model-profile discovery. The existing `--model <directory>` remains valid, but diagnostics expose:

```text
variant=gemma4-26b-a4b
weight_profile=gem16-qat-hybrid-v1
head_format=q4_0
text_only=true
supports_mtp=false
supports_vision=false
```

Do not infer user-visible capability from architecture alone.

## CLI options

Existing options continue to work where supported:

```text
--model
--max-context
--kv-cache
--temperature
--top-k
--top-p
--repetition-penalty
--seed
```

26B-specific diagnostic options may include:

```text
--print-model-profile
--print-memory-plan
--require-native-path
--head-format-experimental <q4_0|nvfp4>
--correctness-reference
```

Experimental options must never be enabled implicitly in performance commands.

## Unsupported options

For first release:

- assistant model/MTP draft count;
- image/audio/video input;
- multimodal processor behavior.

Failure message example:

```text
Gemma 4 26B A4B profile gem16-qat-hybrid-v1 is text-only and has no
qualified MTP assistant. Remove --assistant-model/--mtp-draft-tokens.
```

Do not ignore unsupported input.

## Result metadata

Extend inference/benchmark results with:

```cpp
ModelVariant model_variant;
WeightProfile weight_profile;
TiedWeightFormat tied_weight_format;
std::uint64_t router_weight_bytes;
std::uint64_t expert_weight_bytes;
std::uint64_t shared_mlp_weight_bytes;
std::uint64_t attention_weight_bytes;
std::uint64_t tied_head_bytes;
std::uint64_t prefill_assignment_bytes;
bool native_moe_decode;
bool native_moe_prefill;
bool text_only;
```

Serialization remains backwards-compatible through additive JSON fields.

## `/health`

Expose:

```json
{
  "model_profile": "...",
  "model_variant": "...",
  "text_only": true,
  "supports": {
    "text": true,
    "vision": false,
    "audio": false,
    "video": false,
    "mtp": false
  },
  "max_context": 32768,
  "resident_weight_bytes": 0,
  "slot_bytes": 0,
  "admission_margin_bytes": 0,
  "native_paths": {
    "fp8_attention": true,
    "nvfp4_moe_decode": true,
    "nvfp4_moe_prefill": true
  }
}
```

## `/v1/models`

Use a stable product ID and include optional metadata without breaking OpenAI SDK parsing.

Do not claim `owned_by: google` or `owned_by: unsloth` for the derived artifact. Use project ownership/derivation wording.

## Chat Completions and Responses

26B must preserve:

- text/reasoning stream separation;
- tools and tool results if template-compatible;
- sampling;
- cancellation;
- resident continuation;
- usage/cache metrics;
- ordered content validation.

Media content parts cause a clear request error.

## Studio

Display:

- download size and disk requirement;
- recommended 16 GB Blackwell GPU;
- 32K default context;
- 64K status if qualified;
- text-only badge;
- model provenance/derived status;
- resident memory warning;
- unsupported MTP/media.

Studio must not reuse 12B capability flags.

## Backward compatibility

- 12B model IDs remain valid;
- existing API fields remain;
- new fields are additive;
- 12B Studio/default settings unchanged;
- server can host one selected target profile per process unless a new multi-model decision is made.

## Security

- model IDs cannot select arbitrary filesystem paths through API;
- downloads verify locks;
- health output does not expose private tokens/paths;
- request media decoding stops before large allocations when unsupported.

## Tests

- CLI help and invalid combinations;
- additive JSON parsing;
- OpenAI SDK smoke;
- unsupported media/MTP;
- capability output;
- model ID selection;
- Studio state migration;
- 12B compatibility.
