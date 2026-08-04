# Example `gem16_compilation.json`

This example is illustrative. M03/M08 must replace tensor names and exact dimensions with validated data.

```json
{
  "schema_version": 1,
  "artifact_profile": "sm120-text-hybrid-v1",
  "model_variant": "gemma4-26b-a4b",
  "text_only": true,
  "source": {
    "repository": "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
    "revision": "<full immutable commit>",
    "lock_sha256": "<sha256>"
  },
  "compiler": {
    "repository": "Danmoreng/gem16",
    "commit": "<commit>",
    "dirty": false,
    "profile_sha256": "<sha256>",
    "toolchain_lock_sha256": "<sha256>"
  },
  "formats": {
    "attention": "fp8-per-channel-v1",
    "shared_mlp": "nvfp4-group16-divisor-v1",
    "routed_experts": "nvfp4-group16-divisor-v1",
    "router": "bf16-source",
    "tied_embedding_head": "q4_0-v1"
  },
  "capabilities": {
    "text": true,
    "vision": false,
    "audio": false,
    "video": false,
    "mtp": false
  },
  "omitted_tensor_groups": [
    {
      "name": "vision",
      "reason": "text-only 16GB profile",
      "source_tensor_count": 0,
      "source_bytes": 0
    }
  ],
  "tensors": [
    {
      "name": "model.language_model.embed_tokens.weight_q4_0",
      "role": "tied_embedding_and_output",
      "storage_dtype": "U8",
      "physical_shape": [415236096],
      "logical_shape": [262144, 2816],
      "byte_length": 415236096,
      "sha256": "<sha256>",
      "source_tensors": [
        "model.language_model.embed_tokens.weight"
      ],
      "transformation": {
        "name": "q4_0-v1",
        "block_size": 32,
        "scale_dtype": "F16"
      },
      "aliased": true,
      "disk_layout": "q4_0-reference-block-order",
      "runtime_layout": "same"
    }
  ],
  "totals": {
    "tensor_count": 0,
    "payload_bytes": 0,
    "resident_bytes": 0,
    "omitted_source_bytes": 0
  }
}
```

## Required runtime checks

- artifact/profile/schema supported;
- source/compiler hashes present;
- every listed tensor exists and matches;
- no unlisted payload tensor;
- tied alias is unique;
- omitted modality not present;
- totals reconcile;
- quantizer parameters supported.
