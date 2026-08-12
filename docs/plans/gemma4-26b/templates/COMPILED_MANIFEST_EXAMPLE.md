# Example `gem16_compilation.json` — Fast Track R4

This example is illustrative. M03/M08 must replace names, dimensions, byte totals and protocol identifiers with validated data.

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
    "native_protocol": "<versioned-protocol>",
    "native_executable_sha256": "<sha256>",
    "threads": 1,
    "profile_sha256": "<sha256>",
    "toolchain_lock_sha256": "<sha256>"
  },
  "formats": {
    "attention": "fp8-per-channel-v1",
    "shared_mlp": "nvfp4-group16-divisor-v1",
    "routed_experts": "nvfp4-group16-divisor-v1",
    "router": "bf16-source",
    "tied_embedding_head": "nvfp4-group16-divisor-v1"
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
      "name": "model.language_model.embed_tokens.weight_packed",
      "role": "tied_embedding_and_output",
      "storage_dtype": "U8",
      "physical_shape": [262144, 1408],
      "logical_shape": [262144, 2816],
      "byte_length": 369098752,
      "sha256": "<sha256>",
      "source_tensors": [
        "model.language_model.embed_tokens.weight"
      ],
      "transformation": {
        "name": "nvfp4-group16-divisor-v1",
        "group_size": 16,
        "local_scale_tensor": "model.language_model.embed_tokens.weight_scale",
        "global_scale_tensor": "model.language_model.embed_tokens.weight_global_scale",
        "input_scale_tensor": "model.language_model.embed_tokens.input_global_scale",
        "global_scale_role": "divisor"
      },
      "aliased": true,
      "disk_layout": "canonical-row-major-packed",
      "runtime_layout": "sm120-row8-k64"
    },
    {
      "name": "model.language_model.embed_tokens.weight_scale",
      "role": "tied_embedding_and_output_scale",
      "storage_dtype": "F8_E4M3",
      "physical_shape": [262144, 176],
      "byte_length": 46137344,
      "sha256": "<sha256>"
    },
    {
      "name": "model.language_model.embed_tokens.weight_global_scale",
      "role": "tied_embedding_and_output_global_scale",
      "storage_dtype": "F32",
      "physical_shape": [1],
      "byte_length": 4,
      "sha256": "<sha256>"
    },
    {
      "name": "model.language_model.embed_tokens.input_global_scale",
      "role": "tied_embedding_and_output_input_scale",
      "storage_dtype": "F32",
      "physical_shape": [1],
      "byte_length": 4,
      "sha256": "<sha256>"
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
- tied alias is unique and no duplicate `lm_head` payload exists;
- omitted modality tensors are absent;
- totals reconcile;
- quantizer parameters and scale relationships are supported.

Q4_0 examples belong only to an optional M24 profile and must not be copied into the base manifest.
