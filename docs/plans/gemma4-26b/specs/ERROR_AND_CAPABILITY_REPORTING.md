# Error and capability reporting specification

## Principle

Unsupported precision, model feature or memory state must fail visibly. Users and benchmarks must be able to prove which path executed.

## Capability report

`--print-kernel-capabilities` or equivalent must report:

```text
compiled_architectures
gpu_name
compute_capability
cuda_runtime_version
cuda_driver_version
native_fp8_attention
native_nvfp4_moe_decode
native_nvfp4_moe_prefill
q4_0_head
nvfp4_head
model_variants
schema_versions
cuda_graphs
```

Distinguish:

- compiled capability;
- hardware capability;
- selected runtime path;
- qualified product capability.

A compiled kernel is not a qualified feature.

## Model-load report

Print or serialize:

```text
model variant
artifact profile
source/compiler provenance
head format
text-only status
weight bytes
context
KV bytes
workspace
admission margin
unsupported media/MTP
```

## Error categories

Suggested status distinctions:

- invalid argument;
- source/model not found;
- data loss/corruption;
- unsupported model contract;
- unsupported precision/path;
- resource exhausted;
- CUDA/internal;
- quality/benchmark precondition failure.

## Required error detail

Example:

```text
unsupported Gemma 4 26B tensor:
name=model....experts.gate_up_proj.weight_packed
logical_shape=[128,1408,2816]
quantization_class=NVFP4_PACKED
reason=group size 32; profile requires group size 16
artifact_lock=...
```

Memory error:

```text
cannot create 32K execution slot:
free=...
required_slot=...
probe_resident=...
required_margin=...
shortfall=...
```

## Benchmark preconditions

Benchmark command exits nonzero when:

- artifact hash mismatch;
- fallback count nonzero;
- CPU offload;
- busy GPU;
- wrong power profile where required;
- insufficient repetitions;
- output invalid/non-deterministic;
- dirty worktree unless explicitly labeled;
- unsupported native path.

Do not merely write a warning into JSON and return success.

## API errors

Use clear 4xx for unsupported request features and 5xx/resource codes only for genuine server/runtime failure.

Unsupported media example:

```json
{
  "error": {
    "type": "unsupported_feature",
    "code": "gemma4_26b_text_only",
    "message": "This model profile does not support image input."
  }
}
```

## Metrics

Counters:

```text
gem16_fallback_total
gem16_resource_exhaustion_total
gem16_unsupported_feature_total
gem16_model_validation_failure_total
gem16_token_loop_allocation_total
```

A release benchmark requires fallback/allocation counters unchanged at zero.

## Tests

- exact status/category;
- useful message fields;
- CLI exit code;
- API JSON;
- metrics increment;
- no secret paths/tokens;
- 12B behavior remains compatible.
