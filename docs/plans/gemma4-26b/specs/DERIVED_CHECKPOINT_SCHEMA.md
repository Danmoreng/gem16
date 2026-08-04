# Derived Gemma 4 26B checkpoint schema

## Design goals

The compiled artifact must be:

- standard Safetensors plus JSON;
- immutable and reproducible;
- text-only;
- self-describing;
- independent of a driver version;
- directly loadable by gem16;
- auditable back to source tensors;
- incapable of silently selecting another precision.

## Directory layout

Suggested layout:

```text
gemma4-26b-a4b-gem16-hybrid/
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
  model.safetensors.index.json
  model-00001-of-NNNNN.safetensors
  ...
  gem16_compilation.json
  README.md
```

Do not include vision/audio processor files unless the product deliberately needs them for explaining unsupported input. They must never imply support.

## Config extension

Preserve model architecture facts and add a namespaced block:

```json
{
  "architectures": ["Gemma4ForConditionalGeneration"],
  "model_type": "gemma4",
  "text_config": { "...": "..." },
  "tie_word_embeddings": true,
  "gem16": {
    "schema_version": 1,
    "profile": "sm120-text-hybrid-v1",
    "variant": "gemma4-26b-a4b",
    "text_only": true,
    "head_format": "q4_0",
    "supports_mtp": false,
    "supports_vision": false,
    "supports_audio": false,
    "supports_video": false
  }
}
```

Do not rewrite architecture values merely to simplify the runtime.

## Tensor classes

Normative classes:

| Class | Required storage |
|---|---|
| `FP8_WEIGHT_E4M3` | E4M3 payload |
| `FP8_WEIGHT_SCALE` | BF16 `[rows, 1]` |
| `NVFP4_PACKED` | U8, two E2M1 per byte |
| `NVFP4_LOCAL_SCALE_E4M3` | E4M3 `[rows, K/16]` |
| `NVFP4_GLOBAL_SCALE` | F32 scalar |
| `NVFP4_INPUT_SCALE` | F32 scalar |
| `Q4_0_PACKED` | U8 block payload, 18 bytes per 32 logical weights |
| `BF16` | source BF16 |
| `FP32` | source F32 |

If Q4_0 is represented as a single opaque byte tensor, its logical row/block metadata must be in `gem16_compilation.json`. Prefer separate scale/payload tensors only if it simplifies loader validation without changing bytes.

## Canonical logical roles

Every tensor has one role:

- tied embedding/output;
- final norm;
- attention Q/K/V/O;
- attention norm/scale;
- shared dense gate/up/down;
- router norm/scale/projection/per-expert scale;
- routed expert gate/up/down or canonical fused gate-up;
- feed-forward norms;
- layer scalar;
- cache scale metadata.

Role names, not string regexes in kernel code, drive binding.

## Naming

The compiler may preserve source names where they map cleanly. For synthesized or split tensors, use a versioned deterministic namespace.

Example only:

```text
model.language_model.layers.0.experts.gate_up_proj.weight_packed
model.language_model.layers.0.experts.gate_up_proj.weight_scale
model.language_model.layers.0.experts.gate_up_proj.weight_global_scale
model.language_model.layers.0.experts.gate_up_proj.input_global_scale
```

Do not adopt this example until M03 confirms actual source names and M06 chooses the canonical layout.

## Expert physical shape

The manifest must carry both physical and logical shapes.

Example for fused gate/up:

```text
logical: [128, 2 * 704, 2816]
physical packed U8: [128, 1408, 1408]
local scales: [128, 1408, 176]
```

Actual Safetensors can only store ordinary dimensions. Flattening expert and row axes is allowed only with explicit logical metadata and tested offset calculations.

## Tied head

The artifact stores exactly one tied tensor family. It must be marked:

```json
{
  "role": "tied_embedding_and_output",
  "aliased": true
}
```

No `lm_head.weight` duplicate is permitted.

## Omitted modalities

`gem16_compilation.json` must list source tensor families omitted and their source byte totals. Runtime validation must not expect them.

Example:

```json
{
  "omitted_tensor_groups": [
    {
      "group": "vision",
      "reason": "text-only 16GB profile",
      "source_tensor_count": 123,
      "source_bytes": 1190000000
    }
  ]
}
```

Never include placeholder zero tensors.

## Runtime layout

On-disk layout is a logical/auditable checkpoint layout. Runtime may stream-transform NVFP4 payload and scales into Row8/K64 final device order. The manifest must distinguish:

```text
disk_layout = source/canonical row-major packed
runtime_layout = sm120-row8-k64
persistent_repack_bytes = 0
```

Do not write driver- or CUTLASS-internal layouts to disk in schema version 1.

## Schema validation

Validate:

- architecture and traits;
- tensor count;
- unique names;
- non-overlapping payloads;
- expected dtypes/shapes;
- quantizer relationships;
- logical byte counts;
- tied alias;
- no modality tensor;
- source/compiler provenance;
- output hashes;
- supported schema/profile/head format.

Unknown schema or profile is an error.

## Versioning

- additive metadata compatible with old runtime: minor version;
- tensor naming/layout/quantizer contract change: new profile or schema major;
- never reinterpret old bytes under a new meaning.

## Example manifest

See [`../templates/COMPILED_MANIFEST_EXAMPLE.md`](../templates/COMPILED_MANIFEST_EXAMPLE.md).
