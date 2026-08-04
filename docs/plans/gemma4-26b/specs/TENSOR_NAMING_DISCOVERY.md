# Tensor naming and inventory discovery protocol

## Principle

Never design bindings from model-card prose or expected names alone. The exact pinned checkpoints and reference implementation define the inventory.

## Required inputs

Inventory all of:

- Google ordinary BF16 26B;
- Google QAT unquantized BF16 26B;
- Unsloth NVFP4 26B;
- official Google Q4_0 GGUF;
- current 12B pinned checkpoint for regression comparison.

## Discovery output

For every tensor:

```json
{
  "name": "...",
  "dtype": "BF16",
  "physical_shape": [ ... ],
  "logical_shape": [ ... ],
  "bytes": 0,
  "shard": "...",
  "absolute_offset": 0,
  "role": "...",
  "layer": 0,
  "expert_axis": null,
  "source_family": "qat_bf16",
  "quantization_group": null
}
```

## Classification rules

Classify by validated role:

- embedding/head;
- final norm;
- attention Q/K/V/O;
- attention q/k norm and cache scales;
- shared dense gate/up/down;
- router normalization scale;
- router projection;
- per-expert scaling;
- routed expert fused gate-up or separate gate/up;
- routed expert down;
- feed-forward norms;
- layer scalar;
- per-layer embedding/projection if present;
- modality;
- unknown.

Regex can assist discovery but may not be the final authority.

## Cross-checks

For each source pair:

### Ordinary BF16 versus QAT BF16

- same tensor names or explicit mapping;
- same shapes;
- expected dtype;
- tokenizer/config relationship;
- report added/removed tensors;
- compute weight differences without loading whole model.

### Ordinary BF16 versus Unsloth

- map quantized module to source module;
- identify ignored BF16 tensors;
- identify fused or split expert representation;
- verify every scale family;
- report modality inclusion.

### QAT BF16 versus official Q4_0

- map GGUF tensor names to source logical roles;
- verify transposes and axis order;
- identify tensors kept higher precision;
- identify separate multimodal file.

## Expert axis proof

Do not assume whether experts are:

```text
[experts, 2I, H]
[2I, experts, H]
[experts * 2I, H]
```

Prove axis semantics through:

- config dimensions;
- reference module declaration;
- tensor shape;
- selected expert slice output against framework capture.

Pin at least one real expert gate/up/down fixture with known source offsets and output.

## Layer inventory proof

Generate a table for every layer:

```text
layer
attention type
owns K
owns V
KV producer if shared
shared MLP tensor family
router family
expert family
norm/scalar family
total bytes
```

Reject missing or unexpected tensors.

## Unknown tensors

Every unknown tensor must be:

- classified and supported;
- explicitly omitted with reason;
- or cause validation failure.

No silent ignore list outside a versioned model contract.

## Tools

Suggested:

```text
gem16-inspect --model ... --json ...
python tools/compare_model_manifests.py ...
python tools/map_gguf_to_safetensors.py ...
```

Reports must be committed or stored as immutable benchmark/golden artifacts, not pasted only into issue comments.

## Exit artifact

M03 produces:

```text
benchmarks/goldens/gemma4_26b/manifests/
  ordinary-bf16.json
  qat-bf16.json
  unsloth-nvfp4.json
  google-q4_0.json
  cross-map.json
  layer-table.json
```
