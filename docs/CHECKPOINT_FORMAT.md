# Checkpoint format

## Pinned sources

- Weight/checkpoint repository: `unsloth/gemma-4-12b-it-NVFP4`
- Weight/checkpoint revision: `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`
- `tokenizer_config.json` repository: `google/gemma-4-12B-it`
- `tokenizer_config.json` revision: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- `tokenizer_config.json` SHA-256: `a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9`
- Compressed-tensors version declared by config: `0.17.2.a20260707`
- Lock: `models/gemma4-12b-nvfp4.lock.json`

The downloaded directory is an explicitly composite Hugging Face snapshot: model weights, quantization schema,
tokenizer vocabulary, generation controls, and chat template remain byte-identical to the locked Unsloth source;
only `tokenizer_config.json` is sourced from the official Google instruction-model repository. The lock's
per-file `source` object records this exception. No weight payload is converted, repacked, or duplicated.

The same snapshot contains 104,759,808 bytes of BF16 image/audio embedding and projection tensors that the current
text-only residency policy skips. Video reuses the image tensors. Their exact inventory, processor metadata,
placeholder semantics, and planned residency modes are specified in [MULTIMODAL.md](MULTIMODAL.md).

## Pinned MTP assistant

The optional MTP checkpoint is a separate direct-load snapshot:

- Repository: `google/gemma-4-12B-it-assistant`
- Revision: `364bd03c9952e5b7da73665ee30c9eccfc408345`
- Lock: `models/gemma4-12b-mtp-assistant.lock.json`
- Architecture/model type: `Gemma4UnifiedAssistantForCausalLM` / `gemma4_unified_assistant`

`gem16-inspect --validate` accepts this assistant without weakening primary-model validation. It requires exactly
48 BF16 tensors and 845,713,928 payload bytes: the tied `[262144,1024]` assistant embedding/output matrix, four
Q-only decoder layers, final norm, `[1024,7680]` pre-projection, and `[3840,1024]` post-projection. The first three
layers require local Q/O dimensions 4,096; the final full-attention layer requires 8,192. Any K/V tensor, duplicate
LM head, extra tensor, wrong dtype, or wrong shape is rejected. `gem16-run --assistant-model` uploads and binds this
exact inventory into a separate BF16 arena beside the target. Adding `--mtp-draft-tokens 1|2|4` enables the
correctness scheduler; residency-only use continues to report `execution_enabled=false`. See [MTP.md](MTP.md).

The engine parses and validates the Google tokenizer metadata at startup. The tokenizer-level
`model_max_length` value is Google's intentionally unbounded sentinel and never drives arena sizing; the model
contract remains `config.json:text_config.max_position_embeddings = 262144`. Response close markers declared by
`response_template` must each encode to one token and appear in `generation_config.json:eos_token_id`.

This is a mixed checkpoint, not an all-NVFP4 checkpoint. The pinned config targets attention projections with
per-channel FP8 weights and per-token dynamic FP8 inputs. It targets language-model MLP gate/up/down projections
with packed NVFP4 weights, group size 16, E4M3 local scales, and tensor-global scales.

The verified source contains 1,389 tensors. NVFP4 modules use `.weight_packed`, `.weight_scale`,
`.weight_global_scale`, and `.input_global_scale`. Packed U8 shapes halve the logical contracting dimension;
local scales are E4M3 with one value per 16 logical contracting elements, and both global scales are scalar F32.
Attention projection weights are E4M3 with BF16 per-output-channel `.weight_scale` tensors. Sliding-attention
layers contain named `v_proj` tensors; the eight full-attention layers omit `v_proj` weight and scale tensors under
unified K/V semantics. Execution must follow this per-layer inventory. Nibble order still requires byte-pattern
validation before kernel work. No persistent repacked layout is defined.

The implemented FP8 execution contract is:

```text
input_scale   = max(abs(a_real)) / 448          # one dynamic FP32 scale per token
qa            = round_e4m3fn(a_real / input_scale)
w_real[row,k] = qw_e4m3fn[row,k] * weight_scale_bf16[row]
projection    = sum(qa[k] * qw[row,k]) * input_scale * weight_scale_bf16[row]
```

An all-zero token uses input scale `1.0`. CPU and CUDA implementations produce identical activation bytes and
scale bits for the deterministic real-shape fixtures. Layer-0 Q/K/V/O consume source rows directly with
`QMMA.16832.F32.E4M3.E4M3`; no persistent FP8 repack is required.

Compressed-tensors stores the NVFP4 global values as divisors. For a stored weight divisor `gw`, input divisor
`ga`, packed E2M1 values `qw`, and local E4M3FN scales `sw`, the expected W4A4 execution contract is:

```text
w_real       = qw * sw / gw
a_scaled     = a_real * ga
a_scaled     ~= qa * sa                 # dynamic E2M1 plus E4M3FN scale per 16 values
projection   = sum(qa * sa * qw * sw) / (ga * gw)
```

This interpretation must be verified against the pinned trusted runtime and CPU oracle before it becomes a kernel
contract. Gate and Up divisors are bit-identical for all 48 layers in the pinned checkpoint. All 530,841,600 bytes
of the 144 MLP local-scale tensors are positive, nonzero, and avoid the E4M3FN NaN encoding.
