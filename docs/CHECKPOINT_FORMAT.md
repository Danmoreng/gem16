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

The same snapshot contains 104,759,808 bytes of BF16 image/audio embedding and projection tensors. The runtime
loads all 1,389 checkpoint tensors into one fixed-address Target arena; the legacy `loaded_in_text_only_mode`
manifest field is provenance metadata and no longer controls residency. Video would reuse the resident image
tensors, but no video-frame adapter is implemented. Runtime preprocessing and execution contracts are specified in
[AUDIO.md](AUDIO.md) and [VISION.md](VISION.md).

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

## Gemma 4 26B inspection boundary

M02 recognizes the locked Gemma 4 26B A4B source configuration as explicit variant `gemma4_moe_26b_a4b`. The
canonical fixture is byte-identical to the QAT-BF16 `config.json` at revision
`f1e06dc520982d9b9edd76859fdb7ab209449949`: 3,810 bytes and SHA-256
`ece3392c07744553f4e8bb2b5905bc68b0a7d7ab2927133bb621875b8e4a3289`. Validation fixes the full 30-layer
local/global schedule, 2,816 hidden width, 2,112 shared MLP width, 704 routed-expert width, 128 experts, top-8
routing, attention/KV dimensions, no KV sharing, context, local/global RoPE and source modality metadata.

Manifest JSON schema 2 reports `model_variant`, the MoE dimensions, static capabilities, `runtime_supported` and
`tensor_contract_validated`. For the 26B source, inspection and generic Safetensors safety validation succeed while
`runtime_supported=false` and `tensor_contract_validated=false`. The first product capability is text-only; vision,
audio, video and MTP report false even though the source vision metadata is validated for identity. M03 owns exact
canonical 26B tensor-name/shape validation. M02 does not upload a 26B tensor, allocate a 26B arena or select a CUDA
kernel. The 12B direct-load variant remains executable and retains exact tensor validation.

M03 closes the tensor-inspection boundary without making 26B executable. Manifest JSON schema 3 adds
`checkpoint_profile`, `validation_contract`, an exact `tensor_role` and `residency_class` for every byte, logical
dtype/shape, layer/expert axes, quantization component/producer, scale direction and planned final GPU layout.
`tensor_contract_validated=true` now means one of the exact M03 source contracts passed; it does not imply
`runtime_supported=true`.

The QAT and ordinary BF16 checkpoints share one exact source contract: 1,013 BF16 tensors and
51,611,872,412 payload bytes. Of these, 657 text tensors occupy 50,466,283,580 bytes. The remaining 356 vision
tensors occupy exactly 1,145,588,832 bytes and are classified `compile_excluded_vision`; they remain required for
source completeness and are forbidden in the first compiled artifact. No source MTP, audio or video tensor exists.
The tied `[262144,2816]` embedding is the sole embedding/output payload; any `lm_head` duplicate is rejected.

Every source layer contains fused routed weights:

```text
experts.gate_up_proj  BF16 [128, 1408, 2816]  axis expert,gate_then_up,input
experts.down_proj     BF16 [128, 2816, 704]   axis expert,output,input
```

Expert axis 0 and Gate-before-Up ordering are frozen in the canonical contract. All 30 layers contain exact BF16
router scale `[2816]`, projection `[128,2816]` and per-expert scale `[128]`. The five full-attention layers
`5,11,17,23,29` omit V projection exactly; each reuses raw K as V input before distinct K/V post-processing. The
other 25 layers require separate V.

The external Unsloth reference has 47,478 tensors and 16,903,408,612 payload bytes. It retains the same exact
vision exclusion, but serializes every layer's 128 experts as separate Gate, Up and Down llm-compressor families.
Each family has U8 packed values, E4M3 group-16 local scales and F32 weight/input global divisors. Attention uses
E4M3 weights with BF16 per-output-channel scales. Its profile is `external_unsloth_nvfp4`, never
`gem16_compiled_hybrid`; parsing it does not authorize direct production execution.

| Role | BF16 source | External Unsloth | Official Q4_0 | Frozen compiled-hybrid role |
|---|---|---|---|---|
| Tied head | one BF16 tensor | one BF16 tensor | one Q6_K tensor | one physical Q4_0 or NVFP4 family; M07 selects |
| Attention | BF16 Q/K/O and local V | FP8 weight + channel scale | Q4_0 | FP8 weight + BF16 channel scale |
| Shared MLP | three BF16 matrices | three separate NVFP4 families | Q4_0 | three gem16 NVFP4 families |
| Router | BF16 scale/projection/per-expert scale | same BF16 tensors | F32 | source BF16 |
| Routed experts | fused Gate/Up plus fused Down | 128 separate Gate/Up/Down families | fused Q4_0 | fused expert-major gem16 NVFP4 families |
| Vision | required source family | present but excluded | separate mmproj | absent |

The future compiled validator is deliberately separate from the BF16 and external-reference validators. Its frozen
text-only role mapping has 1,282 tensors with a Q4_0 head or 1,285 with an NVFP4 head. All NVFP4 records identify
producer semantics, E4M3 group-16 scales, divisor direction and `sm120_row8_k64` or
`expert_major_sm120_row8_k64` final layout. M04 may consume this mapping, but M05-M08 still own actual encoding,
provenance and artifact loading. Compact canonical metadata and the complete source cross-map are under
`benchmarks/goldens/gemma4_26b/manifests/`; the large immutable raw header inventories remain under
`benchmarks/goldens/gemma4_26b/source-inventories/`.

## Gemma 4 26B M04 compiler scaffold

M04 adds a deterministic offline compiler container without adding production quantization or runtime loading. Its
only profile is `synthetic-copy-v1`, its only encoder is byte-identical `copy-v1`, and every output records
`artifact_status=m04_scaffold_not_runtime_loadable`. The compiler plan is bound to one source-lock hash and covers
every source tensor exactly through an output operation or explicit exclusion. Vision, audio, video and MTP are
always named omissions; there is no unknown-tensor warning path.

Output is ordinary deterministically sharded Safetensors, a canonical index, approved byte-identical locked
metadata and `gem16_compilation.json` schema 1. Every output record carries operation/source names, source
shard/range and payload hashes, transformation/version, physical/logical dtype/shape, axis mapping, quantizer
parameters, dequantization equation, role, residency, disk/runtime layout, output shard/range and output hash.
Excluded tensors retain the same source identity and exact family/reason. File hashes cover every shard, index and
metadata copy; M08's external artifact lock will provide the compilation manifest's non-circular self-hash.

Source files are completely size/hash verified before Safetensors interpretation. Tensor bytes use bounded read-only
mmap windows; output is staged under `<output>.incomplete`, fsynced, strictly verified and atomically renamed. No
output is overwritten. Resume is deliberately restart-only until a cryptographically bound partial-state schema is
accepted. The complete CLI, schema, memory and canonical-platform contract is in
[GEMMA4_26B_CHECKPOINT_COMPILER.md](GEMMA4_26B_CHECKPOINT_COMPILER.md).

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
