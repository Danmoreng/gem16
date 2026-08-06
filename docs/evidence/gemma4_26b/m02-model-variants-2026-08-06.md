# Gemma 4 26B M02 model-variant handoff

Date: 2026-08-06

Branch: `feat/26b-m02-model-traits`

Accepted M01 parent: `59996f56be81655bb35857edc3c911015358de1a`

Status: accepted by the owner on 2026-08-06; all M02 exit criteria pass

## Scope and implementation

M02 adds a host-only model boundary with explicit `gemma4_unified_12b`, `gemma4_moe_26b_a4b` and assistant
variants. Classification uses parsed architecture/model identifiers and the MoE declaration, never a checkpoint
path. Immutable traits expose inspectability, execution eligibility, layer count and text/vision/audio/video/MTP
capabilities.

The 26B validator fixes the locked QAT-BF16 contract:

- 30 decoder layers and the exact five-local/one-global schedule repeated five times;
- hidden width 2,816, shared MLP width 2,112 and routed-expert width 704;
- 128 experts with top-8 routing;
- 16 query heads, 8 local KV heads, 2 global KV heads, 256/512 head widths and no KV sharing;
- 1,024 local window, 262,144 maximum positions and tied 262,144-row embeddings;
- exact activation, norm, softcap, cache, local/global RoPE and modality metadata.

The canonical test fixture is byte-identical to the locked source `config.json`: 3,810 bytes, SHA-256
`ece3392c07744553f4e8bb2b5905bc68b0a7d7ab2927133bb621875b8e4a3289`. Tests bind that identity to model-lock
SHA-256 `3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230`.

Manifest schema 2 now includes `model_variant`, MoE dimensions, static capabilities, `runtime_supported` and
`tensor_contract_validated`. This is an intentional additive schema revision. The existing 12B checkpoint reports
runtime and tensor-contract support. The 26B checkpoint reports text inspection support but runtime false and tensor
contract false because M03 owns exact canonical tensor-name/shape validation.

No CUDA source, kernel selection, tensor upload, allocator, arithmetic, precision or 12B execution plan changed.
The existing fixed 48-layer target binding and hot path remain specialized for 12B.

## Real-checkpoint inspection evidence

The Host Debug inspector ran against the locked QAT-BF16 checkpoint:

```text
build/Linux/host-debug/bin/gem16-inspect \
  --model models/checkpoints/google-gemma-4-26b-a4b-it-qat-bf16-f1e06dc \
  --validate --json build/m02/gemma4-26b-manifest.json
```

It completed in 42 ms without reading tensor payloads into RAM and reported 1,013 tensors,
`gemma4_moe_26b_a4b`, all exact dimensions, text-only capability, `runtime_supported=false` and
`tensor_contract_validated=false`. The compact tracked record is
[`m02-model-variant-capability.json`](m02-model-variant-capability.json).

A corresponding locked 12B inspection succeeded with 1,389 tensors, `gemma4_unified_12b`,
`runtime_supported=true` and `tensor_contract_validated=true`. These are inspection results, not runtime,
quality, memory-residency or performance claims.

## Tests

- Host Debug configure/build and CTest: 1/1 passed.
- Host ASan/UBSan build and CTest: 1/1 passed.
- Blackwell Release build and CTest: 2/2 passed.
- Python tests: 69/69 passed.
- Real locked QAT-BF16 `gem16-inspect --validate`: passed, 1,013 tensors.
- Real locked 12B `gem16-inspect --validate`: passed, 1,389 tensors.
- Imported plan package: all 130 checksums pass.
- Markdown links, JSON parsing and `git diff --check`: pass.

The Blackwell build emits the pre-existing NVCC `nodiscard`/unused-constant diagnostics; M02 adds no CUDA warning
or CUDA source change.

## Exit checklist

| M02 criterion | Result | Evidence |
|---|---|---|
| 12B and 26B are distinct variants | PASS | explicit enum, immutable traits and real-checkpoint inspection |
| Every 26B architectural dimension is validated | PASS | locked fixture plus positive/negative host tests |
| No CUDA kernel is selected from filename/directory comparisons | PASS | parsed identifiers only; 26B runtime support remains false |
| Existing 12B and assistant contracts pass | PASS | Host and Blackwell CTests plus locked 12B inspection |
| M03/common code can consume one explicit variant | PASS | `ModelConfig::variant`, traits and manifest `model_variant` |

The owner accepted this M02 handoff on 2026-08-06. M03 exact tensor inventory is unblocked but remains unstarted;
it must use a separate milestone branch based on the accepted M02 closure. Any 26B memory plan, compiler/runtime
work and all CUDA implementation remain out of scope and unstarted.
