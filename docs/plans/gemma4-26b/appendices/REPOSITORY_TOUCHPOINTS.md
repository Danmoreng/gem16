# Current repository touchpoints

Anchored at `1c4287965d318ba32a68e597f9d7b6678b883376`.

## Current direct loader

`src/cuda/engine/target_model.cu` currently:

- validates one primary Gemma 4 architecture;
- allocates one aligned device weight arena;
- streams Safetensors tensors;
- transforms NVFP4 weights/scales into Row8/K64 final layout;
- binds 48 layers;
- hard-codes hidden 3840, intermediate 15360 and vocabulary 262144;
- binds BF16 tied embedding;
- binds dense gate/up/down per layer.

26B impact:

- variant traits;
- 30-layer storage;
- quantized tied binding;
- routed expert sets;
- shared dense MLP;
- router tensors;
- text-only omission.

## Current model contract

`src/model/config.*`, `manifest.*` and `checkpoint_loader.cpp` validate 12B primary and MTP assistant contracts and compressed-tensors FP8/NVFP4 schemas.

26B requires:

- additional config fields;
- exact model variant validation;
- expert/source mapping;
- compiled artifact schema;
- quantized head class;
- omitted modalities.

## Current memory planner

`src/runtime/memory_plan.cpp`:

- counts manifest tensors;
- supports explicit separate K/V;
- computes local/full cache by layer types;
- has fixed context profiles;
- builds one simple weights/scales/KV arena plan.

26B requires named workspaces and selected residency, plus session admission based on real slot cost.

## Checkpoint compiler data plane

M04's accepted control plane is under `tools/gem16_compile/` and remains responsible for source locks, exact plans,
coverage, publication and provenance. M05 adds the first promoted native data-plane seed:

- `src/compiler/fp8_batch_encoder.h` / `.cpp` — bounded native BF16-to-FP8 rowwise batch conversion;
- `src/cli/fp8_compiler_main.cpp` — current native M05 batch executable;
- `tools/gem16_compile/native_fp8.py` — Python job/telemetry adapter, not the numerical implementation.

M06 NVFP4, M07 head encoders and M18 large comparisons must extend this native data plane. Do not add promoted
`quantize_*.py` elementwise loops or a silent Python fallback. See
[`../specs/NATIVE_CONVERTER_ARCHITECTURE.md`](../specs/NATIVE_CONVERTER_ARCHITECTURE.md) and the retained local
llama.cpp study [`../../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md`](../../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md).

## Current FP8/NVFP4 kernels

The repository already has:

- host codecs/reference paths;
- direct T=1 NVFP4 projection;
- batch projection;
- fused gate/up;
- CUTLASS block-scaled prefill;
- FP8 Q/K/V/O paths;
- SM120 source layout transformation.

Reuse arithmetic contracts; do not assume 12B shapes or dense-layer scheduling.

## Current output head

`src/cuda/output_head.cu` hard-codes:

```text
hidden = 3840
vocabulary = 262144
weights = BF16
```

It implements:

- full logits;
- fused candidates;
- batch candidates up to T=5;
- deterministic tie break;
- softcap;
- suppression.

Refactor stable candidate/argmax semantics and provide Q4_0/NVFP4 weight readers.

## Current engine

`InferenceEngine` owns:

- target weights;
- hybrid KV;
- fixed workspaces;
- prefill;
- whole-model decode graphs;
- optional MTP;
- sampling;
- diagnostics.

26B base profile must instantiate without assistant/media and preserve 12B features.

## Current product

- CLI run/chat/bench/server;
- OpenAI-compatible endpoints;
- resident sessions;
- Studio model download/management;
- shared HF cache.

26B adds a model profile and capability restrictions, not a second application.

## Current governance conflict

Root `AGENTS.md` permits reproducible profile-specific compiled artifacts. M00 must freeze the stricter QAT-derived
26B contract before compiler work:

- deterministic repository-owned compiler;
- immutable provenance;
- no runtime compilation;
- no opaque/driver-tied format;
- direct Unsloth baseline retained;
- 12B policy unchanged.
