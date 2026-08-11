# Proposed repository directory tree

This is a responsibility map, not a mandatory naming decree.

```text
gem16/
├─ models/
│  ├─ gemma4-26b-qat-bf16.lock.json
│  ├─ gemma4-26b-base-bf16.lock.json
│  ├─ gemma4-26b-unsloth-nvfp4.lock.json
│  ├─ gemma4-26b-qat-q4_0.lock.json
│  └─ gemma4-26b-gem16-hybrid.lock.json
├─ tools/
│  ├─ compile_gemma4_26b.py
│  ├─ verify_compiled_model.py
│  ├─ compare_model_manifests.py
│  ├─ compare_quantized_checkpoints.py
│  ├─ capture_gemma4_26b_goldens.py
│  ├─ validate_gemma4_26b_full_model.py
│  ├─ analyze_gemma4_26b_conversion.py
│  ├─ evaluate_gemma4_26b_quality.py
│  ├─ benchmark_gemma4_26b_cross_engine.py
│  ├─ validate_gemma4_26b_long_context.py
│  └─ gem16_compile/
│     ├─ plan.py
│     ├─ reader.py
│     ├─ writer.py
│     ├─ native_fp8.py          # M05 control-plane adapter; native backend is authoritative
│     ├─ quantize_fp8.py        # independent oracle/fixtures only, never promoted conversion
│     └─ report.py
├─ src/compiler/
│  ├─ fp8_batch_encoder.*       # current M05 native seed
│  ├─ nvfp4_batch_encoder.*     # planned M06 extension of the shared data plane
│  ├─ q4_0_batch_encoder.*      # planned M07 extension
│  ├─ checkpoint_compiler.*     # planned unified native family
│  └─ provenance.*
├─ src/
│  ├─ model/
│  │  ├─ model_variant.*
│  │  ├─ gemma4_26b_contract.*
│  │  └─ compiled_checkpoint.*
│  ├─ numeric/
│  │  ├─ q4_0.*
│  │  └─ gemma4_26b_moe.*
│  └─ cuda/
│     ├─ moe/
│     │  ├─ moe.h
│     │  ├─ reference.cu
│     │  ├─ router_reference.cu
│     │  ├─ router_sm120.cu
│     │  ├─ decode_sm120.cu
│     │  ├─ prefill_router.cu
│     │  ├─ prefill_permute.cu
│     │  ├─ prefill_sm120.cu
│     │  └─ prefill_reduce.cu
│     ├─ embedding/
│     │  ├─ lookup.cu
│     │  ├─ q4_0_reference.cu
│     │  └─ nvfp4_reference.cu
│     ├─ output_head_q4_0.cu
│     └─ output_head_nvfp4.cu
├─ tests/
│  ├─ unit/
│  │  ├─ gemma4_26b_config_test.cpp
│  │  ├─ gemma4_26b_manifest_test.cpp
│  │  ├─ gemma4_26b_memory_plan_test.cpp
│  │  ├─ gemma4_26b_moe_test.cpp
│  │  └─ q4_0_test.cpp
│  ├─ cuda/
│  │  ├─ gemma4_26b_moe_reference_test.cu
│  │  ├─ gemma4_26b_moe_decode_test.cu
│  │  ├─ gemma4_26b_moe_prefill_test.cu
│  │  ├─ gemma4_26b_attention_test.cu
│  │  └─ gemma4_26b_output_head_test.cu
│  └─ python/
│     ├─ test_checkpoint_compiler.py
│     ├─ test_fp8_compiler.py
│     └─ test_nvfp4_compiler.py
├─ benchmarks/
│  ├─ goldens/gemma4_26b/
│  ├─ quality/gemma4_26b/
│  └─ baselines/gemma4_26b_16gb/
└─ docs/
   ├─ GEMMA4_26B.md
   ├─ GEMMA4_26B_CHECKPOINT.md
   ├─ GEMMA4_26B_MEMORY.md
   ├─ GEMMA4_26B_ATTENTION.md
   ├─ GEMMA4_26B_CORRECTNESS.md
   ├─ GEMMA4_26B_RUNTIME.md
   ├─ GEMMA4_26B_CONVERSION_STUDY.md
   ├─ GEMMA4_26B_QUALITY.md
   ├─ GEMMA4_26B_BENCHMARKING.md
   └─ GEMMA4_26B_LONG_CONTEXT.md
```

Prefer existing repository conventions over this tree when equivalent responsibilities already have a clear home.
