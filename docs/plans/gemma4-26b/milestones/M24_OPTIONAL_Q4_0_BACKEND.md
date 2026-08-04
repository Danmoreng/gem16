# M24 — Optional full-model Q4_0 reference backend

## Objective

Optionally implement a complete Q4_0 weight backend for Gemma 4 26B as an internal quality/performance reference, without replacing the native NVFP4 production path by default.

## Why this milestone exists

A native in-engine Q4_0 path can remove cross-runtime timing and semantic differences from comparisons and may identify T=1 shapes where Q4_0 W4A16 is competitive.

## Prerequisites

- M23 release or explicit owner authorization
- M07 Q4_0 codec
- Stable production path

## Repository areas to inspect first

- `src/numeric/q4_0.cpp`
- `src/cuda/output_head_q4_0.cu`
- `src/cuda/nvfp4/`
- `src/cuda/engine/inference_engine.cu`
- `pinned llama.cpp CUDA Q4_0 kernels`

## Suggested additions or boundaries

- `src/cuda/q4_0/gemv.cu`
- `src/cuda/q4_0/gemm.cu`
- `src/cuda/q4_0/reference.cu`
- `docs/GEMMA4_26B_Q4_0_BACKEND.md`

## Implementation sequence

1. Define whether the backend consumes official GGUF directly or a Safetensors-mapped Q4_0 artifact; do not add two loaders casually.
2. Implement exact dequantization/reference operations for every required tensor family.
3. Implement T=1 W4A16 GEMV and batch/prefill candidates with explicit memory/compute trade-offs.
4. Support router/norms and cache semantics identical to the production model.
5. Run quality parity against official Google Q4_0 and performance against llama.cpp under common token inputs.
6. Keep all path metadata and fallback reporting explicit.
7. Promote any production use only through a new decision and full M19–M23 rerun.

## Required tests

- Official Q4_0 tensor decode/quantizer fixtures.
- Operator and full-logit parity against pinned llama.cpp or an independent CPU reference.
- Full-model deterministic and quality comparison.
- No CPU offload or token-loop allocation.
- Memory and performance matrix at required contexts.

## Evidence and documentation outputs

- `docs/GEMMA4_26B_Q4_0_BACKEND.md`
- `artifacts/m24/q4_0-parity.json`
- `artifacts/m24/q4_0-performance.json`

## Suggested commands

```text
python tools/validate_gemma4_26b_q4_backend.py --model "$GOOGLE_Q4" --output artifacts/m24
```
```text
build/blackwell-release/bin/gem16-bench decode --model "$GEM16_Q4_26B" --context 8192 --tokens 256 --warmups 3 --repetitions 10
```

## Risks to watch in this milestone

- Q4_0 lacks the same native block-scaled FP4 path and may require dequantization-heavy kernels.
- GGUF integration can broaden the loader beyond the project's narrow design.
- Maintaining two full production backends increases test burden.
- Cross-runtime parity may still differ because of kernel reduction order.

## Forbidden shortcuts

- Reopening the released production choice without full qualification.
- Calling Q4_0 Tensor-Core native NVFP4.
- Copying llama.cpp code without license and attribution review.
- Adding silent Q4 fallback to unsupported NVFP4 shapes.

## Exit criteria

- [ ] Optional backend is exact enough for its stated reference purpose.
- [ ] Path and format are reported explicitly.
- [ ] No released NVFP4 behavior regresses.
- [ ] Any production promotion is deferred to a separate decision and full gate rerun.

## Downstream milestones unblocked

- Future research comparisons

## Codex execution prompt

```text
You are implementing M24: Optional full-model Q4_0 reference backend in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M24. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M24 exit criterion passed. Stop before starting the next milestone.
```
