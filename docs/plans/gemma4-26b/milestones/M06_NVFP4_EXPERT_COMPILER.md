# M06 — Native NVFP4 expert compiler

Status: accepted 2026-08-12 at implementation commit `81055eb48e05321481a8b63dd0dc5e7e017a7c00`
Class: critical
Depends on: M05
Unblocks: M07 and the quantized adapter of M10

Normative inputs: [Native converter architecture](../specs/NATIVE_CONVERTER_ARCHITECTURE.md), [NVFP4 quantization](../specs/NVFP4_QUANTIZATION_SPEC.md), [Checkpoint compiler](../specs/CHECKPOINT_COMPILER_SPEC.md).

## Outcome

Extend the accepted native converter family with the exact NVFP4 contract for every shared and routed expert matrix and produce one clean full QAT expert conversion.

## In scope

- E2M1 packing, group-16 local scales and tensor-global divisor semantics;
- deterministic native scalar/reference and optimized bounded encoder;
- exact fused/separate gate/up mapping from the M03 inventory;
- streaming directly to final Safetensors payloads;
- exhaustive codec, tie, zero, range, shape and byte-count tests;
- real-shape CPU/CUDA/operator consumption for representative layer classes;
- one clean full QAT-BF16 expert conversion with complete hashes and memory telemetry;
- small/sample ordinary-BF16 versus Unsloth diagnostics sufficient to catch convention errors;
- relevant 12B NVFP4 and generation regressions.

## Not required for this gate

- a complete ordinary-BF16 model conversion;
- exhaustive billion-element ordinary-versus-Unsloth attribution;
- Q4_0;
- runtime kernels beyond consumption probes;
- quality claims.

Those attribution tasks move to M18 when triggered or to a final causal report.

## Parallelism

M10 phase A, M12 phase A, M09 formula work, harness work and M25 feasibility may run concurrently. No other lane edits the native converter protocol or `src/compiler/nvfp4*`.

## Full-run rule

Before the QAT full conversion, freeze the versioned scale/divisor algorithm, exact sampled Ordinary/Unsloth tensor list and diagnostic acceptance conditions in the compiler configuration. The full run starts only from a reviewed, committed, clean worktree after preflight. Development work uses fixtures and bounded probes.

## Exit gate

- [x] The scale/divisor algorithm and sampled diagnostic set are frozen and the sampled Ordinary/Unsloth convention check passes.
- [x] Native codec/quantizer is deterministic across supported thread counts.
- [x] Every QAT shared/routed expert tensor compiles with exact shape and byte accounting.
- [x] Peak host memory is bounded and the full expert family is never resident at once.
- [x] Representative real-shape operators consume the output correctly.
- [x] Invalid scales/NaNs/ambiguous axes fail visibly.
- [x] Relevant 12B tests remain green.

## Evidence

`artifacts/m06/` contains compiler config, clean Release QAT reports, memory telemetry, hashes, representative
Ordinary/Unsloth diagnostics and exact commands. The acceptance summary is `artifacts/m06/acceptance.json`.

## Agent task

Read the active contract, native converter architecture and NVFP4 spec. Implement only M06-owned compiler paths. Return one or more reviewable commits and do not begin M07 in the same sub-agent slice.
