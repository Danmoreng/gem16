# M08 — Complete derived artifact and direct loader

Status: accepted 2026-08-14
Class: critical
Unblocks: final M09 reconciliation, M11, M12 runtime and M13

Normative inputs: [Derived checkpoint schema](../specs/DERIVED_CHECKPOINT_SCHEMA.md), [Checkpoint provenance](../specs/CHECKPOINT_PROVENANCE_SPEC.md), [Checkpoint compiler](../specs/CHECKPOINT_COMPILER_SPEC.md).

## Outcome

Build and load one complete text-only QAT-derived artifact with one final weight layout and an immutable external lock.

## In scope

- final Safetensors schema and compilation metadata;
- all FP8 attention, NVFP4 expert/shared, provisional NVFP4 head and small BF16/F32 tensors;
- explicit modality omission;
- complete provenance and per-file/tensor hashes;
- direct loader bindings with no runtime quantization;
- two clean full builds with identical hashes;
- exact-arena GPU admission telemetry proving no duplicate persistent device layout;
- 12B loader/download/inspect regressions.

## Parallelism

Final schema and loader files are integration-owned; sub-agents do not modify them concurrently. The required owner
instruction to start M08 was received on 2026-08-14.

## Exit gate

- [x] Two clean complete builds are byte-identical.
- [x] External lock and provenance are complete.
- [x] Inspect/validate binds every expected tensor and rejects corruption/extra modalities.
- [x] Loader retains one head and one expert layout.
- [x] Exact single-arena admission does not transiently OOM on the reference GPU.
- [x] Predicted immutable bytes are ready for M09 reconciliation.

## Evidence

`artifacts/m08/compiler-plan-summary.json` freezes the 1,285-output, 14,696,569,196-byte partition.
`artifacts/m08/diagnostic-summary.json` records two byte-identical dirty-tree builds, identical external locks, Python
verification and direct C++ loader validation. It remains explicitly non-acceptance diagnostic evidence.

`artifacts/m08/acceptance.json` records two byte-identical clean builds from implementation commit
`f433358b8e2c1250b95801fc898faee4fcedcbe5`, independent Python verification, direct C++ loader validation, the
protected 12B inspect regression and exact-arena admission on the reference GPU. The admission probe is synthetic,
not model execution; real payload upload, named CUDA-region reconciliation and the full 32K residency gate remain
M09 phase B work.
