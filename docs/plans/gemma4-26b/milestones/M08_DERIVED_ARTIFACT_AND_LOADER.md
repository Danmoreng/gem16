# M08 — Complete derived artifact and direct loader

Status: blocked by M06/M07
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
- startup telemetry proving no duplicate persistent device layout;
- 12B loader/download/inspect regressions.

## Parallelism

M09 phase A, M10 and M12 fixture work may continue. Final schema and loader files are integration-owned; sub-agents do not modify them concurrently.

## Exit gate

- [ ] Two clean complete builds are byte-identical.
- [ ] External lock and provenance are complete.
- [ ] Inspect/validate binds every expected tensor and rejects corruption/extra modalities.
- [ ] Loader retains one head and one expert layout.
- [ ] Startup does not transiently OOM on the reference GPU.
- [ ] Predicted immutable bytes are ready for M09 reconciliation.

## Evidence

`artifacts/m08/` contains compiled manifest, reproducibility hashes, load telemetry and validation output.
