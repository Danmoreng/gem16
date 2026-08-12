# Native checkpoint-converter architecture — Fast Track R4

## Accepted boundary

M04 is the accepted Python standard-library control plane for immutable locks, exact plans, tensor coverage, report schemas, atomic publication and small independent fixtures. M05 is the accepted first native C++20 numerical backend for BF16-to-FP8 attention conversion.

M06 and M07 extend the same native converter family:

| Milestone | Native responsibility |
|---|---|
| M06 | BF16 shared/routed experts to the locked NVFP4 contract |
| M07 | QAT tied embedding/head to provisional NVFP4 |
| M08 | no new quantizer; assemble and validate the complete artifact |
| M18 | conditional large comparisons/alternative conversions |
| M25 | optional assistant conversion needed for MTP residency |

Q4_0 conversion is not part of M06–M08. It may be added under M24 or a targeted M18 diagnosis.

## Control plane

Python may:

- verify sources/locks and parse model metadata;
- generate exact native jobs and bounded I/O ranges;
- assemble canonical metadata and Safetensors shards;
- run small independent reference fixtures;
- collect native telemetry and publish atomically.

Python may not provide a production elementwise fallback or own billion-element conversion/comparison loops.

## Native data plane

The native backend owns:

- promoted BF16/FP8/NVFP4 conversion and dequantization;
- deterministic rounding, packing and scale encoding;
- bounded shard/row/tile streaming;
- coupled weight/scale generation from one source traversal;
- large error metrics and comparisons;
- explicit thread-count identity and deterministic telemetry.

The backend receives descriptor-bound source/output ranges, validates every contract field and fails visibly on unsupported precision/layout/producer semantics.

## Artifact layout

The on-disk artifact remains auditable and model-semantic. Do not write an opaque SM120-only monolith. Runtime-specific Row8/K64 or expert-major tiling may be produced during bounded load into the single final device representation, but the loader must not retain both source-order and runtime-order copies.

## Determinism and full-run policy

- Exhaustive codecs, byte fixtures, malformed inputs and bounded thread-identity tests are required.
- M06 requires one clean full QAT expert conversion, not a full ordinary control.
- M07 requires one clean QAT tied-head conversion.
- Complete ordinary-BF16 conversion belongs to conditional M18/final attribution.
- M08 requires two clean complete-artifact builds with identical hashes.
- Full runs require reviewed committed code, a clean worktree and preflight.

## Publication

Stage below `<output>.incomplete`, use checked writes/fsync, reopen and validate, then atomically rename on the same filesystem. M08 creates the external artifact lock binding source, compiler, metadata and output files.

## External code

llama.cpp and other converters are references only. Any copied code requires an exact source pin, license/provenance and differential tests. Their format, scale, tie, fallback and intermediate-file choices do not override the Gem16 contract.
