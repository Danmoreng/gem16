# Gemma 4 26B checkpoint compiler specification — Fast Track R4

## Base transformation

```text
QAT BF16 source
  ├─ attention Q/K/V/O → accepted FP8 + row scales
  ├─ shared MLP → NVFP4 + group-16 scales/divisors
  ├─ routed experts → NVFP4 + group-16 scales/divisors
  ├─ router/norms/scalars → source precision
  ├─ tied embedding/head → provisional NVFP4
  └─ modality/MTP tensors → omitted from base artifact
```

Q4_0 is not a required compiler stage. M25 may compile a separately locked assistant artifact.

## Architecture

The Python control plane validates sources, generates complete descriptor-bound jobs, publishes Safetensors and records provenance. The native C++20 data plane owns all promoted large numerical transformations and comparisons. Missing native support fails visibly; no Python production fallback exists.

## Required actions

A single user-facing command may expose plan/compile/verify, but each action is explicit and records the full configuration. Compiler planning must cover every source tensor exactly as transformed, copied, deferred or excluded.

## Resource bounds

- read-only bounded mappings/staging;
- no whole-model tensor materialization;
- explicit max host memory and staging bytes;
- checked range/shape/byte arithmetic;
- direct output to canonical shards;
- staged atomic publication.

## Milestone gates

- M06 owns expert/shared NVFP4.
- M07 owns provisional tied-head NVFP4.
- M08 assembles the complete artifact and supplies the external lock.
- M18 owns optional full ordinary/alternative conversion for diagnosis.
- M24 owns optional Q4_0.
- M25 owns any assistant conversion.

## Verification

Verification checks source lock, metadata schema, tensor coverage, file/tensor hashes, byte totals, formats, shape/axis mappings, omitted families and provenance. Complete-artifact reproducibility is proven by M08's two clean builds, not by re-running every partial stage twice.
