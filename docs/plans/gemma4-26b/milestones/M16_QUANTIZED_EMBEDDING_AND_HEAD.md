# M16 — Production T=1 tied embedding/head

Status: accepted at `9a374c3dda10b7ae870c712cd70a60aa0a9e2c52`
Class: parallel/conditional
Unblocks: M17

Normative inputs: [Tied embedding/head](../specs/EMBEDDING_HEAD_SPEC.md), [Benchmark matrix](../specs/BENCHMARK_MATRIX.md).

## Outcome

Turn the provisional NVFP4 tied matrix into the production T=1 lookup/output path, or record that the M07 path is already sufficient.

## In scope

- optimized lookup and T=1 projection/candidate selection;
- softcap, suppression, sampling handoff and deterministic tie rule;
- one resident tied payload;
- head-specific quality localization and latency A/B.

## Out of scope

- T=3/T=5 or MTP verifier batches;
- mandatory Q4_0 implementation;
- retaining multiple head formats.

## Exit gate

- [x] T=1 outputs match the reference path.
- [x] One physical matrix serves lookup and projection.
- [x] Head quality remains inside the M13 envelope.
- [x] The optimized path wins or a documented skip allows M17 to retain M07.

## Implementation evidence (2026-08-22)

The T=1 SM120 head consumes the same resident tied row8/K64 matrix as lookup
and writes the required BF16 boundary into the existing FP32 logits workspace.
Together with M14, the two-run full-model comparison improves from 6150.06833
ms to 4818.03496 ms while generated tokens, eight captured layers, router IDs
and the full 262144-logit payload remain bitwise identical. See
`artifacts/m16/acceptance.json`.
