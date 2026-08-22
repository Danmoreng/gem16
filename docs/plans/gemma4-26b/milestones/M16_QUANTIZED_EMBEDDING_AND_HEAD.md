# M16 — Production T=1 tied embedding/head

Status: ready; M07 and M13 accepted
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

- [ ] T=1 outputs match the reference path.
- [ ] One physical matrix serves lookup and projection.
- [ ] Head quality remains inside the M13 envelope.
- [ ] The optimized path wins or a documented skip allows M17 to retain M07.
