# M07 — Provisional NVFP4 tied embedding/head

Status: blocked by M06
Class: critical-lite
Unblocks: M08

Normative inputs: [Tied embedding/head](../specs/EMBEDDING_HEAD_SPEC.md), [NVFP4 quantization](../specs/NVFP4_QUANTIZATION_SPEC.md).

## Outcome

Provide the smallest correct tied embedding/output-head path needed for the first complete artifact. The provisional format is NVFP4 and exactly one physical matrix is resident.

## In scope

- reuse the accepted NVFP4 codec and metadata contract;
- compile the QAT tied matrix once;
- CPU/reference embedding lookup and T=1 output projection/argmax;
- model embedding scale, final softcap, suppression and deterministic lowest-token tie rule;
- diagnostic full-logit output for small fixtures;
- exact byte and tied-pointer validation.

## Out of scope

- native Q4_0 encoding or kernels;
- BF16 residency on 16 GB;
- T=3/T=5 verifier batches;
- final head-performance tuning;
- broad head-format A/B studies.

Q4_0 remains an external reference and optional M24 work. Multi-row verification belongs to M25.

## Exit gate

- [ ] One QAT NVFP4 tied matrix compiles and validates.
- [ ] Lookup and T=1 head match independent dequantized references within the accepted tolerance.
- [ ] Softcap, suppression and tie behavior are deterministic.
- [ ] Only one physical tied payload is planned/resident.
- [ ] M08 has a stable provisional head contract.

## Evidence

`artifacts/m07/` contains head bytes, fixture comparisons and T=1 reference timing only. It is not a final quality/performance selection report.
