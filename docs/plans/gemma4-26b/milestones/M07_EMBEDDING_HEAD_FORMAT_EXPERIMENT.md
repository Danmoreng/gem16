# M07 — Provisional NVFP4 tied embedding/head

Status: accepted 2026-08-12 at implementation commit `60f500b7be567fafd483ebd6f5f9b07988197ca1`
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

- [x] One QAT NVFP4 tied matrix compiles and validates.
- [x] Lookup and T=1 head match independent dequantized references within the accepted tolerance.
- [x] Softcap, suppression and tie behavior are deterministic.
- [x] Only one physical tied payload is planned/resident.
- [x] M08 has a stable provisional head contract.

## Evidence

`artifacts/m07/` contains compact verification, hash summary, actual-artifact lookup/T=1 diagnostic, command
transcript, compiler config and acceptance record. Expanded plan/compile/manifest reports are pruned from Git;
`artifacts/raw-evidence-index.json` retains their original paths, sizes and SHA-256 values. The 415 MB payload
remains external. This is not a final quality/performance selection report.
