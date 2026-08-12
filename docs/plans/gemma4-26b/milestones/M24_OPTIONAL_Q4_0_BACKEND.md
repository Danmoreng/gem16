# M24 — Optional internal Q4_0 backend

Status: optional
Class: diagnostic/research
Depends on: M13 or M23

Normative inputs: [Q4_0 reference/optional backend](../specs/Q4_0_SPEC.md).

## Outcome

Implement an internal Q4_0 reference only when it materially helps head-quality diagnosis, causal comparison or a measured T=1 product alternative.

## Rules

- never block M08, M13, M17, M23 or M25;
- keep official Q4_0 external reference usable without this backend;
- reuse no copied code without exact pin/license/provenance;
- retain only one production head/weight representation;
- do not label Q4_0 as native NVFP4 Tensor Core execution.

## Exit gate

A Q4_0 backend is retained only if it provides a concrete diagnostic or product benefit under the same quality/memory/performance rules. Otherwise record rejection and stop.
