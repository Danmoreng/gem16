# M22 — CLI/server product integration

Status: accepted 2026-08-23 at implementation commit `f0aa302aa0246d44e1c8477dbbbb67fbbe2d2037`
Class: product; closes before bounded fixed-target prefill/decode work and the final M21/M20 evidence freeze

Normative inputs: [API/CLI changes](../specs/API_CLI_CHANGES.md), [Capability reporting](../specs/ERROR_AND_CAPABILITY_REPORTING.md), [Session ownership](../specs/SESSION_OWNERSHIP_AND_CONCURRENCY.md).

## Outcome

Expose the base 26B profile through CLI and server with accurate capability and memory reporting.

## Critical scope

- model/profile selection and immutable download/verify flow;
- max-context/admission errors;
- CLI and OpenAI-compatible server smoke;
- one-slot behavior and cancellation/resident continuation;
- `supports_mtp=false` until M25;
- precise rejection of media and second-slot requests.

Studio/catalog work is a parallel nonblocking subtask and may follow M23.

## Exit gate

- [x] CLI/server load the exact frozen artifact.
- [x] Profile, context and native-path metadata are accurate.
- [x] Unsupported MTP/media/multi-slot requests fail clearly.
- [x] 12B API behavior remains compatible.

Compact acceptance: `artifacts/m22/acceptance.json`. Raw product/driver reports remain ignored under
`artifacts/raw/m22/qualification-20260823-final/` and are referenced by size and SHA-256.
