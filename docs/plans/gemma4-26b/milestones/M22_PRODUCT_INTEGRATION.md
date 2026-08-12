# M22 — CLI/server product integration

Status: blocked by a frozen M17 profile contract
Class: product; may overlap M19–M21

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

- [ ] CLI/server load the exact frozen artifact.
- [ ] Profile, context and native-path metadata are accurate.
- [ ] Unsupported MTP/media/multi-slot requests fail clearly.
- [ ] 12B API behavior remains compatible.
