# Test matrix — Fast Track R4

## Tiers

| Tier | Scope | Typical use |
|---|---|---|
| T0 | schema, overflow, codecs, fixtures | every local change |
| T1 | synthetic CPU/CUDA operator parity | arithmetic changes |
| T2 | real tensor/layer probes | compiler/runtime slices |
| T3 | full model, generation, allocations | M13/M17 integration |
| T4 | held-out quality, performance, long context | M19–M21 |
| T5 | product/release/MTP | M22–M25 |

## Change-based policy

- Compiler-only: T0, relevant T1/T2, one 12B smoke. Full conversion only at the milestone gate.
- Loader/memory: T0/T2 plus real allocation and full 12B load regressions.
- CUDA arithmetic: T0–T2 plus targeted sanitizers.
- Whole-model integration: T0–T3 and full relevant 12B matrix.
- Frozen qualification/release: T4/T5 once per exact artifact/binary hash.

## Base head tests

Lookup, tied pointer, T=1 projection, softcap, suppression, deterministic tie, diagnostic logits and sampling handoff. T=3/T=5 are not base-path tests.

## Q4_0 tests

Only external-reference checks are required on the base path. Native Q4_0 codec/backend tests run only if M24 is active.

## 26B slot tests

One-slot success and second-slot rejection. Positive multi-slot scaling is not required.

## MTP tests — M25 only

Assistant compatibility, proposal modes, multi-row Target verification, exact ordinary/MTP identity, transactional KV/hidden/RNG/repetition commit, stop/tail/ring-wrap, no allocation, memory admission, acceptance and speed.

## Determinism and sanitizers

Use repeated same-process and fresh-process checks on promoted deterministic paths. Run memcheck/racecheck/initcheck on affected CUDA suites. Retain exact commands and logs.
