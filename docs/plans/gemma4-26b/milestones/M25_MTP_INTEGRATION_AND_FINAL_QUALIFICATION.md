# M25 — 26B MTP integration and final qualification

Status: ready; technical M23 Target accepted, bounded external 26B MTP characterization runs before integration
Class: required final target

Normative inputs: [MTP platform contract](../../../MTP.md), [Tied embedding/head](../specs/EMBEDDING_HEAD_SPEC.md), [Memory arena](../specs/MEMORY_ARENA_SPEC.md), [Session ownership](../specs/SESSION_OWNERSHIP_AND_CONCURRENCY.md), [API/CLI changes](../specs/API_CLI_CHANGES.md), [Test matrix](../specs/TEST_MATRIX.md).

## Outcome

Add exact Target-verified MTP to the frozen 26B base target and qualify the highest safe MTP context on the 16 GB GPU.

## Phase A — early feasibility, parallel from M06

- identify and immutably lock a target-compatible 26B assistant source;
- validate tokenizer, vocabulary, hidden interfaces, layer/cache dependencies and maximum positions;
- inventory exact tensors and calculate BF16/FP8/NVFP4 resident sizes;
- model 32K/64K base+assistant+verifier memory;
- determine whether the assistant can read Target KV without an independent long-context cache;
- record any asset incompatibility early.

This phase may add docs, locks, inventory tools and fixtures only. It does not edit the base runtime.

## Phase B — assistant artifact and loader

After M23, compile a separately locked assistant artifact if direct BF16 residency is not viable. Prefer quantization that preserves draft acceptance while fitting the target margin. Keep target and assistant provenance separate.

## Phase C — exact verification runtime

- proposal lengths selected from the existing verified design (for example D1/D2/D4), based on measured benefit;
- multi-row Target verification and any T>1 head kernels are implemented here;
- proposals never directly determine emitted output;
- tentative Target K/V, hidden, RNG and repetition state commit transactionally;
- ordinary and MTP output IDs match under the same deterministic controls;
- no token-loop allocation or host-driven per-token decision.

## Phase D — memory, context and performance

- qualify ordinary versus MTP speed and acceptance;
- pass 32K with at least 700 MiB free-device margin; if this fails, M25 fails, the M23 base profile remains supported and the blocker is recorded;
- attempt 64K; advertised MTP contexts at 64K or above require at least 500 MiB free-device margin;
- determine `mtp_max_context` independently from `base_max_context`;
- ensure enabling MTP never silently reduces the requested context.

## Exit gate

- [ ] Compatible assistant source/artifact is locked and validated.
- [ ] Assistant plus verifier memory is fully named and admitted at 32K with at least 700 MiB free-device margin.
- [ ] Exact ordinary/MTP output identity and transactional state tests pass.
- [ ] MTP provides a measured benefit for at least one supported mode; otherwise M25 fails and the M23 base profile remains the supported result.
- [ ] 64K has an explicit pass/fail result; `mtp_max_context`, acceptance and speed are published honestly.
- [ ] Base 26B and 12B paths remain unchanged when MTP is disabled.
- [ ] Deferred M19 is accepted before program-complete, shipping or production-quality status is claimed.

Vision is explicitly excluded and belongs to a separate future program.
