# M23 — Technical base Target freeze and rollback

Status: blocked by M20/M21; M22 is accepted and full M19 is owner-deferred
Class: technical base checkpoint; not a shipping release while M19 is pending

Normative inputs: [Checkpoint provenance](../specs/CHECKPOINT_PROVENANCE_SPEC.md), [Test matrix](../specs/TEST_MATRIX.md), [Telemetry artifact](../specs/TELEMETRY_ARTIFACT_SPEC.md).

## Outcome

Freeze a stable engineering Target for subsequent MTP and performance work without duplicating unchanged
qualification work. Carry the deferred M19 limitation explicitly.

## In scope

- verify all available reports reference the same artifact, binary and configuration hashes;
- rerun only evidence invalidated by a changed hash or relevant runtime component;
- package locks, engineering notes, capability statement and rollback;
- record default context and `base_max_context`;
- preserve raw evidence.

## Exit gate

- [ ] Hash/evidence reconciliation is complete.
- [ ] Performance, 32K/64K and product gates are accepted.
- [ ] Deferred M19 and the prohibition on shipping/production-quality claims are explicit in every capability statement.
- [ ] Rollback to the prior supported profile is documented.
- [ ] The base target is frozen as the M25 Target.

M23 is an engineering checkpoint and may become the M25 Target. It is not a valid shipping checkpoint until the
deferred M19 exit gate passes. Program completion requires both M25 and final M19 acceptance.
