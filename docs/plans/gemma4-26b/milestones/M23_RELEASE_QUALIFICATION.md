# M23 — Base target evidence freeze and rollback

Status: blocked by M19, M20, M21 and M22
Class: base release checkpoint

Normative inputs: [Checkpoint provenance](../specs/CHECKPOINT_PROVENANCE_SPEC.md), [Test matrix](../specs/TEST_MATRIX.md), [Telemetry artifact](../specs/TELEMETRY_ARTIFACT_SPEC.md).

## Outcome

Freeze a usable base 26B text release candidate without duplicating unchanged qualification work.

## In scope

- verify all reports reference the same artifact, binary and configuration hashes;
- rerun only evidence invalidated by a changed hash or relevant runtime component;
- package locks, release notes, capability statement and rollback;
- record default context and `base_max_context`;
- preserve raw evidence.

## Exit gate

- [ ] Hash/evidence reconciliation is complete.
- [ ] Base quality, performance, 32K and product gates are accepted.
- [ ] Rollback to the prior supported profile is documented.
- [ ] The base target is frozen as the M25 Target.

M23 is a valid shipping checkpoint, but the program goal is not complete until M25.
