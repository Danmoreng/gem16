# Coding-agent handoff map — Fast Track R4

## Current parallel handoffs

- Product lane: M22 automated acceptance and 12B regressions.
- Prefill lane: profile-driven bounded optimization after M22 and before the final evidence freeze.
- Runner lane: align M20/M21 contracts and freeze a clean candidate.
- Context lane: real M21 32K/64K execution after GPU serialization.
- Benchmark lane: bounded formal M20 evidence consuming that matching M21 result.
- MTP feasibility lane: M25 phase A assets/inventory/memory only.
- Quality lane: M19 task/prose work is owner-deferred until the end.

M23 freezes a technical Target after M20–M22, with M19 explicitly pending. M25 consumes that Target; deferred M19
still gates shipping and production-quality claims.

## Handoff packet

```text
base and result commit
writable paths and shared interfaces
artifact/source hashes
exit criteria
exact test commands/results
evidence paths
memory/performance delta
known risks
merge dependency
```

Conversation summaries are not sufficient. Global status is updated by the integration owner after merge and acceptance.
