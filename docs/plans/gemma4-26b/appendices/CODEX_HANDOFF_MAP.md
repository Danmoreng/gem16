# Coding-agent handoff map — Fast Track R4

## Current parallel handoffs

- Compiler lane: M06 → M07 → M08 integration.
- MoE semantics lane: M10 phase A/B → M11.
- Attention lane: M12 phase A/B.
- Memory lane: M09 phase A → final reconciliation after M08.
- Harness lane: future validation/report tooling without claims.
- MTP feasibility lane: M25 phase A assets/inventory/memory only.

After M13 passes, M14/M15/M16 hand independent commits to the M17 integration owner. M19/M20/M21/M22 operate on one frozen M17 artifact. M23 freezes the base target; M25 completes MTP.

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
