# Session ownership and concurrency specification

## First release policy

The primary 26B profile targets batch-one interactive inference. Continuous batching is not required. Multiple resident sessions are allowed only when admission proves they fit.

On a 16 GB card, the expected default is one 32K slot.

## Ownership

```text
Process
  ModelRuntime (one immutable target)
    ExecutionSlot 0
      SessionState A
    ExecutionSlot 1
      SessionState B
```

Weights are shared. KV and mutable workspaces are isolated.

## Slot construction

A slot receives:

- model traits;
- immutable tensor bindings;
- selected context;
- KV mode;
- sampling capability;
- no assistant/media capability.

It allocates all mutable memory before admission.

## Session identity

A resident conversation retains:

- exact token prefix;
- pending not-yet-forwarded token state;
- KV positions;
- sampling RNG/repetition;
- response-channel parser;
- model/artifact identity.

A continuation must extend the exact prefix. Model/profile changes invalidate the session.

## Concurrency

Two slots may execute concurrently only after:

- memory admission;
- independent streams/graphs;
- no shared mutable scratch;
- correct cancellation;
- measured contention.

Server defaults should not imply that multiple slots are efficient or supported merely because weight sharing exists.

## LRU/admission

- active sessions never evicted;
- inactive session eviction releases mutable slot state according to existing policy;
- resource exhaustion returns visibly;
- no CPU offload;
- no silent context reduction.

## Graph addresses

Each slot owns graph-bound addresses. Sharing a graph executable across slots is allowed only if node parameters/addresses can be updated safely and benchmarked. Simpler first design: one captured graph per slot.

## Compiler/model lifetime

The compiler never runs inside the server. ModelRuntime loads a verified final artifact.

## Tests

- two sessions with distinct prompts and deterministic outputs;
- no KV/router state leakage;
- cancellation one lane while another runs;
- session eviction/recreation;
- memory counters;
- impossible second slot rejected;
- shared weight pointer identity;
- mutable pointer inequality;
- 12B MTP/session behavior unchanged.
