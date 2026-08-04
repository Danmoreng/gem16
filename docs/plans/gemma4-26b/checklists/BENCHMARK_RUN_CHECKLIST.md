# Benchmark run checklist

## Before run

- [ ] Clean or explicitly labeled code tree.
- [ ] Exact binary hash.
- [ ] Exact model/artifact lock.
- [ ] Competitor versions/patches verify.
- [ ] Token workload hash verifies.
- [ ] GPU idle.
- [ ] Required power/profile active.
- [ ] Start temperature criterion met.
- [ ] Host RAM/swap state recorded.
- [ ] No CPU offload.
- [ ] Native path capability verified.

## Semantics

- [ ] Batch one.
- [ ] Same input IDs.
- [ ] Same context/output count.
- [ ] Same stop/sampling behavior.
- [ ] No asymmetric prompt cache.
- [ ] KV format disclosed.
- [ ] Only selected/accepted target tokens counted.
- [ ] Output validity/checksum gate enabled.

## Measurement

- [ ] Model load separate.
- [ ] Prompt/TTFT boundary documented.
- [ ] First-token decode handling documented.
- [ ] 3 warmups.
- [ ] 10 retained runs.
- [ ] Raw ITLs retained.
- [ ] Telemetry active.
- [ ] Engines run serially.
- [ ] No profiler in retained timing.

## After run

- [ ] Deterministic outputs where expected.
- [ ] Fallback count zero.
- [ ] Token-loop allocations zero.
- [ ] Process peak and margin recorded.
- [ ] Statistics regenerated from raw values.
- [ ] Confidence intervals.
- [ ] Environment/commands/checksums stored.
- [ ] Result directory not overwritten.
- [ ] Claims and caveats match formats/boundaries.
