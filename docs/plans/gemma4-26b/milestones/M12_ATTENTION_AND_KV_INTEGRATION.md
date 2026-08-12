# M12 — 26B attention, RoPE and KV

Status: phase A parallel-ready; runtime phase waits for M08/M09
Class: parallel-critical
Unblocks: M13

Normative inputs: [Attention/KV](../specs/ATTENTION_KV_SPEC.md), [Model variant traits](../specs/MODEL_VARIANT_TRAITS_SPEC.md).

## Phase A — start now

- validated 30-layer local/global trait table;
- Q/K/V/O shape fixtures and missing-V semantics;
- RoPE fixtures including 1023/1024 and long positions;
- local ring and global extent ownership tests;
- exact cache-byte formulas.

## Phase B — after M08/M09

- bind real FP8 weights and separate K/V caches;
- implement local/global reference execution;
- validate producer/consumer ownership and any cross-layer sharing;
- reconcile actual bytes with M09.

## Exit gate

- [ ] All layer traits and tensor bindings are validated.
- [ ] Local/global reference comparisons pass.
- [ ] Final K and V are physically distinct where required.
- [ ] Ring wrap and global append/read pass.
- [ ] Cache bytes match the memory planner at 8K, 32K and 64K.
- [ ] 12B attention/long-context tests remain green.
