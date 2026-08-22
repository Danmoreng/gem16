# Gem16 roadmap

## Current priority: Gemma 4 26B Fast Track

The active execution policy is [`ACTIVE_DECISIONS.md`](ACTIVE_DECISIONS.md). The detailed current task entry is
[`plans/gemma4-26b/ACTIVE_CONTRACT.md`](plans/gemma4-26b/ACTIVE_CONTRACT.md). Historical decisions and benchmark
ledgers remain evidence, not default task instructions.

M00–M17 of the Gemma 4 26B track are accepted. The frozen experimental text-only SM120 profile now enters base
qualification and product integration:

```text
accepted artifact/loader → real 32K one-slot fit → deterministic reference
→ accepted optimized M17 runtime → M19/M20/M21/M22 → M23 base freeze
```

The repository has an executable, fixed-address, all-resident 26B runtime with native SM120 MoE/head execution,
controlled FP8 attention, grouped prefill and whole-model decode graph replay. It remains experimental: held-out
quality, controlled performance, real 64K execution and CLI/server product gates are not yet accepted.

## 26B checkpoints

- **M06:** accepted native NVFP4 expert compiler and complete QAT expert conversion;
- **M07:** accepted provisional NVFP4 tied head compiler/reference stage;
- **M08–M09:** accepted complete artifact/direct loader and one-slot 32K/64K residency;
- **M10–M13:** accepted semantic oracles, CUDA references and early full-model quality screen;
- **M14–M17:** accepted native operators and optimized all-resident runtime;
- **M18:** conditional source/quantizer/head diagnosis only when triggered;
- **M19–M22:** ready quality, performance, context and CLI/server product work;
- **M23:** blocked only on acceptance of M19–M22, then freezes the base profile;
- **M25:** post-freeze MTP compatibility, exactness and separate MTP context qualification.

M19–M22 may proceed in disjoint slices against the same frozen M17 profile. Internal Q4_0 work, positive multi-slot
admission, Studio polish and vision do not block the base vertical path. Vision is outside the 26B Fast Track. The
12B production path and its existing contracts remain regression-protected and unchanged.
