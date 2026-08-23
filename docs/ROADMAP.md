# Gem16 roadmap

## Current priority: Gemma 4 26B Fast Track

The active execution policy is [`ACTIVE_DECISIONS.md`](ACTIVE_DECISIONS.md). The detailed current task entry is
[`plans/gemma4-26b/ACTIVE_CONTRACT.md`](plans/gemma4-26b/ACTIVE_CONTRACT.md). Historical decisions and benchmark
ledgers remain evidence, not default task instructions.

M00–M17 of the Gemma 4 26B track are accepted. The frozen experimental text-only SM120 profile now enters base
qualification and product integration:

```text
accepted artifact/loader → real 32K one-slot fit → deterministic reference
→ accepted optimized M17 runtime → M22 → bounded prefill → clean freeze → M21 → M20 → technical M23 freeze
→ M25 MTP/performance work → deferred M19 release-quality gate
```

The repository has an executable, fixed-address, all-resident 26B runtime with native SM120 MoE/head execution,
controlled FP8 attention, grouped prefill and whole-model decode graph replay. The latest bounded 16K+64 ordinary
decode characterization reaches 120.398 tok/s median versus llama.cpp 119.494 tok/s, but formal M20 telemetry is
still pending. The profile remains experimental: full held-out quality, controlled performance, real 64K execution
and CLI/server product gates are not yet accepted.

## 26B checkpoints

- **M06:** accepted native NVFP4 expert compiler and complete QAT expert conversion;
- **M07:** accepted provisional NVFP4 tied head compiler/reference stage;
- **M08–M09:** accepted complete artifact/direct loader and one-slot 32K/64K residency;
- **M10–M13:** accepted semantic oracles, CUDA references and early full-model quality screen;
- **M14–M17:** accepted native operators and optimized all-resident runtime;
- **M18:** conditional source/quantizer/head diagnosis only when triggered;
- **M19:** bounded Q4 numerical comparison passes; the remaining multi-hour task/prose suite is owner-deferred;
- **M20–M22:** active performance, context and CLI/server product work;
- **M23:** after M20–M22, freezes a technical Target with M19 explicitly pending and no shipping-quality claim;
- **M25:** post-freeze MTP compatibility, exactness and separate MTP context qualification;
- **deferred release gate:** M19 must eventually pass before production-quality or shipping claims.

M22 closes first. A bounded profile-driven prefill decision then produces the final candidate. M21 context runs
first, and M20 consumes the matching M21 evidence, avoiding invalidation and duplicate full evidence runs. No
multi-hour broad quality benchmark is authorized in this wave. Internal Q4_0 work, positive
multi-slot admission, Studio polish and vision do not block the base vertical path. Vision is outside the 26B Fast
Track. The 12B production path and its existing contracts remain regression-protected and unchanged.
