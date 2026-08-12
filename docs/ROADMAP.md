# Gem16 roadmap

## Current priority: Gemma 4 26B Fast Track

The active execution policy is [`ACTIVE_DECISIONS.md`](ACTIVE_DECISIONS.md). The detailed current task entry is
[`plans/gemma4-26b/ACTIVE_CONTRACT.md`](plans/gemma4-26b/ACTIVE_CONTRACT.md). Historical decisions and benchmark
ledgers remain evidence, not default task instructions.

M00–M07 of the Gemma 4 26B track are accepted. Work is paused before M08 by owner request. The priority remains the vertical experimental result:

```text
QAT-derived NVFP4/FP8 artifact → direct loader → real 32K one-slot fit
→ first deterministic reference execution → optimized runtime → later qualification
```

The current repository still has no executable 26B runtime path. Existing 26B source classification, inventories,
memory estimates and the accepted native M05 FP8/M06 NVFP4 compiler stages are preparation only. Do not describe
the partial artifacts as loadable or 26B execution as implemented.

## 26B checkpoints

- **M06:** accepted native NVFP4 expert compiler and complete QAT expert conversion;
- **M07:** accepted provisional NVFP4 tied head compiler/reference stage;
- **M08:** next complete artifact and direct loader stage, not started;
- **M09:** one fully resident 26B slot at 32K with at least 700 MiB directly measured free CUDA memory;
- **M13:** slow deterministic text execution and the early quality screen;
- **M17:** optimized all-resident runtime;
- **M19–M23:** later quality, performance, context and base-profile qualification;
- **M25:** post-freeze MTP compatibility, exactness and separate MTP context qualification.

M10 and M12 fixture/oracle work may proceed in disjoint slices. M18 is conditional diagnosis; internal Q4_0 work,
positive multi-slot admission, Studio polish and vision do not block the first vertical path. Vision is outside the
26B Fast Track. The 12B production path and its existing contracts remain regression-protected and unchanged.
