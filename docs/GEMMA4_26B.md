# Gemma 4 26B A4B in Gem16

**Status:** M00–M05 accepted; M06 active. The active policy is [`ACTIVE_DECISIONS.md`](ACTIVE_DECISIONS.md), and
the current task entry is [`plans/gemma4-26b/ACTIVE_CONTRACT.md`](plans/gemma4-26b/ACTIVE_CONTRACT.md).

## Goal

Reach an experimental text-only Gemma 4 26B A4B execution on one approximately 16 GB NVIDIA Blackwell GPU as
quickly as possible. The first useful result is a directly loadable QAT-derived FP8/NVFP4 artifact, one resident
slot and real 32K execution with at least 700 MiB directly measured free CUDA memory. Later work qualifies quality,
performance, longer context and product behavior.

The current source tree does **not** yet contain an executable 26B runtime. M05 is an accepted attention-only native
FP8 compiler stage and remains non-runtime-loadable. The 12B Unified path is the production baseline and must remain
unchanged.

## Fast-track path

```text
M06 NVFP4 experts → M07 provisional NVFP4 tied head → M08 artifact/loader
→ M09 32K residency → M13 slow reference execution → M17 optimized runtime
→ base qualification → MTP
```

M06 uses one clean full QAT expert conversion, exhaustive small codec/shape/determinism and representative operator
checks, bounded-memory evidence and sampled Ordinary/Unsloth diagnostics. Full Ordinary attribution, internal Q4_0,
broad head A/B work and M18 diagnosis are conditional rather than first-path blockers. Native execution may be
experimental before final qualification, but must disclose missing gates and never silently fall back or offload.

## Scope

The first profile is text-only, batch one, one fully resident 26B slot, FP8 attention/KV, NVFP4 experts/shared MLP and
a provisional NVFP4 tied head. MTP starts only after the base target is frozen and requires a separately validated
assistant and memory/context qualification. Vision is a separate later track. The 12B CLI/server/runtime behavior
remains regression-protected.
