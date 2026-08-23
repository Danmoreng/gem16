# Gemma 4 26B A4B in Gem16

**Status:** M00–M17 and M22 accepted; bounded prefill/decode optimization, M21/M20 and a technical M23 freeze are next. The active policy is [`ACTIVE_DECISIONS.md`](ACTIVE_DECISIONS.md), and
the current task entry is [`plans/gemma4-26b/ACTIVE_CONTRACT.md`](plans/gemma4-26b/ACTIVE_CONTRACT.md).

## Goal

Deliver an experimental text-only Gemma 4 26B A4B execution on one approximately 16 GB NVIDIA Blackwell GPU. The
current path directly loads the compiled QAT-derived FP8/NVFP4 artifact, owns one resident slot and executes native
SM120 prefill/decode/head work without CPU weight offload or recurring token-loop allocation.

The current source tree contains an executable fixed-address 26B runtime, public chat/server integration and native
SM120 MoE, attention and tied-head dispatch. The latest adjacent two-run controlled 16K+64 development candidate
reaches 5,050.92 prompt tok/s and 139.106 ordinary-decode tok/s with the accepted output hash. This is not yet the
formal M20 3-warm-up/10-retained result. The 12B Unified path remains the production baseline and must remain
unchanged.

The owner-set M20 objective now targets vLLM-class performance on the exact 16K+64 ordinary path: retained medians
of at least **6,000 prompt token/s** and **150 decode token/s**, with **6,500 prompt token/s** as the non-blocking
competitive stretch target. The current adjacent development candidate is 5,050.92/139.106 token/s; formal retained
medians are pending. MTP/speculative decode remains disabled for this gate, and no prompt, cache, precision, sampling
or timing-boundary change may count as a speedup.

## Fast-track path

```text
M06 NVFP4 experts → M07 provisional NVFP4 tied head → M08 artifact/loader
→ M09 32K residency → M13 slow reference execution → M17 optimized runtime
→ accepted M22 product → bounded prefill/decode optimization → clean candidate freeze
→ M21 real 32K/64K → M20 performance → technical M23 freeze
→ M25 MTP → deferred M19 release-quality gate
```

M00–M17 and M22 are accepted. M22 now pins the exact M08 identity, exposes CLI/server provenance and memory
reporting, enforces text-only one-slot behavior and passes automated 26B product plus protected 12B regressions.
The fixed performance targets must first be reached or explicitly revised by the owner. M21 still needs repeated
real 32K plus explicit 64K execution, then M20 needs its bounded formal 3/10 run against that
matching context evidence. The owner
deferred the remaining multi-hour M19 task/prose suite until the end of the implementation/performance program.
Therefore M23 may freeze an engineering Target for later work, but it cannot be described as a shipping or
quality-qualified release while M19 is pending.

## Scope

The first profile is text-only, batch one, one fully resident 26B slot, FP8 attention/KV, NVFP4 experts/shared MLP and
a provisional NVFP4 tied head. MTP starts only after the technical base Target is frozen and requires a separately
validated assistant and memory/context qualification. Vision is a separate later track. The 12B CLI/server/runtime
behavior remains regression-protected.
