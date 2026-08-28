# Gemma 4 26B A4B in Gem16

**Status:** M00–M17, M20–M23 and M25 accepted; qualified text-only 26B checkpoint publication is active. The active policy is [`ACTIVE_DECISIONS.md`](ACTIVE_DECISIONS.md), and
the current task entry is [`plans/gemma4-26b/ACTIVE_CONTRACT.md`](plans/gemma4-26b/ACTIVE_CONTRACT.md).

## Goal

Deliver a qualified text-only Gemma 4 26B A4B execution on one approximately 16 GB NVIDIA Blackwell GPU. The
current path directly loads the compiled QAT-derived FP8/NVFP4 artifact, owns one resident slot and executes native
SM120 prefill/decode/head work without CPU weight offload or recurring token-loop allocation.

The current source tree contains an executable fixed-address 26B runtime, chat/server/Studio integration and native
SM120 MoE, attention, tied-head and fixed-D2 MTP dispatch. M20 accepted retained medians of 6,572.809 prompt and
150.615 ordinary-decode token/s. M25 publishes `mtp_max_context=86,016`; the Target-only maximum remains 98,304.
The 12B Unified path remains the multimodal default and must remain unchanged.

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
→ M25 MTP → qualified Target/Assistant publication → Main promotion
```

M00–M17, M20–M23 and M25 are accepted. The owner accepted full GSM8K and AIME 2026 plus bounded sampled/product
evidence as the checkpoint quality gate and waived the broader historical M19 suite. The qualified claim remains
limited to text-only SM120, one resident slot, fixed D2 and the published context limits. Target and Assistant are
distributed as separately pinned Hugging Face repositories; see
[`GEMMA4_26B_HUGGING_FACE.md`](GEMMA4_26B_HUGGING_FACE.md).

## Scope

The first profile is text-only, batch one, one fully resident 26B slot, FP8 attention/KV, NVFP4 experts/shared MLP and
a provisional NVFP4 tied head. MTP starts only after the technical base Target is frozen and requires a separately
validated assistant and memory/context qualification. Vision is a separate later track. The 12B CLI/server/runtime
behavior remains regression-protected.
