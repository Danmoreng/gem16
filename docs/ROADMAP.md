# Gem16 roadmap

## Current priority: productization baseline

The active execution policy is [`ACTIVE_DECISIONS.md`](ACTIVE_DECISIONS.md). The owner-approved scope is
[`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md), and the coding-agent API boundary is
[`OPENAI_AGENT_CORE_V1.md`](OPENAI_AGENT_CORE_V1.md). Historical milestone documents and benchmark ledgers remain
evidence, not default task instructions.

The next product increments are, in order:

1. keep `AGENTS.md`, active decisions, README, Studio/server docs, and the central `VERSION` consistent;
2. provide one equal native model-management flow for installing, verifying, selecting, and removing either public
   12B Unified or 26B Compact Vision side by side while retaining NVFP4 internally;
3. qualify OpenAI Agent Core v1 with the official Python and JavaScript/TypeScript SDKs and one unmodified external
   coding-agent workflow on both model profiles and both product platforms;
4. produce equally supported Windows and Linux archives with exact dependency contracts, licenses/notices,
   manifests, hashes, and clean-machine smoke evidence;
5. harden installers, signing, updates, and release automation after the portable two-platform gate passes.

The Kotlin/Compose GUI is deprecated. All current UI and release work belongs in `nativeStudio/`. The 26B decode
optimization phase is frozen; new decode tuning requires a separate owner decision.

## Completed Gemma 4 26B qualification track

M00–M17, M20–M23 and M25 of the Gemma 4 26B track are accepted. The former public text-only SM120 profile remains an
internal qualified rollback path. The two public choices are 12B Unified and 26B Compact Vision; their components are
published from immutable Hugging Face revisions and Studio installs them into verified cache snapshots.

```text
accepted artifact/loader → accepted optimized M17 runtime → accepted M22 product
→ accepted M20/M21 performance and context → accepted M23 freeze
→ accepted M25 fixed-D2 product → split immutable publication → accepted Main promotion
```

The repository has an executable, fixed-address, all-resident 26B runtime with native SM120 MoE/head execution,
controlled FP8 attention, grouped prefill and whole-model decode graph replay. The qualified M25 profile is text-only,
batch one, one resident session, fixed D2, FP8 KV, and supports up to 86,016 MTP context tokens with a 200 MiB reserve.
The separate Target-only ceiling is 98,304 tokens. The owner accepted the bounded GSM8K/AIME and product evidence and
waived the remaining broad M19 suite for this checkpoint; claims do not extend beyond that recorded evidence.

## 26B checkpoints

- **M06:** accepted native NVFP4 expert compiler and complete QAT expert conversion;
- **M07:** accepted provisional NVFP4 tied head compiler/reference stage;
- **M08–M09:** accepted complete artifact/direct loader and one-slot 32K/64K residency;
- **M10–M13:** accepted semantic oracles, CUDA references and early full-model quality screen;
- **M14–M17:** accepted native operators and optimized all-resident runtime;
- **M18:** conditional source/quantizer/head diagnosis only when triggered;
- **M19:** full GSM8K/AIME plus bounded sampled/product evidence accepted as the owner-approved replacement; the
  remaining broad multi-hour task/prose suite is waived for this checkpoint;
- **M20–M21:** accepted fixed-target performance and context work; M20 passed 6,000 prompt / 150 ordinary-decode
  token/s and the 6,500 prompt stretch, while M21 established `base_max_context=98,304`;
- **M22:** accepted CLI/server product integration and protected 12B behavior at `f0aa302aa0246d44e1c8477dbbbb67fbbe2d2037`;
- **M23:** accepted technical Target freeze after M20/M21 and M22;
- **M25:** accepted fixed-D2 compatibility, exactness, product and 86,016-token MTP context qualification;
- **Main promotion:** accepted after current-HEAD release, CUDA/sanitizer, real 12B/26B product and server lifecycle
  revalidation.

A bounded profile-driven prefill/decode optimization slice produced the promoted candidate. M21 context ran first,
and M20 consumed the matching M21 evidence, avoiding invalidation and duplicate full evidence runs. No multi-hour
broad quality benchmark is required for this checkpoint. Internal Q4_0 work, positive
multi-slot admission, Studio polish and vision do not block the base vertical path. Vision is outside the 26B Fast
Track. The qualified 12B path and its existing contracts remain regression-protected and unchanged.

## Retained model-specific backlog

### Reduce Gemma 4 26B server startup latency

Do not change the loader or compiled artifact before the currently planned M19 quality benchmark. Afterwards, remove
the dominant startup-time weight transformation from the 26B path:

- make the offline checkpoint compiler emit the exact final SM120 Row8/K64 weight and scale byte order consumed by
  the runtime, while preserving the immutable-source provenance, hashes, strict validation and single persistent
  device representation;
- replace the current CPU tiling through the 4 MiB staging buffer with validated direct or large contiguous uploads
  into the final GPU arena; do not move quantization or another transformation into server startup;
- add phase timings for manifest validation, file/page-cache I/O, arena allocation, layout work, host-to-device
  transfer, workspace/KV initialization, CUDA Graph construction and server admission;
- flush or line-buffer startup progress so Studio and redirected benchmark logs do not appear hung while loading.

The motivating 2026-08-26 GSM8K smoke run measured approximately 73.6 seconds from gem16 process start to health
readiness, versus approximately 4.1 seconds from llama.cpp's logged model-load start to listener readiness. The
current compiled artifact contains 14,696,569,196 payload bytes; 13,562,689,536 bytes (92.3%) take one of the
load-time SM120 tiled paths. Treat those observations as diagnostic evidence, not a controlled startup benchmark.

Acceptance requires byte-identical final device tensors and unchanged generation/correctness results, no extra
persistent host or device weight copy, no startup-time quantization, preserved 12B behavior, and a separately
recorded cold-cache and warm-cache before/after startup measurement on the reference machine.
