# Repository touchpoints — Fast Track R4

Accepted baseline: M05 implementation commit `d91388113d68974f9ab7cec1a90ef768285c0645`. Inspect the current tree before relying on paths.

## Parallel ownership

| Lane | Typical paths |
|---|---|
| M06/M07 compiler | `src/compiler/**`, native compiler CLI/tests, `tools/gem16_compile/**` |
| M10 oracle | new numeric MoE oracle and tests |
| M09 memory | `src/runtime/memory_plan*`, tests/tools |
| M12 traits/attention | model traits and dedicated attention/KV/RoPE tests |
| M11/M14/M15 CUDA MoE | `src/cuda/moe/**` |
| M13/M17 integration | target model/inference engine; integration-owner only |
| M19–M21 harness | validation/evaluation/benchmark tools/prompts |
| M22 product | CLI/server; Studio optional |
| M25 MTP | assistant compiler/loader, `src/cuda/mtp/**`, verifier integration |

## Current reusable assets

- M04 Python compiler control plane;
- M05 native FP8 batch backend;
- host/runtime NVFP4 codecs and SM120 layout helpers;
- existing T=1 and batched projection primitives;
- output-head softcap/suppression/tie semantics;
- 12B MTP exact-verification infrastructure.

Reuse contracts, not hard-coded 12B shapes. Base M07/M16 need T=1 only. Multi-row output-head work is deferred to M25.

Shared public structs and inference-engine orchestration require an integration-owned interface commit before parallel lanes edit against them.
