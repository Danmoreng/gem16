# Milestone status board

Update this file in the working copy. Do not mark a milestone complete from prose alone; link its evidence.

Development branch for M03-M25: `feat/gemma4-26b`

| Milestone | Status | Code commit | Artifact/evidence | Owner sign-off |
|---|---|---|---|---|
| M00 Policy/bootstrap | ✓ | `3bf7b6e` | [M00 policy review](../../evidence/gemma4_26b/m00-policy-review.md) | accepted 2026-08-06 |
| M01 Source locks/goldens | ✓ | `59996f5` | [M01 handoff](../../evidence/gemma4_26b/m01-source-locks-and-goldens-2026-08-06.md) | accepted 2026-08-06 |
| M02 Model variants | ✓ | `f79eeb6` | [M02 handoff](../../evidence/gemma4_26b/m02-model-variants-2026-08-06.md) | accepted 2026-08-06 |
| M03 Tensor inventory | ✓ | `06b72e4` | [M03 handoff](../../evidence/gemma4_26b/m03-manifest-and-inventory-2026-08-11.md) | accepted 2026-08-11 |
| M04 Compiler scaffold | ✓ | `edd80cb` | [M04 handoff](../../evidence/gemma4_26b/m04-checkpoint-compiler-scaffold-2026-08-11.md) | accepted 2026-08-11 |
| M05 FP8 compiler | ✓ | `d913881` | [M05 acceptance](../../evidence/gemma4_26b/m05-fp8-attention-compiler-acceptance-2026-08-11.md) | accepted 2026-08-11 |
| M06 NVFP4 compiler | ☐ | | | |
| M07 Head experiment | ☐ | | | |
| M08 Artifact/loader | ☐ | | | |
| M09 Memory planner | ☐ | | | |
| M10 CPU MoE oracle | ☐ | | | |
| M11 CUDA MoE reference | ☐ | | | |
| M12 Attention/KV | ☐ | | | |
| M13 Full reference model | ☐ | | | |
| M14 Native MoE decode | ☐ | | | |
| M15 Grouped MoE prefill | ☐ | | | |
| M16 Quantized head | ☐ | | | |
| M17 Optimized integration | ☐ | | | |
| M18 Converter A/B | ☐ | | | |
| M19 Quality qualification | ☐ | | | |
| M20 Performance qualification | ☐ | | | |
| M21 Long context | ☐ | | | |
| M22 Product integration | ☐ | | | |
| M23 Release qualification | ☐ | | | |
| M24 Optional Q4 backend | ☐ | | | |
| M25 Future MTP/vision | ☐ | | | |

Status values:

```text
☐ not started
◐ in progress
⚠ blocked
✓ passed
✗ rejected/stopped
```

## Current milestone state

```text
M05 — accepted; M06 is dependency-unblocked but not started
```

The project owner accepted M05 on 2026-08-11 at implementation commit `d913881`. The exact Ordinary/QAT plans,
versioned native C++20 encoder/comparator, clean full conversion per source, structural verification, complete
hashes, semantic reports, native comparison, bounded host/sanitizer/CUDA gates and exact 12B regression pass. Python
is control-plane/oracle support only and cannot replace the production native data plane. Clean evidence is retained
in [M05 acceptance](../../evidence/gemma4_26b/m05-fp8-attention-compiler-acceptance-2026-08-11.md); earlier dirty runs
remain labeled diagnostic history. The M05 partial artifact is non-runtime-loadable. M06 is dependency-unblocked but
has not started. The cross-milestone architecture remains
[NATIVE_CONVERTER_ARCHITECTURE.md](specs/NATIVE_CONVERTER_ARCHITECTURE.md), with version-scoped llama.cpp research
retained in [M05 evidence](../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md).
