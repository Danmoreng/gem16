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
| M05 FP8 compiler | ◐ | | [M05 diagnostic runs](../../evidence/gemma4_26b/m05-native-fp8-implementation-and-diagnostic-runs-2026-08-11.md) | diagnostic full runs passed; clean acceptance pending |
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
M05 — current/in progress; M06+ remain dependency-gated
```

The project owner explicitly authorized starting M05 on 2026-08-11 after first pausing it; that instruction
supersedes the pause. M00-M04 remain accepted. The exact Ordinary/QAT plans and plan gates pass. The Python codec is reference/oracle support; the promoted/native
C++20 batch backend exists in the dirty worktree. A short native throughput probe, one diagnostic full Ordinary run,
one diagnostic full QAT run, structural verification, the weight-only comparison and the CUDA/12B regression gates
are retained in [M05 diagnostic evidence](../../evidence/gemma4_26b/m05-native-fp8-implementation-and-diagnostic-runs-2026-08-11.md).
All currently retained M05 compiler artifacts record `compiler_dirty=true`, so these are not an accepted M05 gate.
The owner has authorized the implementation commit and clean-revision evidence runs. Remaining gates are one clean
native Ordinary plus one clean native QAT run, final verification/review and final hashes/status. M05
remains non-runtime-loadable and M06+ remain dependency-gated. The cross-milestone converter
architecture is documented in [NATIVE_CONVERTER_ARCHITECTURE.md](specs/NATIVE_CONVERTER_ARCHITECTURE.md), with the
version-scoped llama.cpp research retained in [M05 evidence](../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md).
