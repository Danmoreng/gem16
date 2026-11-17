# Retained 12B Decode optimizations — 2026-11-16

The owner approved publishing these two measured patches after stopping the
five-experiment campaign. The final direct original-parent/cumulative-package
comparison was interrupted. **There is no qualified combined speedup.**
The previously provisional Assistant patch is published by this explicit owner
decision; its original assessment and samples remain unchanged.

## Changes and measured effects

| Change | Full Ordinary throughput | Full fixed-D2 throughput | Decision |
|---|---:|---:|---|
| Post-attention rounding/norm/residual/NVFP4 quantization fusion | +0.708245% | +0.06491%, effectively unchanged | Retained |
| Assistant BF16 GEMV with exact original reduction tree | −0.01550%, effectively unchanged | +0.393532% | Retained with owner publication approval |

Each row compares its own pinned parent and candidate. The Assistant parent
already includes the fusion. Different rows/modes are not pooled or added.
Gains use ratios of mean decode durations; descriptive throughput medians and
all retained individual samples are in the linked JSON files.

Fusion Ordinary AB/BA gains are +0.744559% / +0.671935%, with respective 95%
intervals [+0.721992%, +0.764807%] / [+0.658173%, +0.684220%].
Assistant D2 AB/BA gains are +0.343685% / +0.443385%, with intervals
[+0.263057%, +0.418951%] / [+0.394455%, +0.488106%].
The Assistant did not meet the former individual 0.5% minimum; that historical
result is preserved. Both experiments pass the owner's Decode lower 95% margin
of −2% and Prefill/inverse-TTFT margin of −1%.

- [Fusion samples, source/model/binary identities, dispatch and regressions](../../../artifacts/gemma4_12b/norm-fusion-qualified-2026-11-16.json)
- [Assistant samples, counters, safety, native sequence and regressions](../../../artifacts/gemma4_12b/assistant-exact-tree-2026-11-16.json)

The fusion selects only the 12B T1/H3840 post-attention checkpoint-FP8 path
without state capture. BF16 rounding boundaries, both RMS reduction trees,
residual addition, FP4 payload and FP8 scale bytes are preserved. Other paths
use the existing implementation.

The Assistant changes only five GEMV shapes (output, contracting):
(4096,1024), (8192,1024), (1024,4096), (1024,8192), (3840,1024).
BF16 pair loads, each thread's FMA sequence and the original 256-thread
addition tree remain exact. Shared-memory communication uses one CTA barrier
instead of nine source barrier sites. PreProjection/head and other shapes
retain their previous dispatch.

## Measurement and validation boundary

Hardware: RTX 5080 **Laptop** GPU, 16 GB / 60 SMs. Production toolchain:
CUDA 13.4.92, GCC 16.2.1, CMake 4.5, Release-O3/C++20/SM120a, RDC enabled.
The measured worktree is based on 58febf4d66bb028a35f6f7f9528d2206cfab2475;
exact measured source/binary/model hashes are preserved in each artifact.
The newer upstream release/documentation commits contain no runtime changes.

Each retained patch has full Ordinary and D2 AB/BA cohorts, three warmups and
ten retained full requests per cohort. Requests use identical 16,384 prompt IDs,
1,135 outputs / 1,134 decode intervals, requested context 17,519, batch 1, greedy,
suppression 1/50/106/258882/258883, FP8 KV and 8192-token Prefill chunks without
prefix reuse. Primary throughput excludes profilers, observers and checkpoint
loading. Native `prompt_ms` includes the first selected token, so fixed-prompt
Prefill-throughput and inverse-inference-TTFT ratios coincide.

Existing checks, reused without launching new benchmarks for publication:

- Fusion: 132 actual/raw-O/finite/alias/producer comparisons, full memcheck,
  synccheck and racecheck. Fixture alignment uses layer/decode phase/family;
  raw unrounded O outputs are restored before each replay. The producer chain
  uses the unchanged native O projection.
- Assistant: 126 actual calls across all 42 GEMVs in early/middle/late D2 groups;
  882 numeric/finite/guard comparisons, memcheck and synccheck. Racecheck passes
  all 126 actual full-grid changed-kernel calls. Two earlier all-value-repeat
  attempts timed out and are not passes.
- Full 1,135 output IDs, all 1,014 D2 proposals, 507 groups / 626 accepted tokens and
  the entire 311,287,808-byte target KV arena match. KV SHA256:
  `6b03676412022f6be775dfec7ec4e9e68ef3d997a28e6f5f3f100ff36c577231`.
- Production dispatch/instruction/resource evidence is recorded separately from
  laboratory derivatives. Reported arenas and sampled device peaks match.
  The fusion's static 2 KiB versus loaded 1 KiB shared-memory readings are both
  disclosed; the conservative budget is 2 KiB.
- Relevant 12B text/image/audio/session and 26B Compact Vision/internal NVFP4
  regressions, plus host and CUDA tests, passed on Linux. Live media checks
  compare responses; they do not claim raw media token-ID coverage.

The measured baseline also contains the earlier local Prefill-chunk diagnostic
control; this publication excludes that unrelated change. The control was
unset and resolves to the same 8192-token default used by the existing runtime.
Consequently, archived binary hashes identify the measured worktree, not a
fresh binary built from this publication commit.

## Limits and stopped work

All five individual experiments completed. QKV 64-bit reconstruction, Down
weight pipelining and global-attention preloading were rejected and restored.
Their runtime code is not included. The final cumulative test stopped during
its first candidate Ordinary cohort; partial samples are not an accepted
comparison. The planned 4096/8192 Prefill profile is unrun and is not restarted.

The existing Prefill CUTLASS wrappers repack weights into transient scratch
on each request/chunk. That inherited behavior conflicts with the permanent
no-recurring-repack rule; these Decode changes do not introduce or repair it.
It is not hidden by the reported zero persistent-repack bytes.

This is one fixed workload and Linux execution evidence. No new Windows run,
release qualification, reference-engine rerun or claim of a smaller llama.cpp/
vLLM gap follows from this publication. No benchmark is launched by this task.
