> Historical exploratory record. Current qualification and final dispositions
> are indexed in [the closed archive](../benchmark-archive-2026-11-17.md).
> This is not an active benchmark or optimization task.

# 12B Decode opportunities — 2026-11-15

Assessment only: no runtime patch, candidate operator timing, gain, commit or promotion.
Existing Ordinary/D2 primary results are unchanged. One fresh native D2 trace preserves
all1,135 existing output IDs. Profiling timings never replace the3+10 primary samples.

## Measured budgets

### ordinary

GPU busy union covers98.84% of the selected CPU NVTX span. Kernel shares below use summed GPU kernel time.

| Family | Share | Calls |
|---|---:|---:|
| `Sm120DirectProjection` | 34.03% | 163296 |
| `Sm120DirectProjection` | 25.53% | 108864 |
| `FusedOutputHeadCandidates` | 12.34% | 1134 |
| `RmsNormResidualBf16` | 7.48% | 108864 |
| `SplitOnlineDecodeAttentionFp8GlobalGqa` | 7.09% | 9072 |
| `SplitOnlineDecodeAttentionFp8` | 5.87% | 45360 |
| `RmsNormQuantizeToken` | 2.63% | 54432 |
| `QuantizeTokenReference` | 1.16% | 54432 |
### d2

GPU busy union covers98.35% of the selected CPU NVTX span. Kernel shares below use summed GPU kernel time.

| Family | Share | Calls |
|---|---:|---:|
| `Sm120MatrixProjection` | 17.61% | 24336 |
| `Sm120MatrixProjection` | 17.54% | 24336 |
| `Sm120DirectProjectionFixedBatch` | 14.64% | 48672 |
| `FusedOutputHeadFixedBatchCandidates` | 10.67% | 507 |
| `SplitOnlineDecodeAttentionFp8GlobalGqa` | 8.23% | 4056 |
| `SplitOnlineDecodeAttentionFp8LocalFixedSharedKv` | 5.42% | 20280 |
| `AssistantOutputCandidates` | 5.36% | 1014 |
| `Bf16Gemv` | 4.67% | 21294 |

## Proposed order

1. Scope NVFP4 fused Gate/Up's D2 short batch to the12B geometry and test oneM16 tile
   instead of the observed two. The current3-row path still executes the second,
   inactive16-row tile's MMA loop. Preserve weights/scales/K order and BF16 boundaries.
2. Test12B-specific packed FP8 T3 for QKV/O. Actual12B dispatch uses three separate MMA
   rows; the packed path exists for26B dimensions. Adapt and validate rather than
   assuming that adding a dispatch condition is sufficient or faster.
3. Diagnose real-input Ordinary FP8 QKV/O: current direct kernels dominate25.53% of
   summed decode GPU time. Measure DRAM transactions, issue stalls and occupancy before
   choosing one memory/launch geometry experiment; older eight-warp/overlap failures
   remain historical rejections, not new candidates.
4. Consider12B-specific normalization/residual/quantization fusion. Preserve every
   BF16 rounding and reduction contract; several fusions already exist, so target
   the remaining measured boundaries rather than deleting numerical steps.
5. Diagnose Assistant BF16 GEMV and vocabulary projection (combined about10% ofD2
   kernel time). Keep Assistant weights and compare group cost separately from
   acceptance; native61.74% and vLLM63.17% draft acceptance are too close to explain
   the complete20.67% throughput target by acceptance alone.

The first two are concrete source/dispatch findings, not qualified gains. Native full
CUDA Graphs and device D2 chaining already exist. The tied BF16 head is already
fused with greedy selection, and its T3 path shares one weight load across rows;
the historical12B T3 WMMA head lost about9% end-to-end and should not be repeated
without a different premise. Do not promote a runtime patch from kernel timing alone.

Any authorized implementation should test one candidate at a time: actual-input
numerical/boundary checks, memory checks, dispatch/resource evidence, then adjacent
unprofiled parent/candidate Ordinary andD2 measurements with identical workload,
output parity, VRAM and protected12B media/26B regressions. Retain only measured gains.
The standing tuning freeze is not superseded by this read-only assessment.

[Machine-readable attribution and hashes](../../../artifacts/benchmark_history/2026-11-15/12b-decode-opportunities.json).
Raw analysis/profile scripts: `benchmarks/results/2026-11-15/58febf4/12b-decode-opportunities/`.
