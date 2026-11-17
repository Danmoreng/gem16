# 12B Ordinary Decode — 2026-11-17

The completed MTP benefits were secured first in commit
`4beb06e48621f7dd5e02f1e3f61106d25f54ce72`; no push. Ordinary measurements below
use that exact runtime source, before the subsequent owner request to implement
kernel improvements and graph overlap. Historical measurements remain intact.

## Fresh primary results

Same existing Wikipedia fixture: 16,384 identical input IDs, 1,135 output IDs,
1,134 timed decode intervals, batch1, greedy, suppress1/50/106/258882/258883,
MTP0, context requested17,519, resident GPU weights, no prefix reuse/offload.
One benchmark warmup plus three retained requests per engine.

| Engine | Retained tok/s | Median tok/s | Sampled peak VRAM MiB |
|---|---|---:|---:|
| gem16 | 50.6648,50.6718,50.6698 | 50.6698 | 10921 |
| llama.cpp | 57.1246,57.0868,57.4290 | 57.1246 | 9797 |
| vLLM | 56.8920,56.8973,56.8474 | 56.8920 | 14663 |

llama.cpp is12.74% faster than native; vLLM12.28%. This is Ordinary, not the
prior speculative117tok/s path. Each engine repeats its own output exactly;
the three engines' complete output hashes differ. No quality equivalence claim.
llama uses mixed Q8-attention/NVFP4 GGUF and Q8_0 KV; native/vLLM use the pinned
NVFP4/FP8 checkpoint and FP8 KV, with different numerical implementation.
llama internally pads context to17,664. vLLM reserves GPU memory at0.90
utilization; the NVML200ms peaks include startup and cache reservation.
The vLLM Gumbel kernel JIT warning occurs during the one benchmark warmup;
there is no corresponding warning during retained requests.

Central llama was initially stale and was updated to verified official master
`d92ea9bb126ab957849dd0c0f1f881a64de7dc86`, built CPU+SM120a CUDA with the
central configuration, no source patch. Full build commands and binary hashes:
`benchmarks/results/2026-11-17/ordinary-kernel-comparison/central-build-manifest.json`.
vLLM0.33.0/torch2.14.0/FlashInfer0.7.2.post1; all3,231 installed vLLM Python
source hashes still match the retained manifest. Native binary SHA256:
`b979998e11d4ee9016ea35a693de5dac991863bcc3072c2c747e4c54524e2e53`.
Toolchain: RTX5080Laptop SM120, driver620.70.06, NVCC13.4.92, GCC16.2.1,
Nsight2026.5.2; vLLM torch CUDA13.0. Raw environments and commands retained.

## Actual warmed dispatch

Separate diagnostic64-output requests reproduce each primary output prefix.
Nsight native/llama capture node-level CUDA graphs and CUDA API activity;
vLLM worker torch profiler captures16 Ordinary iterations. Its actual annotations
show query1, context0, generation1, KV lengths16,415..16,430, not the initially
expected16,417. Native decode indices30..45 and corresponding llama requests
are selected to match the observed window. llama startup warmup is excluded:
the final64 head calls are prefill output+63 decode outputs, decode0 is eager,
then62 repeated six-part graph replays. The analyzer validates this mapping.

| Diagnostic window, per output step | gem16 | llama.cpp | vLLM |
|---|---:|---:|---:|
| GPU kernels | 868 | 1352 | 793.06 |
| CUDA graph launches | 1 | 6 | 1 |
| Sum of kernel durations, ms | 19.508 | 19.870 | 17.325 |
| Union of kernel intervals, ms | 19.508 | 16.983 | 17.241 |
| Window span/16, ms | 19.708 | 17.429 | 17.447 |
| Non-kernel intervals/16, ms | 0.200 | 0.446 | 0.207 |

The window starts at its first GPU activity and ends after its last:16 complete
steps with15 inter-step gaps. Profiled values are diagnostics, not primary
throughput. API synchronization overlaps device work; do not add it as host
cost. vLLM's final profiler drain is diagnostic. Non-kernel time includes copies,
scheduling and dispatch; no hardware stall attribution was collected.

llama graph-node intervals overlap; their sum exceeds their union by2.886ms/step.
Observed overlapping nodes include small embedding/broadcast/norm/quantization
operations. Source also implements concurrent-region fork/join events and
six executable graph chunks (default512 GGML nodes per chunk). This supports
investigating graph dependency/branch overlap. It does not prove a particular
new multistream implementation will save2.886ms on gem16: kernels, formats,
node dependencies and work grouping differ.

## Cost differences and next implementation targets

| Summed family cost, ms/step | gem16 | vLLM |
|---|---:|---:|
| Norm/quant/rotary/round/embedding | 2.591 | 1.014 |
| FP8 QKV/O projections | 4.998 | 3.999 |
| Attention split+merge | 2.661 | 2.181 |
| NVFP4 Gate/Up/Down | 6.701 | 7.171 |
| GELU+Down quantization | 0.048 | 0.087 |
| Output head (+native argmax) | 2.433 | 2.735 GEMV alone |

Family boundaries differ and can overlap, especially in llama; these are
attribution budgets, not additive speedup predictions. vLLM has additional
sampling/metadata0.054ms. gem16's separate Gate+Up are4.133ms, Down2.568ms;
vLLM's fused GateUp4.681ms, Down2.489ms. Therefore simply copying fused GateUp
is not the first measured opportunity, despite reducing launch count.

1. **12B T1 norm/quant boundaries.** Native post-attention fusion0.805ms,
   post-MLP RMS/residual0.760ms, input RMS+FP8 quant0.520ms, O activation
   quant0.228ms. Current one-CTA256-thread shared-memory reduction trees use
   many barriers and reload/recompute values. Try fixed-width register reuse,
   tree-preserving warp shuffles and coalesced subgroup FP4 packing. Preserve
   every BF16 boundary, FP8/FP4 scale rule and floating-point reduction order.
2. **Independent graph branches.** Gate and Up consume the same immutable
   packed activation/scales and produce disjoint buffers; Q/K rotary and V
   normalization also have independent parts. Capture explicit fork/join edges
   with persistent streams/events allocated before graph capture. Join before
   GELU or KV append. Measure separately and combined; shared bandwidth can
   erase any benefit. Keep diagnostic capture/state paths and MTP behavior intact.
3. **FP8 O, then QKV.** Native O2.133ms versus vLLM1.460ms. Native uses
   direct source-row MMA with block128, grid120 for O; vLLM dispatches SM120
   TMA CUTLASS with block384, grid1x60,96,256 shared bytes. Native QKV uses
   grid128x1x3 (local) or256x1x2 (global); vLLM uses merged packed GEMM.
   Layout/load/tiling work needs its own numerical boundary qualification.
4. **Global attention.** Native global split kernel1.396ms/8 layers versus
   vLLM0.606ms/8 layers, before merges. Native local split1.156ms/40 layers
   is already below vLLM local1.467ms. Native global grid35,block256,
   41,056 shared bytes versus vLLM grid2x1x16,block128,41,728 shared bytes.
   The latter is query/KV/split geometry, not batch2. Preserve local/global
   windows, FP8 scales, softmax and cache semantics before changing scheduling.

Actual full demangled names, grid/block, registers/shared memory, counts,
per-family sums, APIs and correlation IDs are in `kernel-accounting.json`.
Native exact-dispatched projection SASS is retained in `ordinary-projection.sass`;
FP8/NVFP4 MMA instructions are present. This analysis does not claim unmeasured
SM stalls, cache hit rates or achieved bandwidth.

## Evidence and reproduction

Machine-readable result: `artifacts/gemma4_12b/ordinary-kernel-comparison-2026-11-17.json`.
All raw samples, exact commands, logs, NVML CSVs, central manifest, source recheck,
Nsight reports, losslessly compressed SQLite exports and torch trace live under
`benchmarks/results/2026-11-17/ordinary-kernel-comparison/`.
`retained-traces.json` records stored and original SHA256/size; compressed exports
were decoded and hashed before acceptance. Analyzer:
`tools/analyze_12b_ordinary_kernels.py`. Original invocation is in the accounting
log; regenerate by unpacking native/llama SQLite into a temporary directory,
placing the retained torch trace in its `vllm-traces/` child, then running the
analyzer with that directory and a new output path. Do not overwrite primaries.

No production runtime change was made to obtain these baseline measurements.
The owner's subsequent instruction explicitly authorizes implementing both
kernel optimization and overlap; that work follows as separate new evidence.
