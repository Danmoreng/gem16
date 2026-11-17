# 12B Ordinary FP8 O and global attention —2026-11-17

The requested Ordinary checkpoint was committed first:941e69e, with lossless
raw-log supplementc9b9527. Storage cleanup and RAM scratch were then prioritized
at the owner's request. After completion, this bounded12B operator work resumes
that explicitly authorized task. No26B tuning, push or second optimization
commit is implied. The subsequent runtime changes remain uncommitted.

## Fresh unprofiled result

| Native12B Ordinary | Samples, tok/s | Median | Sampled peak VRAM MiB |
|---|---|---:|---:|
| Secured norm+overlap parent | 51.6942,51.7013,51.7001 | 51.7001 | 10927 |
| New FP8-O bundle4 + globalGQA4 | 53.1561,53.1760,53.1723 | **53.1723** | 10927 |

The measured gain is **2.8476%** under the unchanged Dynamic Boost policy.
Each cohort uses exactly **one warmup and three retained requests**. GPU jobs
are serialized, each cohort starts at50C or below, and primary throughput has
no profiler. Workload remains16,384 identical input IDs,1,135 outputs,1,134
timed decode intervals, context17,519, batch1, greedy, suppression1/50/106/
258882/258883, no prefix reuse, MTP0, resident pinned NVFP4/FP8 weights and FP8
KV. All1,135 output IDs match the parent in every retained request; loop
allocation/fallback counters remain zero. Model, precision, template, cache,
sampling, token length and timing boundaries are unchanged.

Whole-process high-utilization NVML200ms samples show median SM clocks2392MHz
for parent and2325MHz for candidate, and median power139.1/136.9W. These are
not exact decode-window clock measurements. Raw telemetry is retained; no
frequency normalization, locked-clock superiority, AB/BA confidence or particular
hardware-stall attribution is claimed. NVML peak is a sampled whole-process
peak, including startup. The earlier today's llama.cpp57.1246/vLLM56.8920
measurements were not repeated in this native-only task; formats and outputs
across engines still differ and imply no quality-equivalence result.

## What changed

**FP8 O projection:** the selected fixed12B T1 variant loads four consecutive
K32 fragments into registers before executing their existing ordered FP8 MMA
chain. Source-order weights, output rows, FP32 accumulator order, activation
scale and per-output BF16 weight scale are unchanged. This is temporary register
load scheduling, with no persistent weight repack or second representation.
Actual native dispatch is`TwelveBOrdinaryOPipelineKernel<4>`, grid120x1x1,
block128,46registers, zero observed local-memory bytes. Actual dispatched SASS
contains`QMMA.16832.F32.E4M3.E4M3`; static counts include alternate paths and
are not dynamic instruction counts. Both tested widths4096/8192 use3840 rows.

The first finer-output-row trial was exact and sanitizer-clean but did not
improve both roles. It stays diagnostic. The ordered-load4/8 screen is also
exact; depth4 wins the global-width role and ties depth8 locally, so only4 is
selected. Cold graph-event operator medians, one warmup plus three samples:
local O61.440→45.056us, global O94.208→65.536us. Those microbenchmarks include
no primary speed claim and are not additive predictions of whole-model gain.

**Global attention:** split the16 query heads into four independent groups of
four. The kernel still stages physical FP8 K/V tiles, performs each head's dot
products in the same dimension order, uses the same256-thread max/sum softmax
tree, and accumulates V in the same token order. Only two warps perform each
four-head group's dot/context math; all256 threads participate in staging and
normalization barriers. Partial outputs and LSE keep the existing16-head layout,
and the existing merge kernel remains unchanged. Actual dispatch is
`TwelveBOrdinaryGlobalGqa4Kernel<false,false,true>`, grid35x4x1, block256,
64registers,16,432 dispatched shared bytes, zero observed local-memory bytes.
The split still uses512 tokens; this is head grouping, not a batch-size change.

The first operator screen improves163.840→153.600us at position16,384 and
163.808→153.600us at17,515. Whole-model profiling below shows a smaller gain.
This specialization is enabled only for12B Ordinary FP8-KV capacities
16,384..17,519, including short active prefixes within such a capacity. Outside
that qualified range the previous global kernel remains selected. FP8-O is
selected only for12B Ordinary FP8 KV, the two fixed O shapes and non-state-
capture suffixes. MTP, BF16 diagnostics and existing protected paths retain
parent dispatch. No per-token environment lookup, allocation, compilation,
filesystem access or repack was added.

Startup diagnostic ablations:`GEM16_12B_ORDINARY_O=0|1` and
`GEM16_12B_ORDINARY_GLOBAL=0|1`; both default on in their qualified scopes.
The previous norm and Gate/Up-overlap defaults stay on. Run, chat and server
are linked against the new core.

## Actual full-graph diagnostic evidence

Two separate bounded64-output Nsight requests reproduce their unprofiled
prefixes. Analyze the same16 complete decode steps, KV lengths16,415..16,430.
Each step still has820 kernels, one full`cudaGraphLaunch` and one
`cudaStreamSynchronize`, with no event/allocation API calls inside the selected
window. The new O kernel actually dispatches48 times and new global split8 times
per step. All other pre-existing operator source lines remain identical apart
from blank lines around additions; other model-specific bodies are preserved.

| Diagnostic summed family time, ms/step | Parent | Candidate |
|---|---:|---:|
| FP8 O | 2.1363 | 1.3253 |
| FP8 QKV | 2.8672 | 2.8692 |
| Global attention split | 1.4212 | 1.3788 |
| Global attention merge | 0.0320 | 0.0332 |
| Local attention split/merge | 1.2460 | 1.2709 |
| All-kernel interval union | 19.1281 | 18.6050 |

Profiled times are diagnostic, not primary throughput. Family sums overlap with
Gate/Up parallelism and must not simply be added; other families and clocks
also differ. The O improvement dominates the whole-model gain. Actual full
names, grids, register/shared counts, selected correlations, APIs, SASS and
resources are retained. Nsight static+dynamic shared figures are used above;
cuobjdump reports an additional1024 shared bytes for each new function on this
RDC build. Both tool outputs are retained without deriving unmeasured occupancy
or stall conclusions from that difference. Stack/local resource counts are zero.

## Numerical, memory and regression qualification

- FP8 O:132 exact checks over12 retained real Ordinary inputs across local/global
  widths and three positions, plus finite zero/maximum/mixed FP8 boundaries.
  Parent reproduces captured outputs before each candidate comparison.
- Global attention:60 checks over24 retained real12B global cases, replaying row0
  as T1 from earlier Target captures. Parent must reproduce actual partial/LSE
  bytes; candidate partial/LSE, merged context and destructive merge-LSE agree
  bit-for-bit. Synthetic first/last-token and split/capacity boundaries include
  capacity16,384/16,385/17,408/17,519 and positions0..17,518. Finite outputs and
  guarded allocations pass. These are retained-input replays, not new current
  full-hidden-state captures.
- Memcheck, synccheck and racecheck pass the scoped real-operator suites for
  both families. This does not claim a full-model/all-profile sanitizer pass.
- All host unit tests pass. Functional64-output D2,D3,D4 prefixes match their
  own secured parents, with zero loop allocation/fallback; no new MTP primary
  speed measurements were performed during that initial qualification. The
  owner then requested the primary follow-up below.
- Both generated production headers reproduce byte-for-byte from the generator.
  Source snapshot hashes and canonical run binary match the measured candidate.
  Pre-existing unrelated dirty patches remain byte-identical.

## Storage and reproduction

Exact commands, raw primary JSONs/NVML CSVs, candidate hashes/source snapshot,
operator checks, bounded Nsight reports and dispatch/SASS summaries live in
`benchmarks/results/2026-11-17/ordinary-o-global/`. Machine-readable result:
`artifacts/gemma4_12b/ordinary-o-global-2026-11-17.json`.

All large generated CUDA/libraries and SQL exports use checked tmpfs contexts
via`tools/experiment_scratch.py` and are automatically removed after analysis.
Only two deliberately retained bounded raw reports, each under16MiB, remain as
new dispatch provenance. No new SQLite dump is retained on SSD. Source generator:
`tests/cuda/build_12b_ordinary_o_global.py --o-pipeline`; replay driver:
`tests/cuda/twelve_b_ordinary_o_global_replay.py`; read-only profile analyzer:
`tools/analyze_12b_o_global_profiles.py`. Unselected derivatives remain diagnostic.

## Owner follow-up: current D2/D3/D4 throughput

After the owner requested actual MTP speed measurements, run each depth with
one warmup plus three retained1135-output requests on the same candidate
binary. No rebuild or additional kernel change.

| Depth | Previous median tok/s | New median tok/s | New retained samples |
|---|---:|---:|---|
| D2 | 98.7263 | 98.3044 | 98.6593,98.3044,98.2524 |
| D3 | 100.3176 | 100.1475 | 100.1873,100.1475,99.9251 |
| D4 | 97.1797 | 97.0284 | 97.1689,97.0284,96.8701 |

There is no observed MTP speed gain. All1135 outputs, acceptance/proposal/group
counters and GPU-chained execution agree with each depth's own parent. Loop
allocation/fallback counters remain zero. New Ordinary operators are disabled
for active MTP: its3/4/5-row verification and Assistant paths retain the old
operators. The small differences versus the earlier cohorts do not establish
a kernel-caused regression or gain. These measurements demonstrate why the
Ordinary improvement does not transfer automatically. Targeted MTP/Assistant
porting is separate work; it has not been implemented or claimed here.

## Subsequent Ordinary-only MTP activation

The subsequent owner clarification preserves all optimized MTP verification
and Assistant code. Only the Ordinary startup exclusions during MTP are
removed. The final rebuilt D2/D3/D4 measurements and actual joint dispatch
are recorded in [the follow-up evidence](mtp-ordinary-only-2026-11-17.md).
This supersedes the earlier MTP gating described above; prior results remain
historical evidence for their original configuration.

Portable lossless primary samples, commands and diagnostic summaries:
[`artifacts/gemma4_12b/ordinary-o-global-samples-2026-11-17/`](https://github.com/Danmoreng/gem16/tree/9f7f7a6555ac169a9e49580f11824eef0e55fd85/artifacts/gemma4_12b/ordinary-o-global-samples-2026-11-17).
