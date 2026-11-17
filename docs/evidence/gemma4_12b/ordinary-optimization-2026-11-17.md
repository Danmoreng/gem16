# 12B Ordinary: exact norm boundaries and graph overlap — 2026-11-17

The owner requested both kernel improvements from the vLLM comparison and
independent-work overlap inspired by llama.cpp. Both are implemented for12B
Ordinary T1 with checkpoint FP8 KV. The combined qualified path reaches
**51.6977tok/s, +2.0287%** against the secured parent. This does not close the
remaining gap to llama.cpp57.1246 or vLLM56.8920tok/s.

## Fresh primary evidence

The unchanged parent is commit`4beb06e48621f7dd5e02f1e3f61106d25f54ce72`,
retained executable SHA256`b979998e11d4ee9016ea35a693de5dac991863bcc3072c2c747e4c54524e2e53`.
Its fresh primary is documented in `ordinary-kernel-comparison-2026-11-17.md`.
All variants retain exactly the same16,384 input IDs, context17,519,1,135 outputs,
1,134 decode intervals, batch1, greedy and suppression controls. Each new cohort
uses **one warmup plus three retained measurements**. GPU jobs are serialized,
with cooling to50C or below before each cohort. No primary uses a profiler.

| 12B Ordinary path | Retained tok/s | Median | Change vs parent | Peak sampled VRAM MiB |
|---|---|---:|---:|---:|
| Secured parent | 50.6648,50.6718,50.6698 | 50.6698 | — | 10921 |
| New norm/quant kernels only | 51.5675,51.5879,51.5129 | 51.5675 | +1.7717% | 10921 |
| Gate/Up overlap only | 50.6979,50.7417,50.6196 | 50.6979 | +0.0554% | 10927 |
| Both | 51.6977,51.6982,51.6633 | 51.6977 | +2.0287% | 10927 |

Every retained request reproduces all1,135 parent output IDs exactly, with
zero reported token-loop allocations and no fallback. The combined cohort is
0.2526% above kernel-only, with disjoint observed sample ranges, but these are
three-sample serialized cohorts, not an AB/BA causal confidence result.
**Overlap alone has no reliable speed advantage.** Do not advertise it as an
independent substantial gain. Whole-process NVML200ms peaks include startup.

## Implementation

`src/cuda/norm/twelve_b_ordinary.cu` contains fixed12B operators:

- Input RMSNorm+FP8: cache15 values per thread in registers instead of reading
  and normalizing them repeatedly.
- Attention-output BF16 round+FP8 quantization: preserve the round-before-amax
  rule while combining two launches for the4096/8192-element shapes.
- Post-attention/postnorm/residual/pre-MLP/FP4: cache intermediates; pack each
  parent16-element FP4 group using contiguous16-lane subgroups, with the same
  BF16 boundaries, E4M3 block scale, approximate reciprocals and E2M1 rounding.
- Post-MLP RMSNorm/residual/scalar: fixed3840-element register reuse with the
  same parent arithmetic.

The RMS reduction remains the parent's descending256-lane tree
128,64,32,16,8,4,2,1. Only the final five stages move from shared memory into
warp registers; no reassociation of floating-point sums is introduced. FP4
amax covers the same16 values and retains the zero initialization/NaN-ignore
behavior of the parent max operations. No change to weights, precision, tensor
layout, cache semantics, attention, sampling, output length or timing boundary.
The measured dispatched SASS contains warp shuffles; all five new instantiated
kernels have zero stack and local-memory bytes,35..54 registers and2..3KiB
shared memory in the actual binary resource report.

`OrdinaryDecodeFork` owns one nonblocking auxiliary stream and two timing-disabled
events per12B engine slot. Resources are created during initialization, before
capture. After the packed FFN activation is ready, the graph forks Gate and Up
into disjoint output buffers and joins before GELU+Down quantization. Weights
and the activation/scales are read-only in both branches. Event reuse is at
capture time: recurring full decode is still one`cudaGraphLaunch` and one
`cudaStreamSynchronize` per step, with no per-token event API calls, stream
creation, allocation or weight repacking.

Both changes are enabled by default for the qualified Ordinary FP8 path.
Startup-only diagnostic ablations are
`GEM16_12B_ORDINARY_NORMS=0|1` and`GEM16_12B_ORDINARY_OVERLAP=0|1`.
Invalid values fail. Active MTP and BF16-KV diagnostics retain their previous
operators; diagnostic state capture retains its existing suffix boundaries.
No26B runtime dispatch was modified. Run, chat and server binaries are linked
against the updated core.

## Actual dispatch and limits of overlap

Separate64-output Nsight requests reproduce the same primary prefixes. The
same16 steps, KV lengths16,415..16,430, are used as in the reference comparison.
All868 parent kernels become820 kernels per step:48 attention rounding launches
are removed. Input/post-attention/post-MLP/new attention quantization each
actually dispatch48 times per step. Both variants use one full graph replay.

| Diagnostic cost per step, ms | Parent | Kernel-only | Combined |
|---|---:|---:|---:|
| Norm/quant/rotary/round/embedding sum | 2.591 | 2.119 | 2.118 |
| FFN NVFP4 kernel-duration sum | 6.701 | 6.800 | 9.287 |
| All-kernel duration sum | 19.508 | 19.221 | 21.711 |
| Union of all-kernel intervals | 19.508 | 19.221 | 19.129 |

In combined, **768/768 Gate/Up pairs overlap** across16 steps, totaling
2.582ms of overlap per step. Kernel-only has0/768 overlapping pairs.
The larger combined sum shows why overlap is not free speed: the concurrently
executing projections take longer individually, nearly absorbing the interval
saved. They share GPU resources; these traces do not identify a specific
bandwidth, cache or instruction stall as the cause. No such hardware counters
were collected. The profiler numbers are diagnostic, not primary tok/s.

The larger remaining measured opportunities are still native FP8 O projection
and global attention, plus residual norm/quant cost. A separate precision-
and numerical-boundary qualification is required before importing their
vLLM tiling/layout/scheduling ideas. The output head and aggregate NVFP4 FFN
were already competitive, so this change does not replace them wholesale.

## Validation and retained evidence

- `twelve_b_ordinary_norms`:612 bit-exact checks over random, BF16 rounding,
  zero, tiny/large finite inputs, null weights/scalars, optional norm outputs,
  and residual/output aliasing, with guarded buffers.
- Compute Sanitizer memcheck, synccheck and racecheck: zero errors/hazards on
  that boundary suite. This is scoped operator validation, not a claim of
  full-model/all-profile sanitizer qualification.
- Retained real12B post-attention fixtures:48 combinations across12 actual
  inputs reproduce postnorm, residual hidden, FP4 payload and scales bit-exactly,
  including alias and optional-post-output cases.
- All host unit tests pass. Canonical run binary matches the measured candidate.
- D2,D3,D4 functional64-output prefixes match their own secured parents, with
  zero loop allocation/fallback; these are not new MTP performance controls.
- All unrelated pre-existing dirty patches are byte-for-byte preserved.

Exact commands, primary JSONs, raw NVML CSVs, source snapshot/hashes, measured
candidate binaries, test logs, SASS/resources, complete Nsight reports and
losslessly compressed SQLite exports:
`benchmarks/results/2026-11-17/ordinary-optimization/`.
`retained-traces.json` verifies stored/original SHA256; `source-manifest.json`
records the parent plus the exact changed source. `kernel-accounting.json`
contains actual full names, launch geometry, API counts and pair overlaps.
Analyzer: `tools/analyze_12b_ordinary_optimization.py`.
Machine-readable result:
`artifacts/gemma4_12b/ordinary-optimization-2026-11-17.json`.

The earlier requested checkpoint commit is complete. After this evidence was
reported, the owner explicitly requested a second local checkpoint of these
Ordinary changes before the next FP8-O/global-attention slice. Portable raw
samples, compressed trace exports and real norm fixtures are additionally
retained under [`artifacts/gemma4_12b/ordinary-samples-2026-11-17/`](https://github.com/Danmoreng/gem16/tree/9f7f7a6555ac169a9e49580f11824eef0e55fd85/artifacts/gemma4_12b/ordinary-samples-2026-11-17). No push.
