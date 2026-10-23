# T3 Down paired windows and fixed register pipeline — 2026-10-23

**Accepted: +0.629–0.743% D2 throughput; ordinary adjacent pair -0.010%.**

Owner-authorized bounded continuation after GateUp `629bbac`. Only native/vector
T3 Down is a production candidate. GateUp/M1, target/assistant formats, precision,
K order, scales, H128, vector stores, expert deduplication, per-assignment rows,
cache, sampling and graph/ring safety stay at the accepted parent.

## Screen and countercase

The direct transfer of the GateUp word-window codec is bit exact over all 24 real
projections and 2,097,152 primitive cases. Twelve Down cases span -1.10% to +15.07%
operator throughput, median +2.89%. It fails the conservative all-case screen and
is not promoted as that implementation. Exact pre-build sources and binary remain
in the raw directory, including the screen assertion failure.

Actual-production parent versus diagnostic direct-window shader, warm real inputs,
three warmups, `--cache-control none --clock-control none`:

| Layer0 / initial position | Production parent | Direct-window diagnostic |
|---|---:|---:|
| Registers/thread | 56 | 40 |
| Register occupancy limit, blocks | 4 | 6 |
| Eligible warps/cycle | 2.835 | 3.725 |
| Long-scoreboard / active issue | 0.956 | 1.047 |
| Short-scoreboard / active issue | 0.174 | 0.443 |
| NCU duration us | 33.792 | 34.400 |

The increased theoretical residency does not establish a speedup. Larger
long-scoreboard than short-scoreboard ratios motivate one bounded payload-load
pipeline experiment; they do not prove a causal time budget. A second layer 5
case and all raw metric units are retained. These ratios are not bounded wall-time
shares, and multipass diagnostic durations are excluded from primary measurements.
Any cache percentages above 100% are excluded from physical interpretation.

## Refined candidate

The refinement uses the existing M1 four-iteration register-pipeline pattern for
T3 Down's resident payload words, together with the same fixed-window codec at
the same 32 paired-MMA positions. Only payload load scheduling changes; the
accumulation visits every K iteration in its original order. No new persistent
buffer, allocation, weight representation, quantization or runtime packing.

All 24 captured projections and the primitive matrix remain bit exact. The twelve
Down cases improve 3.50–16.63%, median 5.54%, in isolated warm graph throughput.
This does not predict the whole-model gain. Each diagnostic graph sample contains
16 calls, three warmups and ten alternating samples. The actual production build
has 63 registers, stack/local 0 and 1,536 shared bytes, with native E4M3 and QMMA. Its
resource usage differs from separately compiled lab shaders. No NCU comparison
of the refined production shader is claimed.

## Qualification and primary results

Seven CUDA/host tests pass in 33.96 s, preserving strict K3/K4/mixed, unions 8/16/24,
output-mode and final-reduction tolerances. Targeted Trellis memcheck and full
16K/context 17,519/eight-output D2 memcheck pass with 0 errors (full 238.20 s).
Native ordinary/D2 8/64 prefixes match the pinned fixture. Protected 12B/NVFP4
ordinary/D2 outputs also exactly match the previous accepted parent qualification;
explicit-fixture 12B product and seven offline accounting tests pass.

Frozen parent/candidate, three warmups and ten retained runs in each cohort:

| Cohort, execution order | Decode median tok/s | Decode mean 95% CI | Sampled peak VRAM MiB |
|---|---:|---:|---:|
| D2 parent A | 193.418 | 193.389–193.650 | 12725 |
| D2 candidate B | 194.854 | 194.688–195.017 | 12725 |
| D2 candidate B2 | 194.716 | 194.607–194.912 | 12725 |
| D2 parent A2 | 193.499 | 193.400–193.645 | 12725 |
| Ordinary parent | 137.672 | 137.634–137.898 | 12455 |
| Ordinary candidate | 137.659 | 137.604–137.862 | 12455 |

D2 gains are **+0.7426% / +0.6286%**. Seeded bootstrap median-ratio 95%
intervals are +0.562–0.832% /
+0.539–0.784% (seed 20261023, 10,000 resamples).
These describe retained cohorts, not all between-session/power-state variation.
Ordinary's adjacent pair differs by -0.0097%; no ordinary speed claim or
separate ordinary AB/BA campaign. D2 paired prefill differences are
+0.0556% / -0.0351% and are separately disclosed.

All 60 retained runs match all 1,135 output IDs, no fallback/repack/token-loop
allocation. All 40 D2 retained runs keep 502 groups, 632 accepted/1,004 proposed and
identical precision, sampling, KV layout and weight/KV/workspace byte counts.
Workload: 16,384 prompt IDs, 1,135 outputs/1,134 intervals, context 17,519, batch one, greedy,
FP8 KV, suppression [1,50,106,258882,258883], identical Target/Assistant and cache.
No diagnostic observer, forcing gate or profiler is active in primary runs.

Power remains max-power/active Dynamic Boost with firmware-managed clocks;
cohorts begin at <=50 C. High-utilization clock medians in table order:
2595/2595/2595/2595/2610/2610 MHz; power 171.15/171.13/171.14/171.14/171.38/171.50 W. The 200 ms device-wide telemetry includes
loading/prefill/warmups, not isolated decode clocks or instantaneous allocator peaks.

## Full timeline, dispatch and acceptance

Full 1,135-output candidate node trace: 5,939.644175 ms versus reused
identical-binary parent 5,921.865726 ms, 502 groups, no ordinary tail or
unclassified kernels. GPU interval union 98.7667%.
Actual DownWindow and unchanged GateWindow each dispatch 15,060 calls with 256 threads,
respective grids 22x24 / 11x24; the old Down kernel is absent. All 15,060 MoE blocks
retain a routed terminal branch. The observed winning-path Down cost changes
1.089208 -> 1.014593 ms/group (-6.850% latency);
GateUp 2.253963 -> 2.280668 ms/group. Profiler wall/counters are diagnostic,
not primary throughput or an independent causal speedup model. The diagnostic
trace wall is slightly longer even though the unprofiled AB/BA cohorts improve;
the reused parent trace is not a fresh paired profiler comparison.

The proposal observer reproduces every concrete draft pair from the pinned native
fixture, histogram [138,96,268]. Forced 0/1/2 acceptance, commit/rollback positions
and exact continuation checks pass. Keep only the qualified native/vector Down
window plus fixed-four-iteration pipeline; GateUp/M1 and protected paths remain.

The candidate is about 194.72–194.85 tok/s. The pinned llama.cpp b11552 native-format
D2 reference remains 196.551 tok/s and ordinary 138.312 tok/s; it is reused, not a
fresh/latest comparison. Target/Assistant/KV formats and output sequences differ;
no quality equivalence or general superiority is claimed.

## Evidence and limitations

Raw root: `benchmarks/results/2026-10-23/629bbac/t3-down-window/`.
Frozen parent CLI/replay driver, exact source snapshots for both implementations,
compiled binaries, commands, inputs, samples, telemetry and counters are retained.
The full parent timeline is reused from immediately preceding accepted GateUp
qualification with the identical executable hash; it is not a new parent trace.
The inherited immutable model payloads use per-process lock/layout validation,
not a fresh whole-payload hash. An unused attention derivative supplies only the
diagnostic ABI; no attention/staging change is performed. Linux text evidence
does not extend Windows/media/everyday-context/release qualification. No new
llama baseline or cross-format quality equivalence. No push.

Acceptance record: `artifacts/trellis35/t3-down-window-2026-10-23.json`, retaining
exact sources, scripts, raw samples, telemetry and SHA-256 manifest.
