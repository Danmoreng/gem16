# M1 GateUp paired codec windows — 2026-10-24

**Accepted ordinary gain: +1.868–1.972%; D2 adjacent pair +0.221%.**

Step 1 of the owner-authorized five-step Goal, frozen parent `26c6840`. The
native/vector 2816->1408 M1 GateUp specialization now shares codec word windows
between two N8 fragments across eight warps. The existing four-iteration payload
pipeline, K accumulation, half2/E4M3 conversion, scales, H128 and vector stores
remain. M1 Down and T3 use their previously accepted specializations. There is
no precision, weight-format/layout, sampling, context, cache or graph/ring change.

## Operator and counter evidence

All 24 real projections at four layers and three context positions are bit exact;
all 2,097,152 scoped primitive lane cases pass. Twelve GateUp cases improve
13.57–17.60% in isolated warm graph throughput, median 15.74%, including the
former naive-eight-warp countercases. Each graph sample has 16 calls, three
warmups and ten alternating samples. These diagnostics do not replace primary
whole-model timing.

Warm NCU compares actual production parent and separately compiled candidate for
the former layer5 countercase and layer29 good case. It retains metric units,
three real-input warmups, `--cache-control none --clock-control none`. Duration
falls even though eligible-warps and active-issue ratios decrease. These ratios
alone do not explain performance; neither occupancy nor stall ratios are used
as a causal wall-time budget. Multipass timing is excluded from primary results,
and inconsistent cache percentages above 100% are excluded from interpretation.

A separate NCU capture uses the rebuilt actual production function handle, not
the lab candidate. Its activation, scales, expert IDs and expected output bytes
match the parent capture. Actual production resources: 56 registers, stack/local
0 and 512 shared bytes, native E4M3 conversion and QMMA. Source snapshots preserve
the pre-build diagnostic and production implementations separately.

## Qualification and primary samples

Seven CUDA/host tests pass in 34.06 s, unchanged numerical tolerances. Targeted
Trellis memcheck and full 16K-prompt/context-17,519 ordinary memcheck pass with
zero errors; full check 240.71 s. Native ordinary/D2 8/64 prefixes match the
pinned fixture. Protected 12B/NVFP4 ordinary/D2 and explicit-fixture 12B product
checks pass; protected vectors match the previously qualified parent. Seven
interval-union/critical-branch accounting tests pass.

Three warmups and ten retained runs per frozen-binary cohort:

| Cohort, execution order | Decode median tok/s | Decode mean 95% CI | Sampled peak VRAM MiB |
|---|---:|---:|---:|
| Ordinary parent A | 136.481 | 136.394–136.649 | 12455 |
| Ordinary candidate B | 139.031 | 138.944–139.164 | 12455 |
| Ordinary candidate B2 | 138.992 | 138.916–139.125 | 12455 |
| Ordinary parent A2 | 136.303 | 136.228–136.463 | 12455 |
| D2 parent | 192.600 | 192.512–192.795 | 12725 |
| D2 candidate | 193.025 | 192.933–193.221 | 12725 |

Ordinary gains are +1.8682% / +1.9723%. Seeded bootstrap median-ratio 95%
intervals are +1.714–2.001% /
+1.846–2.093% (seed 20261024, 10,000 resamples).
These describe retained cohorts, not all between-session/power-state uncertainty.
D2 differs by +0.2205% in its adjacent pair, a non-regression result rather than
a D2 speed claim or separate D2 AB/BA campaign. Ordinary paired prefill differences
are -0.1096% / -0.0098% and are reported separately.

All 60 retained runs reproduce the exact 1,135 IDs. All 20 D2 runs retain 502
groups, 632/1,004 accepted proposals and the same arena bytes, precision/sampling
and KV layout. No fallback, runtime repack, teacher forcing or token-loop
allocation. Workload: 16,384 prompt IDs, 1,135 outputs/1,134 decode intervals,
context 17,519, batch one, greedy, FP8 KV, suppression [1,50,106,258882,258883],
identical Target/Assistant and timing/cache boundaries. No observer or profiler
is active in primary runs.

Max-power/active Dynamic Boost remain firmware-managed; cohort starts <=50 C.
Whole-run high-utilization clock medians in table order are 2565/2587/2587/2565/2550/2565 MHz;
power 171.73/171.74/171.74/171.75/171.34/171.25 W. The 200 ms device-wide samples include load/prefill/warmups,
not isolated decode clocks or instantaneous allocator peaks. Fresh same-day
AB/BA controls are used; yesterday's rates are not a gain baseline.

## Actual dispatch, timeline and D2 semantics

Fresh full ordinary parent/candidate node traces reproduce all 1,135 IDs and
have no unclassified kernels. Diagnostic wall: 8361.441 ->
8227.413 ms; candidate GPU interval union
97.1695%. Actual dispatch shows 34,020 new
256-thread GateUp calls with grid 11x8, replacing the parent's 512-thread calls.
Accepted Down stays 256 threads/grid 22x8. Observed terminal-path GateUp cost
1.279789 -> 1.168918 ms/token; Down 0.491731 -> 0.479661. Trace wall/costs
are diagnostic observations, not a replacement for the unprofiled AB/BA result
or an independent causal speedup model.

The legacy `ordinary_tokens` field in the raw cost JSON counts the removed extra
masked Argmax kernel and is therefore zero. Independent GPU counts confirm
1,134 FinalArgmax calls, 1,134 SoftcapArgmaxBlocks calls and 34,020 MoE input-norm
calls in each trace. The corrected count sidecar and initial record-generation
assertion are retained; raw traces and cost files were not rewritten.

Every concrete D2 draft pair matches the pinned native fixture, histogram
[138,96,268]. Forced 0/1/2 acceptance, commit/rollback positions and exact
continuations pass. The accepted production decision is M1 GateUp paired windows
only. A new/current llama.cpp comparison is still required before any general
or current-version superiority claim; formats and output sequences differ.

## Evidence and limits

Raw root: `benchmarks/results/2026-10-24/26c6840/m1-gate-window/`. Acceptance record:
`artifacts/trellis35/m1-gate-window-2026-10-24.json`. Both frozen executables,
pre-build sources, commands, raw samples, telemetry, NCU units, fresh full traces,
proposals and SHA-256 manifest remain available. Previously verified immutable
model payloads are reused with per-process lock/layout validation, not fresh
whole-payload SHA. An unused rejected attention derivative supplies only lab ABI;
no staging or attention change is included. Linux text evidence does not extend
Windows, media, everyday-context or release qualification. No push/release.
