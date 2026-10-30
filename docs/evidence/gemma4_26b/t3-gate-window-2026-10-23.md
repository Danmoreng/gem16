# T3 GateUp fixed codec windows — 2026-10-23

**Accepted: +7.92–8.05% D2 throughput; ordinary -0.13% in the adjacent pair.**

Owner-authorized continuation from accepted Down-only `6895ed2`. Keep the
existing M1/Down and graph/ring safety fixes; no precision, sampling, context,
cache, model-format or persistent weight-layout changes. No push or release.

## Countercase and decision

The naive paired-eight-warp GateUp has a reproducible late-context regression:
layer 29/union 20 approximately -1%, with another late layer 0 countercase. A fresh
24-projection screen reproduces all outputs. Warm actual-input NCU profiles
compare the bad case and good layer 5/union 12 case, with three explicit warmups,
`--cache-control none --clock-control none`. These diagnostics have distinct
cache/timing boundaries from primary throughput and older cold replay.

| Bad layer 29 case | Original16 warps | Naive8 warps | Actual production window candidate |
|---|---:|---:|---:|
| Registers/thread | 42 | 48 | 47 |
| Eligible warps/cycle | 2.398 | 1.835 | 2.572 |
| Issue activity % | 71.43 | 64.99 | 75.20 |
| Short-scoreboard / active issue | 0.471 | 1.170 | 0.181 |
| NCU duration us | 98.272 | 116.064 | 79.936 |

The lower eligible-warp/issue rate and larger short-scoreboard ratio motivate
reducing codec shuffle dependencies, not generic DRAM prefetch. DRAM activity
is small in this diagnostic. Occupancy alone does not explain performance.
The actual candidate is profiled through the handle captured from the production
executable, not a separately compiled lab shader. Its captured activation,
scales, selected IDs and output bytes match the baseline bad case exactly.

Counter ratios retain their units; they are not bounded wall-time shares.
Some multipass cache-hit ratios exceed100%; those inconsistent percentages are
retained but excluded from physical interpretation. NCU durations are not primary
performance measurements or independent causal proof.

## One candidate, one rejected implementation

A general paired-word-window implementation is bit exact over 16,777,216 lane
cases but materially slower on real inputs, with a 16-byte stack frame. It is
archived and rejected before any production promotion. The refinement exploits
only the 32 MMA positions actually used by this N128 paired-fragment geometry.
It gathers three circular payload words once for both N8 fragments, instead of
repeating separate two-word gathers for each adjacent state pair. Accumulation,
codebook mapping, native half2/E4M3 rounding, per-assignment scales and H128 stay
unchanged. No prefetch, task packing, quantization fusion or KV staging is opened.

Let `q=lane&3`, `g=lane>>2`. The first position is
`p=32*g +16*(q&1)+2*(q>>1)`. The first state window begins at
`p*Rate+Rate-16` modulo the circular payload. Its within-word offset is:

| q | K3 bit offset | K4 bit offset |
|---|---:|---:|
| 0 | 19 | 20 |
| 1 | 3 | 20 |
| 2 | 25 | 28 |
| 3 | 9 | 28 |

The four adjacent-state pairs start at p+0/4/8/12. Each spans 16+Rate bits;
the tables yield the fixed funnel/right shifts in the private codec helper.
All 32 supported positions and both rates are checked against the original
decoder over 1024 payload seeds and 32 lanes (2,097,152 lane cases), including
zero and all-one words. The test is now part of the normal Trellis CUDA suite.
The implementation is private to this MMA geometry; it is not a general decoder
for arbitrary position values.

Twelve real GateUp cases (layers0/5/15/29, three context positions) are bit exact
and 11.29–48.26% faster in isolated graph throughput, median 24.93%. Both former
countercases improve. GateUp alone selects the new production kernel for the
native/vector 2816->1408 path. Accepted Down/M1 and other variants retain their
specializations. Real operator samples use 16 operations/graph sample, three
warmups and ten alternating samples; they do not replace whole-model evidence.

## Qualification

- Seven CUDA/host tests pass in 37.06 s; existing K3/K4/mixed, unions 8/16/24,
  output-mode and final-reduction comparisons remain zero-tolerance.
- Targeted Trellis memcheck and full 16K-prompt, context 17,519, eight-output D2
  memcheck pass with zero errors; full-model check 249.30 s.
- Native ordinary/D2 8/64 prefixes, protected 12B/NVFP4 ordinary/D2 64-token
  parity and explicit-fixture 12B product checks pass.
- Seven offline interval-union/fork-join accounting tests pass.
- Actual node dispatch: 60 GateUp calls with 256 threads and grid 11x24 in the
  eight-output D2 trace; accepted Down/M1 dispatch unchanged.
- Production shader: 47 registers, stack 0/local 0, shared 1536 bytes, native
  E4M3 and QMMA instructions. Static `SHFL.IDX` count across both rate branches
  is 16 old / 12 new, with eight instead of sixteen warps for the same N128 tile.

## Primary measurement and final decision

Frozen parent/candidate, three warmups plus ten retained repetitions each:

| Cohort in execution order | Decode median tok/s | Decode mean 95% CI | Sampled peak VRAM MiB |
|---|---:|---:|---:|
| D2 parent A | 179.103 | 179.040–179.282 | 12725 |
| D2 candidate B | 193.284 | 193.162–193.594 | 12725 |
| D2 candidate B2 | 193.354 | 193.242–193.583 | 12725 |
| D2 parent A2 | 178.950 | 178.873–179.099 | 12725 |
| Ordinary parent | 137.593 | 137.475–137.824 | 12455 |
| Ordinary candidate | 137.411 | 137.307–137.637 | 12455 |

D2 gains are **+7.9180% / +8.0492%**. Seeded bootstrap median-ratio intervals
are +7.782–8.066% / +7.930–8.182%. These describe retained cohorts and do not
capture all workload/power-state uncertainty. Ordinary differs by -0.1323%,
with overlapping intervals; disclose this small variation rather than claiming
an ordinary speed gain. This is an adjacent pair, not a separate AB/BA ordinary
campaign. D2 paired prefill differences are -0.2272% / +0.0137%.

All sixty retained runs reproduce all 1,135 output IDs. All forty D2 retained
runs keep 502 groups, 632/1,004 accepted drafts, identical precision/sampling/KV
layout and identical weight/KV/workspace byte counts; no fallback, runtime repack,
teacher forcing or token-loop allocation. Original 16,384 prompt IDs,
1,135 outputs/1,134 intervals, context 17,519, batch 1, greedy, FP8 KV and mask
`[1,50,106,258882,258883]`; identical Target/Assistant and cache semantics.
No diagnostic observer, forced-proposal gate or profiler is active in primary runs.

Max-power and active Dynamic Boost are retained. Each cohort starts <=50 C.
D2 high-utilization clock medians in execution order are 2617/2595/2595/2610 MHz;
power 171.21/171.19/171.24/171.28 W. Ordinary clocks 2602 MHz both, power 171.59/171.57 W.
The 200 ms device-wide samples include prefill/warmups and are not isolated decode
clocks or instantaneous allocator peaks. They do not support a higher-clock
explanation for the candidate gain.

## Final timeline and acceptance

The new full node trace reproduces all 1,135 IDs and contains 502 groups, no
ordinary tail or unclassified kernels. Diagnostic wall is 5,921.866 ms versus
6,437.894 ms in the immediately preceding accepted parent trace. GPU interval
union is 98.7813%; all 15,060 MoE terminal branches remain routed.
Observed winning-path **GateUp falls 3.24263 -> 2.25396 ms/group (-30.49%)**;
accepted Down stays 1.09092 -> 1.08921 ms/group. This localizes the measured change;
profiler wall is not a primary speed claim or causal counterfactual model.

The post-chain observer reproduces every concrete draft pair from the pinned
native fixture, with acceptance histogram `[138,96,268]`. Rebuilt forced 0/1/2
acceptance, commit/rollback position and exact continuation checks pass.
The production decision is **keep native/vector GateUp paired windows only**.
The fixed benchmark now reaches about 193.3 tok/s, still below the pinned llama
D2 reference (~196.55 tok/s); model/assistant/KV formats differ, and no quality
equivalence or general/latest-llama superiority is claimed. No push.

## Evidence and limitations

Raw root: `benchmarks/results/2026-10-23/6895ed2/t3-gateup/`. Every prototype has
actual pre-build source snapshots; binaries, commands, raw samples, counter
units and telemetry are retained. Acceptance record: `artifacts/trellis35/t3-gate-window-2026-10-23.json`, with
source blobs, full raw samples/canonical output hashes, exact commands, counter
units, source snapshots, telemetry and SHA-256 manifest. Parent full timeline is the immediately preceding
accepted Down qualification, with identical frozen executable hash; it is not
misrepresented as a new parent trace.

An NCU import initially assumed the uncompressed extension; importing the actual
compressed report fixed it without repeating that GPU profile. An N128-name-only
dispatch query omitted GateWindow; the corrected query confirms actual calls.
Those analysis failures are retained. Short Nsight captures show long outer
profiling overhead (phase not isolated) while engine timings and token outputs remain normal; no
profiler timing enters primary throughput. Previously verified immutable payloads
are reused with per-process lock/layout checks, not a fresh full-payload hash.
Linux text evidence only; no fresh Windows/media/everyday-context release claim,
no new llama baseline and no cross-format quality-equivalence claim.
