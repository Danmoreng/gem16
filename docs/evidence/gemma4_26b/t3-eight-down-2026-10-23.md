# T3 Down paired-fragment qualification — 2026-10-23

**Accepted: reproducible +3.81–3.86% D2 gain; ordinary unchanged.**

The owner requested continuing D2 from accepted `e815e3d`. The retained ChatGPT
Pro response and work order are the input; no new Pro analysis was requested or
received. The active-decision exception is extended only to current D2 cost
accounting and one independently qualified paired-fragment T3 candidate.

## Refreshed cost budget

Accepted source rebuild, five focused tests and full-capacity ordinary/D2 8/64
prefix checks pass. The 1,135-output node trace reproduces all native IDs and
contains 502 groups, 632/1,004 accepted drafts and no ordinary tail. Its diagnostic
wall is 6,692.723 ms, 13.3321 ms/group; GPU interval union is 98.9219%. Every
kernel is classified, and routed work finishes last in all 15,060 MoE fork/joins.

Observed winning-path GateUp/Down costs are 3.2485/1.6189 ms/group. Exclusive
assistant categories, attention, dense projections, target head, control, overlap
and idle time are retained separately. Kernel sums are never added across
concurrent categories to invent wall time. These are diagnostic captures rather
than primary tok/s or a causal counterfactual speedup model.

## Isolated candidate and numerical boundary

Eight warps compute two N8 fragments each for the same N128 output tile. Their
payload and activation loads are shared; every FP32 accumulator retains its K
iteration order, native E4M3 rounding, per-assignment scaling, inverse H128 and
aligned vector stores. Per-expert sharing across one to three T3 rows remains.

Real inputs from layers 0/5/15/29 and three full-context positions provide 24
GateUp/Down projection cases. All outputs are bit exact directly and under graph
replay. Down improves isolated operator throughput by 16.38–40.45%, median
32.24%; GateUp includes -1.37% and is excluded. These are warm-resident graph
samples (16 operations/sample, three warmups, ten alternating samples), not
whole-model bandwidth or throughput claims.

Only native/vector T3 Down (768 -> 2816) selects the new production kernel.
GateUp, M1, nonnative/nonvector diagnostics and other model profiles retain their
accepted dispatches. No precision, model, cache, graph/ring protocol, sampling,
context, timing or weight representation changes. No runtime repack or allocation.

## Qualification already complete

- Expanded zero-tolerance matrix: K3/K4/mixed and unions 8/16/24 over the existing
  output modes, comparing projection and final reduction bits; 220 zero-mismatch
  comparisons in the targeted sanitizer run.
- Seven CUDA/host CTests pass in 34.81 seconds, including full CUDA, MoE,
  attention, Trellis, NVFP4 consumption and conditional graphs.
- Native 8/64 ordinary/D2 identity, 64-token 12B/NVFP4 ordinary/D2 identity and
  explicit-fixture 12B product checks pass.
- Targeted Trellis memcheck and full-model D2 memcheck (16K prompt, context
  17,519, eight outputs, 239.73 s) report zero errors.
- Actual eight-output D2 trace: 60 new Down calls, 256 threads, grid 22x24;
  60 original GateUp calls, 512 threads, grid 11x24. Ordinary retains its prior
  eight-warp M1 Down dispatch. Native E4M3 and QMMA instructions are verified
  in the actual production object; new kernel uses 56 registers, 1,536 bytes
  shared, zero stack and zero local storage.
- Seven offline interval-union/fork-join accounting tests pass.

## Primary result and final decision

Frozen parent/candidate, three warmups plus ten retained repetitions per cohort:

| Cohort in execution order | Decode median tok/s | Decode mean 95% CI | Sampled peak VRAM MiB |
|---|---:|---:|---:|
| D2 parent A | 171.419 | 171.368–171.513 | 12725 |
| D2 candidate B | 177.953 | 177.867–178.187 | 12725 |
| D2 candidate B2 | 178.445 | 178.307–178.642 | 12725 |
| D2 parent A2 | 171.809 | 171.722–172.022 | 12725 |
| Ordinary parent | 137.641 | 137.594–137.754 | 12455 |
| Ordinary candidate | 137.710 | 137.637–137.885 | 12455 |

D2 gains are **+3.8116% / +3.8625%**. Seeded independent bootstrap intervals
for the median ratios are +3.724–3.952% / +3.681–3.958%; these describe the
retained cohorts, not all workload/power-state uncertainty. Ordinary differs by
+0.0500%, with overlapping intervals; this is an adjacent non-regression pair,
not an ordinary speed claim. D2 paired prefill differences are -0.0628% / +0.0162%.

All sixty retained runs reproduce all 1,135 native output IDs, with zero fallback
and no token-loop allocations. All forty D2 retained runs have identical precision,
sampling, KV layout and weight/KV/workspace arena byte counts. All use the original
16,384 prompt IDs, 1,135 outputs / 1,134 intervals, context 17,519, greedy batch
one, FP8 KV and mask `[1,50,106,258882,258883]`. No profiler/sanitizer runs are
included in primary samples.

Firmware max-power and active Dynamic Boost remain; every cohort starts at <=50 C.
High-utilization clock medians in D2 execution order are 2595/2587/2595/2610 MHz;
power medians 171.32/171.285/171.27/171.25 W. Ordinary clocks are 2610 MHz and
power 171.62 W for both. These 200-ms device-wide samples include prefill/warmup,
not isolated decode clocks. They do not support a higher-clock explanation for
the candidate gain. Sampled VRAM is not an instantaneous allocator peak.

## Rechecked full timeline and acceptance

The candidate full trace executes all 1,135 outputs and reproduces the reference.
Its diagnostic wall is 6,437.894 ms versus 6,692.723 ms parent; 502 groups,
zero ordinary tail, no unclassified kernels. GPU interval union is 98.8688%.
The routed branch still finishes last in all 15,060 source-defined MoE joins.
Observed winning-path Down time falls **1.61893 -> 1.09092 ms/group (-32.61%)**;
GateUp stays 3.24845 -> 3.24263 ms/group. These observed trace costs support the
location of the improvement; profiling wall is not a primary speed measurement.

A separate post-chain observer reproduces every concrete draft pair and counts
from the pinned native fixture: acceptance histogram `[138,96,268]`, 632 accepted
of 1,004 proposed, 502 groups and unchanged router/activation semantics. The
rebuilt forced-proposal driver also passes 0/1/2 acceptance, commit/rollback,
position and exact continuation with its explicit diagnostic gate.

The production decision is **keep native/vector T3 Down only**. GateUp remains
unchanged. This does not yet overtake the pinned llama D2 reference (~196.55 tok/s),
and no new cross-format quality-equivalence claim is made. Graph/ring safety fixes
remain unchanged. No push.

## Evidence and limitations

Raw root: `benchmarks/results/2026-10-23/e815e3d/d2-followup/`. Candidate sources,
parent/candidate binaries, commands, exact samples and telemetry are retained.
The SHA-256 record is `artifacts/trellis35/t3-eight-down-2026-10-23.json`; it
contains full raw samples, canonical output hashes, current model/environment
metadata, source code, commands, diagnostics and a raw-file hash manifest. The
screen translation unit is reconstructed by reversing only its integration
include/symbol renaming; this provenance is explicit, not claimed as a pre-build
source snapshot. Its pre-integration candidate header and actual module are retained.

The initial forced-input setup omitted its required diagnostic environment gate;
it failed visibly before proposals and is excluded. The corrected run explicitly
uses `GEM16_VISION_D2_DIAGNOSTIC=1`; primary runs do not. The lab links an unused
attention derivative solely to preserve its existing ABI, without reopening the
rejected attention experiment. An initial SASS-name parser failure is retained;
corrected symbol extraction passes. Today uses the previously fully verified
immutable payloads with startup lock/layout checks, not a fresh full-payload hash.
Linux text evidence only; no Windows/media/everyday-context release claim. No
fresh llama campaign, old llama execution or cross-format quality equivalence.
