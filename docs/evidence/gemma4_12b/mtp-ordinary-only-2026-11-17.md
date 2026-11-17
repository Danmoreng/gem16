# 12B MTP with optimized Ordinary only — 2026-11-17

Only the Ordinary single-row path changes. The four startup MTP exclusions on
Ordinary norms, Gate/Up overlap, FP8 O and global attention are removed. All
other runtime sources are byte-identical to the previously measured source;
Assistant, verifier/prefill and norm CUDA files also match the committed parent.
The unwanted verifier/Assistant port experiment was fully removed and remains
rejected diagnostic history under `benchmarks/results/2026-11-17/mtp-integrated/`.

Run, chat, server and host units rebuilt successfully. Primary measurements
use the final immutable runner, one warmup plus three retained requests per
depth, no profiler. Same 16,384 input IDs, 1,135 output IDs, 1,134 timed decode
intervals, context 17,519, batch one, greedy, suppression set, pinned models,
resident weights, FP8 KV and full fixed-D conditional graphs as the parent.

| Mode | Parent tok/s | Final tok/s | Delta | Samples | Ordinary tail steps |
|---|---:|---:|---:|---|---:|
| D2 | 98.30438 | 98.67648 | +0.379% | 98.67648, 98.73834, 98.27201 | 1 |
| D3 | 100.14748 | 100.17579 | +0.028% | 100.27855, 100.17579, 99.97179 | 3 |
| D4 | 97.02840 | 97.05996 | +0.033% | 97.16474, 97.05996, 96.88664 | 3 |

All output IDs and all proposed draft IDs match the parent exactly in every
retained request. Acceptance/group counters are unchanged; runtime fallback
and token-loop allocation counters remain zero. Host unit tests pass. Existing
unchanged Ordinary operator evidence covers 132 FP8-O and 60 global-attention
comparisons plus scoped memcheck/synccheck/racecheck; no new numerical kernels
are introduced by this startup-gating correction.

The bounded D3 graph trace confirms actual new Ordinary O/global/norm kernels
alongside the old packed verifier projection and BF16 Assistant exact-tree
kernel. Rejected port kernel counts are zero. SQL exports use checked tmpfs
and are automatically removed. The retained Nsight report is a bounded
diagnostic and is excluded from primary throughput.

The Ordinary path executes only at the request tail when too few output slots
remain for D+1 verification rows. For example, D3 has 437 verification groups
and only three Ordinary steps; these kernels therefore cannot materially
accelerate its entire decode. The small measured deltas do not establish a
performance gain: these are separate cohorts with dynamic laptop clocks,
not AB/BA confidence measurements. Standalone Ordinary remains the previously
qualified 53.17228 tok/s; it was not remeasured because its source/dispatch
and MTP=0 startup behavior are unchanged.

| Mode | Parent sampled peak MiB | Final sampled peak MiB |
|---|---:|---:|
| D2 | 11763 | 11769 |
| D3 | 11767 | 11773 |
| D4 | 11771 | 11781 |

Peaks are whole-process 200ms NVML samples, not exact decode-only maxima.
Runner SHA256: `4ea22a29324d7e45fda0dfe21df636702f85205bd41aa691b5c5a6ea0d214f2d`. Base revision `c9b95271b8c9d628deb207aee97721f4267d362c`
plus the source manifest (dirty working tree). No commit or push.

Raw commands, JSON samples, logs, telemetry, dispatch summary, source hashes
and final binary: `benchmarks/results/2026-11-17/mtp-ordinary-only/`.
Portable summary: `artifacts/gemma4_12b/mtp-ordinary-only-2026-11-17.json`.

Portable lossless primary samples, commands and diagnostic summaries:
[`artifacts/gemma4_12b/ordinary-o-global-samples-2026-11-17/`](https://github.com/Danmoreng/gem16/tree/9f7f7a6555ac169a9e49580f11824eef0e55fd85/artifacts/gemma4_12b/ordinary-o-global-samples-2026-11-17).
