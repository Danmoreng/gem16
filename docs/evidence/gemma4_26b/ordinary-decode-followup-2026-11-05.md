# Trellis35 Ordinary Decode follow-up

On `d1b6a3c66e1f299fa02e74c25bc30cf0300c4db2`, the gap between two Ordinary
CUDA Graphs is about 10–13 µs per token. This falls below the 100 µs screen
for a GPU-side Ordinary loop. The follow-up moved to the real-input
Routed-GateUp diagnosis. The production source and executable stayed at the
N1 baseline.

## Inter-graph gap

The workload uses 16,384 prompt IDs, context capacity 17,519, FP8 KV, batch
one, greedy generation with seed zero, the same suppressions and 1,135
outputs. The first output remains in prefill. The ordinary CLI and public
product-session callback both returned the reference token sequence with
zero fallbacks and no token-loop allocations. The callback flushed one
raw token at a time to a local sink. This measures the session callback
path and excludes HTTP and SSE.

| Mode | Untraced median tok/s | CUPTI mean gap, cohort medians | Sparse CUDA event mean |
|---|---:|---:|---:|
| CLI | 139.95 | 10.82 / 10.80 µs | 11.28 µs |
| Product token callback | 139.84 | 13.44 / 13.25 µs | 10.37 µs |

CUPTI collected graph-level activity alone in off/on/on/off order with
three runs per cohort and matching cooldown. Each measured request had
1,134 graphs and 1,133 gaps. The median of the six per-run 95th percentiles
was 14.34 µs for the CLI and 15.06 µs for the callback. CUPTI raised total
decode time by 0.97–1.03% across the ordered comparisons.

The independent event check instrumented only 36 launches in each request
using a clone of the original 656-node graph with a start and end event.
It sampled 18 boundaries in each of two requests per mode. Event retargeting
on the host cost a median 0.14–0.15 µs per sampled next launch. The highest
sampled gap was 31.33 µs in the CLI and 19.01 µs with the callback. The
original CUDA API failure when the diagnostic clone was destroyed before
event retargeting is retained separately. The corrected observer keeps that
clone alive through the measured launches.

Fully removing the measured mean gap would amount to about 0.15–0.19%
throughput by arithmetic at the untraced time per token. The gap includes
any work between the graph endpoints. The last historical AP1 llama.cpp
comparison required about 221 µs per token for parity. The current
measurement compares native instrumentation modes on one revision.

## Routed-GateUp on real expert routes

The fresh production replay uses the same token history and reproduces all
1,135 predictions. It captures four layers at each of positions 16,384,
16,951 and 17,517. All 12 isolated GateUp projections match the captured
output bitwise. Each has eight distinct selected experts, 2,816 input
channels and 1,408 output channels. The production kernel launches an
11-by-8 grid with 256 threads per block and borrows the unchanged resident
weight binding.

| Per-case CUDA event medians | Median across cases | Range |
|---|---:|---:|
| Warmed with one preceding identical projection | 34.82 µs | 28.67–36.86 µs |
| After a 96 MiB cache sweep | 53.25 µs | 51.20–53.26 µs |
| Paired evicted minus warm delta | +17.39 µs | +14.34–24.58 µs |

There are 32 retained pairs per case in alternating order. The timer
excludes cache preparation. These are isolated conditions for the same
kernel and route. Inference can fall between those conditions and includes
other kernels and concurrent work.

| Counter replay | K3/K4 experts | DRAM reads warm / evicted | Issued warp instructions | FP8 MMA instructions |
|---|---:|---:|---:|---:|
| Middle position, layer 0 | 0 / 8 | 0.030 / 15.947 MB | 7,759,664 | 123,904 |
| Early position, layer 15 | 7 / 1 | 0.031 / 12.476 MB | 9,708,688 | 123,904 |

The memory numbers come from separate single-pass application replays
with the cache state prepared immediately before each measured kernel.
The first multipass kernel-replay traffic samples are retained separately
and excluded from the warm/cold conclusion. L2 read requests were 504,272
sectors for the first route and 395,852 for the second, stable across their
two cache conditions. The unique packed payloads were 15.860 and 12.390 MB.

Source counters and SASS identify the codec's warp shuffle, integer dot
product, half arithmetic and FP8 conversion instructions. Those four
instruction forms account for 2,354,176 issued warp instructions in each
of the two profiles. K3-heavy routing adds about 25% total instructions
with the same tensor work. Selection, shifts, register moves and address
arithmetic differ between the rate branches. The average tensor pipe was
active for 21.9–25.6% of elapsed SM cycles. The largest sampled not-issued
category in both routes was long scoreboard, followed by wait. These
counters identify work and dependencies in the fused kernel. Its
overlapping memory, codec and matrix work has no additive time split
from this measurement.

The next focused trial would simplify the K3 packed-window path with
bit-exact outputs and unchanged layout. Compare real mixed-rate and K4-only
routes in both cache conditions, then check full Ordinary throughput if
the operator result transfers. Around 2.4 µs per layer across 30 layers
would be needed for a 1% throughput gain if the full saving transfers.

The Trellis35 CUDA CTest passed. Compute Sanitizer reported zero errors for
the new diagnostic eviction sweep, including its end guard. The
[machine-readable record](../../../artifacts/trellis35/ordinary-decode-followup-2026-11-05.json)
contains all samples, hardware and executable identities, actual routes,
counter selections, sanitized commands and hashes for the private captures.
