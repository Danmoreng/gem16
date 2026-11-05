# Ordinary local attention and FP8 O experiments

A prior matched-token Ordinary replay identified local attention and O
projection as candidates for further measurement. These experiments
compare native Gemma 4 26B A4B Trellis35 builds on Linux SM120a.
The starting parent is `d1b6a3c66e1f299fa02e74c25bc30cf0300c4db2`.
All timing uses resident weights and firmware-managed clocks on the RTX 5080
Laptop GPU (16,303 MiB, driver 615.78.08, CUDA 13.4.92).

## 1. Local attention

The parent reads each FP8 V element in the weighted accumulation loop. The
candidate issues aligned 16-byte copies of the same ring entries to shared
memory before QK, then waits before reading them. It preserves the original
token and dimension order, BF16 scales, FP8 conversion and merge. Split and
merge remain separate kernels and are measured together.

The observer recorded post-RoPE Q, the physical K/V ring, scales, control and
the consumed FP32 output from six local layers at positions 16,384, 16,951
and 17,517. The capacity and effective full window were both 1,024 tokens.
All 18 parent and candidate outputs match the consumed model bytes.
Parent and candidate also match on the partial output and merge workspace,
with sentinel guards intact. Another 18 cases exercise early lengths and ring
boundaries using the captured payload. Focused memcheck, synccheck and a
racecheck sample all passed.

The first two screen attempts are excluded. Uploads from pageable host
memory could still be in flight when the nonblocking replay stream launched
its first kernel. A second in-graph ring capture produced identical fixture
files. Uploading the fixtures on the replay stream and synchronizing it
before use made the replay agree with the actual model on all cases.

| Split and merge, isolated | Parent, µs/layer | Candidate, µs/layer | Savings, µs/token over 25 layers | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| Warm replay | 17.09 | 14.44 | 65.99 | [65.78, 66.29] |
| After 96 MiB eviction sweep | 38.92 | 28.68 | 255.87 | [254.40, 255.98] |

Each row uses 24 pairs per case in alternating order. Warm samples average
12 Split+Merge pairs per graph. Evicted samples time one pair with the sweep
outside the timed interval. Savings scale the mean of the per-case paired
medians to 25 layers. The intervals resample within the fixed set of model
cases. Both cache conditions passed the 40-µs screen. The isolated timings
omit projections, RoPE, KV writes and adjacent graph work.

In a 32-output production trace, the 31 decode steps each launched 25
specialized splits and 25 original merges. All 32 output IDs and all other
kernel counts and launch resources matched. The split moved from 49 to 43
registers per thread and from 1,056 to 33,824 bytes of static shared memory.
SASS shows 128-bit `LDGSTS` copies and a commit/wait pair. The short trace's
sum of split and merge activity fell from 800.36 to 595.27 µs/step.
Nsight Compute on one isolated split showed long-scoreboard stalls per
active issue falling from 4.65 to 2.85. Both traces describe execution
and exclude any claim about unprofiled generation throughput.

### Ordinary generation

The unprofiled ABBA campaign used the original 16,384-token prompt and
1,135 output tokens with batch one, checkpoint FP8 KV, three warmups and
ten measured runs per cohort. The output IDs, workspace, resident weight
and KV arena sizes, and sampled peak VRAM (12,455 MiB) matched. Every run
reported zero fallbacks and zero recurring allocations.

| Order | Parent, tok/s | Candidate, tok/s | Change | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| Parent then candidate | 138.99 | 141.32 | +1.67% | [+1.56%, +1.74%] |
| Candidate then parent | 139.00 | 141.33 | +1.67% | [+1.57%, +1.75%] |
| Pooled | 138.99 | 141.32 | +1.68% | [+1.59%, +1.73%] |

Both orders passed the Ordinary and prefill/TTFT gates. An earlier partial
parent cohort was discarded because it started without telemetry.

### Fixed-D2 regression check

| Order | Parent, tok/s | Candidate, tok/s | Change | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| Parent then candidate | 199.27 | 199.31 | +0.021% | [−0.094%, +0.151%] |
| Candidate then parent | 199.34 | 199.26 | −0.036% | [−0.193%, +0.119%] |
| Pooled | 199.27 | 199.30 | +0.011% | [−0.111%, +0.092%] |

The lower bounds passed the −0.20% non-inferiority margin in both orders.
Every run produced the same 1,135 IDs with 502 groups, 1,004 proposals
and 632 accepted proposals. Peak sampled VRAM was 12,725 MiB for both builds.
The two prefill and TTFT comparisons also passed their −1.0% margin.

### Qualification and decision

The staged split stays in the Trellis35 controlled Ordinary graph for the
1,024-slot local ring. In the 32-output D2 trace, 69 kernel signatures
matched in counts and launch resources, and the staged specialization
was never dispatched. The separate capture of the actual candidate build
matched all 18 parent boundaries byte for byte.

- The 16K production Ordinary memcheck completed with zero errors and the
  same eight output IDs. The focused CUDA test now checks exact output at
  positions 128, 1,023, 1,024 and 1,100. The model crossed the ring with
  both 1,024- and 2,048-token prefill settings and matched the parent.
- Nine host/CUDA CTest targets, the 12B product test and the 26B token
  selection/reset test passed. 12B Unified and internal 26B NVFP4 matched
  all 64 tokens across both builds and decode modes. Compact Vision
  reproduced the prior text on one real image in both modes at a
  70-soft-token budget.

The tested Linux build is the parent for the O-projection experiment.

## 2. FP8 O projection

The candidate places two adjacent eight-row output tiles in one CTA. It
changes the grid from 352 blocks of 64 threads to 176 blocks of 128
threads. The total warp count, source-order FP8 weights, BF16 scales, two
contiguous Split-K2 halves, FP8 QMMA instructions and FP32 sum are preserved.
The intended gain was lower CTA scheduling overhead with the same weight
traffic.

The screen used fresh one-token activations, scales and reference outputs
from local layers 0 and 15 and global layers 5 and 29 at all three replay
positions. Both implementations reproduce all twelve 2,816-element FP32
outputs byte for byte. The four matrices and their scales were read from
the pinned target and checked against the earlier extraction. Both output
guards, memcheck and synccheck passed.

| O replay | Local savings, µs/layer | Global savings, µs/layer | Savings, µs/token over 30 layers | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| Warm weights | −0.068 | −0.118 | −2.29 | [−3.17, −1.46] |
| Evicted weights | −0.037 | −0.008 | −0.97 | [−9.17, −0.08] |

There were 30 alternating-order pairs per case. The warm graph averages
eight launches per sample. The eviction sweep precedes each timed launch.
Positive values represent savings. The result weights per-case paired
medians by 25 local and five global layers. The screen measures O alone.
The primary evicted result missed the predeclared 30-µs gate.

Nsight Compute confirmed the new geometry. The candidate uses 48 registers
per thread and 64 bytes of static shared memory versus 45 and 32 for the
parent. In the separate cold profiles, eligible warps per active cycle
remained roughly 0.07 for both widths. Barrier stalls per issue rose from
0.30 to 0.86 locally and from 0.24 to 0.44 globally. The lower CTA count
offered no observed improvement in readiness for these profiles. The
kernel was kept in the isolated diagnostic, with the qualified
local-attention build as the production result.

The [machine-readable record](../../../artifacts/trellis35/local-attention-o-2026-11-05.json)
contains all paired operator samples, the unprofiled generation samples,
model and binary identities, and hashes of the raw evidence. The throughput
claim covers one 16K native workload on Linux and uses no new llama.cpp timing.
