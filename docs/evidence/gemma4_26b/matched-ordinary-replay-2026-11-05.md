# Ordinary Decode with a matched token history

This diagnostic replays identical input tokens on gem16
`d1b6a3c66e1f299fa02e74c25bc30cf0300c4db2` and the verified central llama.cpp
`45b6604b20f17b7d7638042306c0cd1b6af5999c`. It measures Gemma 4 26B A4B on
one RTX 5080 Laptop GPU with resident weights. The models are Trellis35
W4A8 with FP8 KV and Google's QAT Q4_0 GGUF with Q8_0 KV. Both weight
payloads were checked against their pinned hashes.

## Replay speed

The two prompts are the full 16,384-token AP1 prompt and its first 2,048
tokens. Each engine processes the same 1,134 forced tokens after prefill,
with batch one and a fresh cache. The forced sequence comes from the
native 16K fixture. The 2K run changes the prompt prefix while keeping that
sequence. Context capacities were 4,096 on both sides for 2K and 17,519
native versus 17,664 effective in llama.cpp for 16K.

| Prompt | gem16 replay steps/s | llama.cpp replay steps/s | llama.cpp advantage in both orders |
|---|---:|---:|---:|
| 2K | 147.32 | 175.42 | +19.05% / +19.09% |
| 16K | 139.91 | 147.05 | +5.09% / +5.11% |

Each median combines six unprofiled runs per engine and prompt length.
The orders were native/llama/llama/native for 16K and the reverse for 2K,
with three retained processes in each cohort and matched cooldown. The
rates invert the mean measured step latency for each run. Model loading
and prefill are outside the step. gem16 times the forward and its normal
token selection. llama.cpp times one C API decode with synchronization and
requested logits. Its host-side masked argmax for sampled steps is
outside the timer. Both paths run the output head. These rates describe
the diagnostic replay boundary. The regular generation benchmark has a
different timing boundary.

Sampled peak VRAM for 2K was 12,319 MiB native and 14,481 MiB in llama.cpp.
For 16K it was 12,455 and 14,633 MiB. GPU clocks were firmware-managed.

## GPU time per step

There is one separate full CUDA trace per engine and prompt length.
The analysis checked all 1,134 step ranges in each trace, the 30 layers
per step and their projection, attention, router and expert anchors.
gem16's local grouped QKV and global combined QK paths were identified
separately. llama.cpp's Q, K, V and O kernels were identified by their
per-layer order, shapes and Flash Attention boundaries. The end of
attention and start of the feed-forward path have a separate transition
bucket for normalization, residual and router preparation.

| GPU interval union, ms/step | 2K gem16 | 2K llama.cpp | 16K gem16 | 16K llama.cpp |
|---|---:|---:|---:|---:|
| Attention block including QKV/O and KV work | 3.077 | 2.202 | 3.406 | 3.082 |
| QKV/O matrix kernels | 1.593 | 1.233 | 1.595 | 1.276 |
| O matrix kernel alone | 0.556 | 0.325 | 0.553 | 0.344 |
| Local attention core and combine | 0.735 | 0.403 | 0.737 | 0.418 |
| MoE from routing through output | 2.656 | 2.621 | 2.674 | 2.744 |
| MoE including transition | 3.111 | 2.671 | 3.132 | 2.801 |
| Routed GateUp matrix kernels | 1.155 | 0.844 | 1.171 | 0.908 |

Each row is a union of the relevant CUDA activity intervals. Rows can
share kernels or overlap in time. The record includes a separate
disjoint partition for transfers, exclusive activity, overlap and idle.
GPU activity timing can include dependency waits inside a kernel.
No causal speedup is obtained by subtracting an individual row.

The native Attention block consistently takes longer, with an observed
gap of 0.874 ms at 2K and 0.324 ms at 16K. The local attention core and O
projection both have stable gaps across the two lengths. The llama.cpp
global attention path also shows Q8_0-to-FP16 cache conversion. Its
interval union grows from 0.064 ms at 2K to 0.502 ms at 16K.

The Routed GateUp gap is real in both traces. The whole MoE block from
routing is much closer, with shared and routed work overlapping on
gem16. Adding the transition shows a larger native gap around the block.
The two engines' formats, router paths, scheduling and clocks all
contribute to the observed costs. A format-only effect would require
another controlled comparison.

## Expert routes

We compared eight consecutive steps near the start, middle and end of
each replay. That gives 720 layer/token pairs per prompt length. Native
copies its router IDs from the diagnostic layer captures. For llama.cpp,
a separate observer reads the eight IDs consumed by the actual routed
GateUp CUDA kernel, with the normal GGML scheduler and graph path. The
observer clears its slots before each sample. All 30 layers repopulate
valid distinct IDs and the sampled predictions match the unprofiled
engine on both sides. Route-capture timings are excluded.

| Prompt | Mean common experts out of eight | Identical sets |
|---|---:|---:|
| 2K | 6.96 | 294 / 720 |
| 16K | 7.40 | 362 / 720 |

The earlier llama.cpp GGML callback probe split the execution graph.
Its sets differed from the direct observation on 96 pairs at 2K and 53
at 16K. It also changed one sampled next-token prediction at 2K. The
route table uses the direct GPU observation.

The next bounded experiment would replay the local attention core at the
same effective sliding window on real inputs. Measure split plus combine
as a unit and keep cache format and attention semantics fixed inside each
engine. The native O projection is a second target with a similar measured
gap at both prompt lengths. A K3 codec trial remains useful for Routed
GateUp, with its benefit judged against the full MoE and model timings.

The [machine-readable record](../../../artifacts/trellis35/matched-ordinary-replay-2026-11-05.json)
contains all unprofiled step times, matched profile windows, sampled
routes, GPU telemetry, source and model identities, and raw-trace hashes.
