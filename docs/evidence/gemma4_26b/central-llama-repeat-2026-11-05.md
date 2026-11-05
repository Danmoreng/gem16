# Gemma 4 26B after the Ordinary attention patch

This Linux repeat compares gem16 `1d46cc5a5441d7883dfd6059225e2d16ca39a7fe`
with central llama.cpp `b1c902f026d32535b5e8eae55e5e47b379e57672` on the
RTX 5080 Laptop GPU (16,303 MiB, NVIDIA driver 615.78.08). The native binary
matches the [Ordinary attention qualification](local-attention-o-2026-11-05.md).
The official llama.cpp master matched the measured build at every llama.cpp
benchmark start. At post-run verification, upstream was at
`6023692b1e13a6d4ee6b93f989c4713e474e6165`. The separate CUDA dispatch traces
used the measured binary by hash.

| Decode | gem16 median | llama.cpp median | llama.cpp lead | 95% bootstrap interval | Previous lead |
|---|---:|---:|---:|---:|---:|
| Ordinary | 141.37 tok/s | 143.58 tok/s | 2.21 tok/s (+1.56%) | [+1.26%, +1.85%] | 4.45 tok/s (+3.19%) |
| D2 | 199.60 tok/s | 202.38 tok/s | 2.78 tok/s (+1.39%) | [+1.31%, +1.45%] | 3.83 tok/s (+1.93%) |

The previous full comparison used gem16 `66b74d7` and llama.cpp `9e2a04a`.
The absolute leads narrowed in this repeat. Measured llama.cpp throughput
changed by −0.26% in Ordinary and +0.18% in D2 across the campaigns.
The earlier and current runs were conducted separately.

| Sampled peak VRAM | gem16 | llama.cpp |
|---|---:|---:|
| Ordinary | 12,455 MiB | 14,633 MiB |
| D2 | 12,725 MiB | 14,989 MiB |

## Controls

Each mode used gem16 / llama.cpp / llama.cpp / gem16 order, three warmups
and ten retained measurements per cohort. The workload used 16,384 identical
prompt token IDs, 1,135 forced greedy outputs and 1,134 timed intervals after
the first output. Requested context was 17,519 and llama.cpp padded it to
17,664. There was one slot, no prompt-cache reuse, and resident model weights.
Clocks were firmware-managed under the same max-power profile. VRAM and
clock telemetry was sampled every 200 ms. The separate traces did not
contribute to throughput.

gem16 used Trellis35 W4A8 with checkpoint FP8 KV. llama.cpp used Google's
QAT Q4_0 GGUF with Q8_0 KV, Flash Attention, eight threads, and prefill
batch/microbatch 2,048/512. The output streams differed across engines.
gem16 reproduced the same Ordinary and Fixed-D2 outputs. llama.cpp
reproduced each mode's previous output stream. The llama.cpp D2 setting
permits up to two draft tokens. Its cumulative server counters indicate
496 two-draft groups and one one-draft group per request in this workload.

The [machine-readable record](../../../artifacts/llama_cpp/central-repeat-2026-11-05.json)
includes all retained timing samples, both order comparisons, input hashes,
binary and library hashes, dispatch shapes, and telemetry summaries. Raw
commands, server logs, and profiler reports remain in the local result directory.
