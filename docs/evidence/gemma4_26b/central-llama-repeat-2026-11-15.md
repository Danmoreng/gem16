# Central llama.cpp repeat on 2026-11-15

Central llama.cpp was updated from the shared installation's November 5
build to official master `de32b268c4ad55d45f3b7b16c029f4e077f4ba74`
(`1.0.0-dev`). The checkout, executable version and official upstream
matched before every benchmark and at the final check.

The native control is the qualified gem16 binary used on November 5. Its
SHA-256 and the benchmark helper's SHA-256 are unchanged. The local checkout
is `58febf4d66bb028a35f6f7f9528d2206cfab2475`. GitHub main was
`655ae68cf14111d098bf5f851c3091745d4a6fb5` at setup. Its two additional
commits concern release, tests and documentation, with identical runtime
sources. The comparison uses the existing qualified binary.

## Decode

| Mode | gem16, tok/s | llama.cpp, tok/s | llama.cpp lead | 95% bootstrap interval | November 5 lead |
|---|---:|---:|---:|---:|---:|
| Ordinary | 141.74 | 144.33 | +1.83% | [+1.77%, +2.02%] | +1.56% |
| D2 | 200.40 | 202.87 | +1.23% | [+0.94%, +1.39%] | +1.39% |

llama.cpp's measured Ordinary rate is 0.52% above November 5 and its D2
rate is 0.24% above it. The unchanged native control is 0.26% and 0.40%
faster respectively. The NVIDIA driver changed from 615.78.08 to 620.70.06.
These cross-date changes include machine and driver effects.

## Prefill and first output

| Mode | gem16 prefill, tok/s | llama.cpp prefill, tok/s | gem16 lead | 95% bootstrap interval | gem16 prompt time | llama.cpp prompt time |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 5,612.11 | 5,358.57 | +4.73% | [+4.46%, +5.07%] | 2.919 s | 3.058 s |
| D2 | 5,632.00 | 5,355.65 | +5.16% | [+4.82%, +5.47%] | 2.909 s | 3.059 s |

Prefill covers the complete cold 16,384-token prompt. The runner uses
engine-reported prompt processing time as its time-to-first-token metric.
Model loading, tokenization and separately observed HTTP/streaming latency
are outside this metric. All samples came from the same runs as Decode.

## Method and limits

Both modes ran native / llama.cpp / llama.cpp / native, with three warmups
and ten retained measurements in each cohort. All twenty samples per
engine and mode are retained. Context requested was 17,519 tokens, padded
to 17,664 in llama.cpp. Both sides emitted 1,135 greedy outputs with
batch one, no prompt-cache reuse and resident weights. Decode measures
1,134 intervals after the first output. The suppressed token IDs, seed,
templates and input IDs match the previous comparison.

gem16 uses Trellis35 W4A8 with checkpoint FP8 KV. llama.cpp uses the
same Google QAT Q4_0 GGUF and Q8_0 KV as before, Flash Attention, eight
threads, prefill batch/microbatch 2,048/512 and CUDA embedding overrides.
Model files were verified against their immutable hashes. The native
Ordinary and Fixed-D2 outputs match each other and the previous fixture.
llama.cpp's output for each mode matches that mode's November 5 output.
The two engines' output streams differ.

The RTX 5080 Laptop GPU has 16,303 MiB. Firmware-managed clocks, active
Dynamic Boost and max-power were retained. Each primary cohort started
at or below 50 C. The new driver reports decimal temperatures. The local
measurement wrapper parses them as floating-point values with the same
50 C boundary. Peak VRAM was 12,457 / 12,727 MiB for native Ordinary/D2,
and 14,635 / 14,989 MiB for llama.cpp. Whole-device telemetry was sampled
every 200 ms and includes load, prefill and decode.

Separate Nsight traces verify CUDA Flash Attention and the active kernel
shapes on the measured llama.cpp binary. Profile timings are excluded
from the throughput samples. The native dispatch evidence is reused
from the identical qualified binary. This workload is a performance
characterization with different formats and output streams.

The [machine-readable record](../../../artifacts/llama_cpp/central-repeat-2026-11-15.json)
contains all retained timing samples, bootstrap intervals, configuration,
input/model/binary/library hashes, GPU telemetry summaries, and raw file
hashes. Raw commands, logs and traces remain under
`benchmarks/results/2026-11-15/58febf4/central-repeat/`.
