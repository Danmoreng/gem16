# Recorded performance

These are retained measurements at the revisions named below, not a new benchmark of current HEAD.
The public profiles and qualification limits are described in the [README](../README.md).

## Gemma 4 26B performance

The repeatable Linux measurements below use the RTX 5080 Laptop GPU at its firmware-managed 175 W ceiling, batch
one, the fixed 16,384-token Wikipedia prompt, checkpoint-FP8 KV, three warm-ups, and ten retained runs or alternating
pairs. Throughput counts generated tokens, not speculative proposals.

| GEM16 26B Target | Prefill tok/s | Ordinary decode tok/s | Fixed-D2 decode tok/s | Generated output | Peak VRAM Ordinary / D2 |
|---|---:|---:|---:|---:|---:|
| **Trellis35 W4A8** — public Compact Vision text Target | 5,666.57 | 130.906 | **182.526** | 1,229 tokens | 12,522 / 12,794 MiB |
| Internal hybrid NVFP4/FP8/BF16 | **6,926.23** | **148.336** | **203.552** | 942 tokens | 14,734 / 15,026 MiB |

For each Target, Ordinary and D2 produced identical token vectors in every retained run. Trellis35 D2 accepted 690
of 1,076 proposals; NVFP4 D2 accepted 523 of 836. The later retained K/V-epilogue, programmatic-dependency, and
fixed-row attention improvements reached short-screen medians of 131.77/approximately 184.0 tok/s for Trellis35
and 150.44/206.40 tok/s for NVFP4 (Ordinary/D2). Those runs intentionally used only one warm-up and three
measurements, so they document the current implementation but do not replace the 3+10 table above. See the
[Trellis35 freeze](../artifacts/trellis35/pfx31-performance-freeze.json),
[NVFP4 recheck](../artifacts/m25/nvfp4-wikipedia-recheck-2026-09-04.md), and
[latest bounded optimization record](../artifacts/m25/optimization-pdl-attention-followup-2026-09-04.json).

### 26B comparison with llama.cpp and vLLM

These rows are useful directional same-machine references, not model-quality parity: the engines use different
checkpoint formats, KV precision, output sequences, and in some cases different output lengths.

| Engine / 26B checkpoint | Prefill tok/s | Ordinary decode tok/s | MTP D2 tok/s | Output / validity | Peak VRAM |
|---|---:|---:|---:|---|---:|
| **gem16**, internal hybrid NVFP4/FP8/BF16 | **6,926.23** | 148.336 | **203.552** | 942 sampled tokens; deterministic; Ordinary = D2 | 15,026 MiB D2 |
| llama.cpp b10623, official QAT Q4_0 GGUF | 4,262.13 | 118.627 | 151.919 | 1,135 forced greedy tokens; Ordinary and D2 internally deterministic but differ from each other | 15,538 MiB D2 |
| vLLM 0.27.1, community W4A16, CUDA Graph | 6,475.80 | **149.348** | — | 64 forced greedy tokens; three different output hashes | 15,818 MiB |

The raw directional ratios put gem16 NVFP4 62.51% ahead of llama.cpp in prefill, 25.04% in Ordinary decode, and
33.99% in D2 decode for these recorded rows. They are not exact semantic comparisons. llama.cpp uses host-controlled
MTP with CUDA-offloaded forwards and its D2 run leaves only 343 MiB free, below gem16's 700 MiB operating-reserve
target.

There is no valid vLLM NVFP4/MTP row to publish. vLLM 0.27.1 could not map Google's official text-only Q4_0 GGUF
and produced no throughput measurement. The community W4A16 Target ran only in Ordinary mode; its repeated greedy
outputs were non-deterministic and its 15,818 MiB peak left 63 MiB free. A fully GPU-resident 26B vLLM MTP engine
did not fit on the 16 GB GPU with either the tested BF16 or ModelOpt NVFP4 Assistant. See the
[llama.cpp 26B record](../benchmarks/baselines/llama_cpp/gemma4-26b-a4b-qat-q4_0-mtp-b10623.json),
[vLLM W4A16 record](../benchmarks/baselines/vllm/gemma4-26b-w4a16-wikipedia-16k64-characterization.json), and
[vLLM loader audit](../benchmarks/baselines/vllm/gemma4-26b-q4_0-load-characterization.json).

## Gemma 4 12B 16K comparison

### Linux

On an RTX 5080 Laptop GPU at its firmware-managed 175 W ceiling, gem16 commit `a819d14c`, vLLM 0.26.0, and
llama.cpp b10240 produce the following same-machine result:

| Engine | Checkpoint / KV | Prefill tok/s | TTFT | Effective D2 MTP tok/s | ITL | Sampled peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| vLLM 0.26.0 | direct FP8/NVFP4 / FP8 | **6,257.37** | **2,618.35 ms** | 82.25 | 12.158 ms | 15,764 MiB |
| **gem16** | direct FP8/NVFP4 / FP8 | 5,866.86 | 2,792.64 ms | **87.66** | **11.408 ms** | 11,746 MiB |
| llama.cpp b10240 | patched NVFP4+Q8_0 GGUF / Q8_0 | 3,941.23 | 4,157.08 ms | 83.89 | 11.921 ms | **10,630 MiB** |

### Windows

On Windows 11 x64, the same RTX 5080 Laptop GPU running the freshly rebuilt gem16 commit `35a57bb` and llama.cpp
b10240 in Lenovo Max Power mode with CUDA 13.3 and driver 596.49 produces the following adjacent same-machine result:

| Engine | Checkpoint / KV | Prefill tok/s | TTFT | Effective D2 MTP tok/s | ITL | Sampled peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| **gem16** | direct FP8/NVFP4 / FP8 | **6,042.99** | **2,711.24 ms** | **89.00** | **11.237 ms** | 11,713 MiB |
| llama.cpp b10240 | patched NVFP4+Q8_0 GGUF / Q8_0 | 3,942.08 | 4,156.18 ms | 86.80 | 11.521 ms | **10,599 MiB** |

All rows use batch one, the same 16,384-token Wikipedia prompt, 1,135 fixed output positions, fixed D2 MTP, three
warm-ups, and ten measurements. The Windows engines ran serially after cooling to at most 50 C and reached
175.86/176.75 W; all measured outputs were deterministic. Both output hashes match the Linux run. vLLM
is omitted on Windows because its pinned runtime is unsupported there.

On Windows, gem16 leads llama.cpp by **53.29% in prefill** and **2.53% in decode**, with 34.77% lower TTFT and
2.47% lower ITL. On Linux, gem16 decode leads vLLM by **6.57%** and llama.cpp by **4.49%**, while prefill is 6.24%
behind vLLM and 48.86% ahead of llama.cpp. These are controlled performance comparisons, not exact format or
semantic parity; checkpoint formats, KV precision, output hashes, and prefill timing boundaries differ.

Reproduce the Linux comparison after preparing the pinned competitor environments:

```bash
sudo systemctl enable --now nvidia-powerd.service
echo max-power | sudo tee /sys/firmware/acpi/platform_profile
systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 \
  ./scripts/benchmark-cross-engine-mtp.sh
```

See the [full methodology](../benchmarks/baselines/cross_engine_mtp/README.md),
[Linux data](../benchmarks/baselines/cross_engine_mtp/characterization-a819d14c.json), and
[Windows data](../benchmarks/baselines/cross_engine_mtp/windows-characterization-35a57bb.json).

## Gemma 4 12B short-context comparison

### Linux

The completed Linux max-power investigation at gem16 commit `065b68f` and llama.cpp b10240 gives this standard
`llama-bench`-shape characterization:

| Engine | Checkpoint / KV | pp512 tok/s | tg128 tok/s | Aggregate time/token |
|---|---|---:|---:|---:|
| **gem16** | direct FP8/NVFP4 / FP8 | **7,026.84** | 52.52 | 19.039 ms |
| llama.cpp b10240 | patched NVFP4+Q8_0 GGUF / F16 | 5,381.87 | **62.76** | **15.935 ms** |

Gem16 leads pp512 by 30.57%; llama.cpp leads the disclosed tg128 counterpart by 19.48%. Nsight attributes only
about 0.013 ms/token to gem16's extra final argmax/publication boundary. Both admitted optimization candidates
failed their rejection screens and were removed, so the production runtime is unchanged. The direct all-regions
probe leaves 4.04 GiB CUDA-visible free, passing the 700 MiB reserve gate. See the
[tracked Linux summary](../benchmarks/baselines/llama_cpp/linux-short-context-065b68f.json).

### Windows

The standard `llama-bench` `pp512`/`tg128` matrix on Windows 11 x64 and the RTX 5080 Laptop GPU gives the following
batch-one characterization at gem16 commit `cc01a05` and llama.cpp b10240:

| Engine | Checkpoint / KV | pp512 tok/s | Prompt time | tg128 tok/s | Aggregate time/token | Sampled peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| **gem16** | direct FP8/NVFP4 / FP8 | **6,877.29** | **74.448 ms** | 52.43 | 19.072 ms | 10,634 MiB |
| llama.cpp b10240 | patched NVFP4+Q8_0 GGUF / F16 | 5,400.91 | 94.799 ms | **62.08** | **16.109 ms** | **9,824 MiB** |

Gem16 is **27.34% faster for pp512** with 21.47% lower reported prompt time. Llama.cpp is **18.39% faster for
tg128** with 15.54% lower aggregate time per token. Both rows use three discarded conditioning samples and ten
reported samples; `llama-bench` also performs its built-in warm-up. The reported runs started from an idle GPU at
50–51 C and reached 66 C.

This is a standardized shape and timing-boundary characterization, not exact token or format parity. `llama-bench`
uses synthetic random token IDs, begins `tg128` at position zero, and reports aggregate generation time. Gem16 uses
its deterministic benchmark tokens, begins after the smallest supported one-token context, and additionally records
per-token latency. Checkpoint attention format and KV precision also differ. See the
[complete samples, commands, and telemetry](../benchmarks/baselines/llama_cpp/windows-short-context-cc01a05.json).

