# Benchmarking

There are no accepted comparative benchmark results yet. `gem16-bench decode` now provides a real,
machine-readable batch-one decode characterization; the other end-to-end benchmark modes still return
`not_implemented` and a non-zero exit code.

The decode command keeps one model instance resident across all runs, clears the preallocated KV cache outside
the timing boundary, performs the configured warm-ups, and retains every measured inter-token latency in JSON.
It reports median/mean throughput, standard deviation, a Student-t 95% confidence interval across runs, and pooled
p50/p95/p99 inter-token latency. Prompt IDs follow the same deterministic formula as the tracked vLLM harness. The
first token selected after prompt ingestion is excluded from decode, after which exactly `--tokens` full forward,
greedy-selection, and device-to-host intervals are timed:

```powershell
.\build\Windows\blackwell-release\bin\gem16-bench.exe decode `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --context 128 `
  --tokens 256 `
  --warmups 3 `
  --repetitions 10
```

The hybrid cache supports the checkpoint's full 262,144-position contract: local-attention layers use a 1,024-token
ring while global-attention layers grow through the configured context. Results remain labeled `characterization`
and `benchmark_qualified: false` until long-context quality gates and required system telemetry
are complete.

Native chunked prefill is the only production path. The JSON records `prefill_path: native_chunked_sm120`. On the Windows Blackwell development machine,
a one-run context-128 parity check produced the same output checksum for both paths and reduced TTFT from 5,238.8 ms
to 1,479.0 ms. This is implementation evidence, not a repeated or accepted performance result.

`gem16-bench prefill` reports prompt token/s and TTFT separately with the same warm-up, repetition, raw-run, and
confidence-interval policy. For example:

```powershell
.\build\Windows\blackwell-release\bin\gem16-bench.exe prefill `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --context 128 --warmups 3 --repetitions 10
```

Future results must use the matrices, timing boundaries, repetition policy, quality gates, and three llama.cpp
baseline labels defined in `AGENTS.md`. Raw runs will be written below
`benchmarks/results/<date>/<git-sha>/<machine-id>/` and never overwritten. Throughput speedup and latency reduction
must be reported separately. Exact board identity, power envelope, clocks, and thermals are recorded for every run;
the project scope remains the 16 GB CUDA target class.

Current upstream llama.cpp is pinned, but its unpatched converter rejects the locked checkpoint's mixed FP8/NVFP4
compressed-tensors groups. A tracked converter patch produces a closest-parity candidate that preserves NVFP4 MLP
tensors and maps FP8 attention weights to BF16. Its inventory, direct-runtime quality comparison, and full-residency
probe are recorded; native-path profiling and quality acceptance remain open gates. This candidate cannot be labeled
exact format parity.

The patched llama.cpp candidate has a retained development characterization covering prefill through 65,536
tokens and decode at context depths through 8,192. Its tracked summary is
`benchmarks/baselines/llama_cpp/characterization.json`; raw samples are retained under `benchmarks/results/`. It is
not an accepted baseline because native dispatch profiling, a quality threshold, inter-token latency capture, and
power/clock/thermal telemetry remain open.

The pinned SM120a competitor build is complete. Its NVFP4 object contains native block-scaled
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` instructions. This is a binary capability check only; runtime dispatch must
still be captured with the selected model before a tier-B result is valid.

## 2026-07-27 current-commit cross-engine characterization

A fresh same-machine run at gem16 commit `c93a40d` uses batch one, identical deterministic prompt-token formulas,
three warm-ups, ten measurements, checkpoint-FP8 KV for gem16 and direct vLLM, and no CPU offload or prefix-cache
hits. The patched closest-parity llama.cpp candidate is freshly measured with BF16 KV, BF16-mapped source FP8
attention weights, and its native aggregate `llama-bench` boundaries. Median throughput is:

| Workload | gem16 | direct FP8 vLLM | patched llama.cpp | gem16/vLLM | gem16/llama.cpp |
|---|---:|---:|---:|---:|---:|
| Prefill 128 | 2,567.35 | 4,499.51 | 2,314.14 | 0.571x | 1.109x |
| Prefill 512 | 4,257.75 | 6,359.49 | 2,554.91 | 0.670x | 1.666x |
| Prefill 2,048 | 4,387.67 | 5,699.97 | 2,467.41 | 0.770x | 1.778x |
| Prefill 8,192 | 3,832.65 | 4,942.20 | 2,325.15 | 0.775x | 1.648x |
| Decode 128/256 | 33.21 | 39.20 | 29.49 | 0.847x | 1.126x |
| Decode 2,048/256 | 33.25 | 38.88 | 28.43 | 0.855x | 1.170x |
| Decode 8,192/256 | 32.59 | 38.06 | 27.99 | 0.856x | 1.164x |

This is still characterization rather than a parity headline. vLLM request TTFT includes scheduling and its first
output token, gem16 reports its documented host forward/selection boundary, and llama.cpp uses narrower
prompt-processing and aggregate-generation metrics. vLLM's FP4 autotuner encountered OOM and fallback tactics,
including an untuned 8K shape; llama.cpp changes attention-weight and KV precision. No run captured continuous
power/clock/thermal telemetry. The result nevertheless establishes the current engineering position: gem16 leads
the closest practical llama.cpp candidate across the matrix, while direct vLLM remains 22–43% ahead in prefill and
14–15% ahead in decode. Raw data and commands are under
`benchmarks/results/2026-07-27/c93a40d/blackwell16gb-linux-cross-engine/`.

## Direct vLLM development comparison

A batch-one vLLM 0.25.1 characterization now loads the pinned Hugging Face checkpoint directly with native FP8
attention weights, NVFP4 MLP weights, BF16 KV, CUDA Graphs, and no prefix caching or CPU offload. Across the common
128-to-8K range, its median prefill result is 1.66x to 2.34x the patched llama.cpp candidate and its median steady
decode result is 1.25x to 1.26x. These are not parity speedups: vLLM keeps FP8 attention while the GGUF maps those
weights to BF16, and the prefill timing boundaries differ.

The full table, methodology, raw samples, and limitations are under `benchmarks/baselines/vllm/`. In particular,
vLLM reported capacity for 10,303 BF16 KV-cache tokens at 95% GPU-memory utilization, so this run cannot cover 32K
or 65K. FlashInfer also used fallback tactics after some autotuning OOMs, including an untuned 8,192-token prefill
shape. The characterization remains development evidence rather than an accepted baseline.

## Shared Wikipedia 16K summarization workload

The retained real-workload characterization gives all three engines the same 16,384 prompt token IDs from a pinned
Wikipedia article revision, permits up to 8,192 generated tokens with normal EOS handling, and measures both TTFT
and the decode intervals after the first token. It uses checkpoint FP8 KV for gem16, FP8 KV for vLLM, and Q8_0 KV
for the patched closest-parity llama.cpp GGUF. The prompt hash, commands, raw samples, confidence intervals, and
format limitations are recorded under `benchmarks/baselines/wikipedia_summary_16k/`.

At base commit `7d29580`, median prefill throughput is 1,897.37/4,328.03/2,160.83 tok/s for
gem16/vLLM/llama.cpp; median decode is 31.324/33.971/28.843 tok/s. The representative outputs are plausible
German summaries, but gem16 produces ten distinct output hashes and lengths across ten nominally greedy runs,
while vLLM and llama.cpp are stable. The result therefore remains a development characterization and records a
determinism/correctness issue. The underlying shared-memory reduction race was subsequently fixed, but the retained
comparison is not rewritten. A same-workload gem16-only follow-up after the fix measures 1,892.37 tok/s prefill
and 31.216 tok/s decode, changes of -0.26% and -0.34% from the original medians. All ten runs now generate the
same 1,106-token output, share one hash, and stop normally. The decode-only barrier cannot explain the similarly
sized prefill shift, so the sub-percent decode delta is conservatively treated as barrier cost plus run-to-run
system drift rather than a precisely isolated kernel penalty.

The previously cited 6,146.50 tok/s vLLM result is specifically the retained 512-token prefill point. The retained
8,192-token point is 3,929.14 tok/s; the shared Wikipedia result above is the separate 16,384-token measurement.
