# Benchmarking

There are no accepted comparative benchmark results yet. `gem16gb-bench decode` now provides a real,
machine-readable batch-one decode characterization; the other end-to-end benchmark modes still return
`not_implemented` and a non-zero exit code.

The decode command keeps one model instance resident across all runs, clears the preallocated KV cache outside
the timing boundary, performs the configured warm-ups, and retains every measured inter-token latency in JSON.
It reports median/mean throughput, standard deviation, a Student-t 95% confidence interval across runs, and pooled
p50/p95/p99 inter-token latency. Prompt IDs follow the same deterministic formula as the tracked vLLM harness. The
first token selected after prompt ingestion is excluded from decode, after which exactly `--tokens` full forward,
greedy-selection, and device-to-host intervals are timed:

```powershell
.\build\Windows\blackwell-release\bin\gem16gb-bench.exe decode `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --context 128 `
  --tokens 256 `
  --warmups 3 `
  --repetitions 10 `
  --enable-fused-gate-up
```

The hybrid cache supports the checkpoint's full 262,144-position contract: local-attention layers use a 1,024-token
ring while global-attention layers grow through the configured context. Results remain labeled `characterization`
and `benchmark_qualified: false` until native prefill, long-context quality gates, and required system telemetry
are complete.

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
