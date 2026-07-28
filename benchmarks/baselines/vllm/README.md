# vLLM direct-checkpoint characterization

This is a development characterization of vLLM 0.25.1 loading the pinned
`unsloth/gemma-4-12b-it-NVFP4` checkpoint directly. It is not an accepted baseline or a gem16 performance claim.

The run used batch one, BF16 KV cache, no CPU offload, token-ID input, no detokenization, no prefix cache, text-only
loading, chunked prefill, greedy decoding, CUDA Graphs, three warmups, and ten measured repetitions. vLLM selected
`CutlassFP8ScaledMMLinearKernel` for FP8 attention projections and `FlashInferCutlassNvFp4LinearKernel` for the
NVFP4 MLP. The full configuration and every sample are retained in `direct-bf16-kv-characterization.json`.

## Gemma 4 MTP characterization

vLLM 0.25.1 directly loads the official BF16 assistant and shares its three sliding layers with target Layer 46
and its full layer with target Layer 47. Its unmodified CUDA-Graph initialization fails because Python-list
suppression indexing constructs a CPU index tensor during capture. The auditable patch in
`patches/gemma4-mtp-suppress-graph.patch` replaces the two-ID list operation with graph-safe scalar indexing while
applying the same two suppression assignments.

On the 16,384-token Wikipedia workload with direct mixed FP8/NVFP4 target weights and FP8 KV, graph D1/D2/D4
screens reach 49.59/58.69/56.06 effective tok/s. The retained D2 3-warmup/10-run characterization reaches 57.390
median tok/s with mean 95% CI `[57.370,57.468]`, 556 accepted drafts over 513 groups, and a sampled 14,166 MiB
peak; median group latency is 36.24 ms. A separate fixed-1,135-token screen reaches 57.363 tok/s and 35.75 ms per
verifier group. Its ordinary
and MTP processes reserve 0.85 and 0.90 GPU-memory-utilization respectively because MTP initialization rejected
the lower KV reservation; this does not qualify as an identical-memory-policy comparison.

Apply the patch only to the pinned reference environment after verifying the original file SHA-256 is
`4eee061c81430be28f029ed66360887a57f8711a75c863067d30e3840a488918`:

```bash
patch -p1 -d third_party/cache/unsloth-nvfp4-env/lib/python3.13/site-packages \
  < benchmarks/baselines/vllm/patches/gemma4-mtp-suppress-graph.patch
```

These are **not exact speculative-decoding baseline results**. vLLM ordinary and MTP are each deterministic, but
they first differ at output index 33 under stop semantics and index 2 in the fixed-length screen. The ordinary
and MTP runs also choose different target batch shapes and, in the stop-terminated characterization, emit 1,215
and 1,068 tokens. The numbers establish hardware/performance headroom only. Machine-readable details are in
`mtp-characterization.json`; raw runs and 200 ms telemetry remain ignored under `benchmarks/results/`.

## Comparison with llama.cpp

The llama.cpp candidate is the patched same-source closest-parity GGUF characterized in
`../llama_cpp/characterization.json`. Its MLP remains NVFP4, but its FP8 attention weights were converted to BF16.
Both runs used BF16 KV and batch one on the same machine. These format and timing-boundary differences prevent an
exact parity or headline speedup claim.

| Workload | vLLM median | llama.cpp median | vLLM / llama.cpp |
|---|---:|---:|---:|
| Prefill 128 | 4,679 tok/s | 2,215 tok/s | 2.11x |
| Prefill 512 | 6,146 tok/s | 2,628 tok/s | 2.34x |
| Prefill 2,048 | 4,913 tok/s | 2,539 tok/s | 1.93x |
| Prefill 8,192 | 3,929 tok/s | 2,362 tok/s | 1.66x |
| Decode at context 128 | 37.06 tok/s | 29.67 tok/s | 1.25x |
| Decode at context 2,048 | 35.98 tok/s | 28.87 tok/s | 1.25x |
| Decode at context 8,192 | 35.36 tok/s | 28.08 tok/s | 1.26x |

For vLLM, prefill throughput is prompt tokens divided by its request-level time to first token, so it includes
scheduling and production of the first token. llama.cpp's `llama-bench` prompt-processing metric uses a narrower
boundary. vLLM decode throughput uses 256 intervals after one untimed first token; llama.cpp reports aggregate
generation throughput. The ratios are useful development indicators, not publication-grade speedups.

## Limitations observed

- The 512-token vLLM prefill point was noisy; all samples and its wide confidence interval are retained.
- FlashInfer autotuning encountered VRAM exhaustion for some tactics and used documented default fallbacks.
- The 8,192-token FP4 prefill shape was outside the tuned bucket range and used the default CUTLASS tactic.
- At 95% GPU-memory utilization, vLLM reported 10,303 BF16 KV-cache tokens. This supports the common comparison
  through 8K but not the required 32K/65K matrix. A later FP8-KV run must be labeled separately.
- Neither characterization captured power, clocks, thermals, profiler-level dispatch, or p95/p99 per-token latency.
- The llama.cpp candidate still lacks an adopted quality threshold and native-dispatch trace.

Reproduce from the pinned external reference environment:

```bash
PATH="$PWD/third_party/cache/unsloth-nvfp4-env/bin:$PATH" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
third_party/cache/unsloth-nvfp4-env/bin/python tools/benchmark_vllm.py \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --output benchmarks/results/<date>/<git-sha>/<machine-id>/vllm/direct-bf16-kv.json \
  --prefill-lengths 128,512,2048,8192 \
  --decode-contexts 128,2048,8192 \
  --decode-tokens 256 --warmups 3 --repetitions 10 \
  --max-model-len 8449 --gpu-memory-utilization 0.95 \
  --kv-cache-dtype bfloat16
```
