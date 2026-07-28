# Wikipedia 16K summarization characterization

This development characterization gives gem16, vLLM, and llama.cpp the exact same 16,384-token prompt and allows
up to 8,192 new tokens. The source is English Wikipedia's `Artificial intelligence` article at revision
`1366077412`; the German instruction requests a structured summary of its concepts, history, applications,
opportunities, and risks. Thinking is disabled.

The prompt token IDs are the comparison boundary. Their SHA-256, serialized as little-endian `uint32`, is:

```text
d07ad4d805944f0b87869da0c5bb44d99e8c43c0eb57d05a108ad80a6abb51a8
```

All engines use batch one, greedy selection, seed zero, the checkpoint stop/suppression controls, three warm-ups,
and ten measured repetitions. TTFT includes prompt processing and selection of the first output token. Decode
throughput measures the `generated_tokens - 1` intervals after that first token. EOS is allowed, so 8,192 is a
limit rather than a forced output length.

## Results

| Engine | KV cache | Prefill | TTFT | Decode | ITL | Generated tokens | Greedy output |
|---|---|---:|---:|---:|---:|---:|---|
| gem16 | checkpoint FP8 | 1,897.37 tok/s | 8,635.11 ms | 31.324 tok/s | 31.925 ms | 1,021-1,254 | 10/10 unique |
| vLLM 0.25.1 | FP8 | 4,328.03 tok/s | 3,785.56 ms | 33.971 tok/s | 29.437 ms | 1,215 | identical 10/10 |
| llama.cpp | Q8_0 | 2,160.83 tok/s | 7,582.29 ms | 28.843 tok/s | 34.670 ms | 1,088 | identical 10/10 |

gem16 reaches 43.84% of vLLM prefill throughput and 92.21% of its decode throughput. Against llama.cpp it
reaches 87.81% of prefill and 108.60% of decode throughput. These are development ratios, not parity speedups:
llama.cpp uses the patched closest-parity GGUF, maps the source FP8 attention weights to BF16, and uses Q8_0 KV.

The three decoded representative outputs are plausible German summaries. However, gem16's ten nominally greedy
runs produced ten different hashes and output lengths. vLLM and llama.cpp each produced one stable hash and length.
The gem16 result is therefore a correctness/reproducibility finding and must be resolved before this workload can
be promoted to an accepted baseline.

The race was subsequently isolated to reuse of shared reduction storage in the long-context online decode
attention and fixed after this characterization. The retained numbers and hashes above intentionally describe the
original base commit.

## Post-fix gem16 rerun

An engine-only rerun after adding the shared-result consumption barriers used the same workload, binary
configuration, three warm-ups, and ten measured repetitions:

| Metric | Original | Post-fix | Change |
|---|---:|---:|---:|
| Prefill | 1,897.37 tok/s | 1,892.37 tok/s | -0.26% |
| TTFT | 8,635.11 ms | 8,657.92 ms | +0.26% |
| Decode | 31.324 tok/s | 31.216 tok/s | -0.34% |
| ITL | 31.925 ms | 32.035 ms | +0.34% |

All ten post-fix runs generate exactly 1,106 tokens, stop normally, and share output hash
`44399253201aa729d72cf7a027b42ce2de9b1aa4ca98c1e9156435bb7e6628a8`. The decoded output is a coherent,
structured German summary covering the article's definitions, history, applications, opportunities, and risks.
The observed performance change is sub-percent. Because the decode-only barrier cannot affect prefill and no
continuous clock, power, or temperature telemetry was captured, this rerun supports a small cost but cannot
separate roughly 0.3% barrier overhead from same-machine run-to-run drift.

The older vLLM value of 6,146.50 tok/s is a 512-token prefill point. Its retained 8,192-token point is
3,929.14 tok/s; neither number describes this 16K summarization workload.

## Current MTP quick characterization

Commit `2dba16d` was run on the same exact 16,384-token prompt with one warm-up and three measured repetitions per
mode. This intentionally shorter repetition policy answers a bounded development question and is not a qualified
benchmark.

| Mode | Median decode | Change vs ordinary | Mean accepted length | Generated tokens |
|---|---:|---:|---:|---:|
| Ordinary | 31.775 tok/s | — | — | 1,135 |
| MTP D1 | 29.634 tok/s | -6.74% | 0.740 | 979 |
| MTP D2 | 31.702 tok/s | -0.23% | 1.240 | 979 |
| MTP D4 | 28.866 tok/s | -9.15% | 1.755 | 979 |

All repetitions are internally deterministic and the three MTP modes produce one common output. However, that
output first differs from ordinary greedy at zero-based generated index 68 (`58158` versus `119615`) and stops 156
tokens earlier. This violates the exact ordinary/MTP gate. The table therefore establishes that the current MTP
path provides no 16K win, but it cannot be used as an exact-output speedup comparison. D2 is effectively throughput
parity in this short run; D1 and D4 regress. The long-context BF16 assistant attention and additional rejected
proposals make longer drafting increasingly expensive.

Raw outputs and the explicit limitation record remain under
`benchmarks/results/2026-07-28/2dba16d/blackwell16gb-wikipedia16k-mtp/`.

### Correctness follow-up

A bounded D2 investigation used no warm-up and one run per candidate. It identified FP8 CUTLASS target Q/K/V as
the numerical divergence: restoring decode-order direct grouped Q/K/V while retaining exact CUTLASS O makes all
1,135 generated tokens equal ordinary. Moving long-context FP8 assistant attention from materialized scores to the
qualified split-online decode kernel retains the exact target output and raises the final D2 characterization to
35.184 tok/s at mean accepted length 1.259. Against the retained 31.775 ordinary median this is a +10.7% indication,
not a qualified comparison because repetition policies differ. The first exact direct candidate measured 30.031
tok/s; retaining CUTLASS O measured 30.692; split-online assistant attention supplied the material gain. An exact
batched-global target-attention experiment reached only 34.767 tok/s and was removed.

## Reproduction

First create the pinned exact-token workload with the local checkpoint tokenizer:

```bash
third_party/cache/unsloth-nvfp4-env/bin/python \
  tools/prepare_wikipedia_benchmark.py \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --output benchmarks/results/<date>/<git-sha>/<machine-id>/wikipedia-summary-16k/workload.json
```

Then run `tools/benchmark_wikipedia_workload.py` once per engine with the shared workload. The tool supports
`--engine gem16`, `--engine vllm`, and `--engine llama-cpp`:

```bash
python3 tools/benchmark_wikipedia_workload.py \
  --engine gem16 --workload <workload.json> --output <gem16.json> \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --executable build/Linux/blackwell-release/bin/gem16-run

# Add these arguments for an explicit MTP mode:
# --assistant-model models/checkpoints/google-gemma-4-12B-it-assistant-364bd03
# --mtp-draft-tokens 1|2|4

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
third_party/cache/unsloth-nvfp4-env/bin/python \
  tools/benchmark_wikipedia_workload.py \
  --engine vllm --workload <workload.json> --output <vllm.json> \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497

python3 tools/benchmark_wikipedia_workload.py \
  --engine llama-cpp --workload <workload.json> --output <llama-cpp.json> \
  --executable build/llama_cpp/release/bin/llama-server \
  --gguf build/llama_cpp/gemma4-12b-nvfp4-patched.gguf \
  --llama-kv-cache-type q8_0
```

Full per-run artifacts remain under `benchmarks/results/`; the tracked `characterization.json` retains the
comparison configuration, medians, confidence intervals, and samples.

## Limitations

- This is a development characterization, not an accepted parity benchmark.
- No continuous power, clock, temperature, or per-token percentile telemetry was captured.
- The engines stopped after different numbers of tokens, so their later decode contexts are not identical.
- gem16 and vLLM consume the direct mixed FP8/NVFP4 checkpoint. llama.cpp consumes the patched closest-parity
  GGUF and therefore differs in attention weight and KV precision.
