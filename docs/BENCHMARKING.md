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

## HTTP server characterization

`tools/benchmark_server.py` measures the protocol-facing batch-one paths rather
than relabeling core CUDA timings. Its four scenarios are:

- a new non-streaming Responses root, including slot creation and prompt work;
- a non-streaming resident continuation whose setup root is outside timing;
- a streamed root with wall time and time to the first reasoning/text delta;
- simultaneous resident continuations on independent pre-created slots, with
  per-lane latency and aggregate accepted output tokens/s.

The default is three warm-ups and ten retained measurements per scenario. JSON
contains every raw run, median/mean/standard deviation, two-sided Student-t 95%
mean intervals, p95/p99, cache-hit/write usage, server health, and Prometheus
counter snapshots/deltas. Setup roots are intentionally excluded from resident
and concurrency timing but remain visible in metric deltas. Concurrency requires
`--max-sessions` at least as large as `--concurrency`.

```bash
python tools/benchmark_server.py \
  --base-url http://127.0.0.1:8080/v1 --model gem16 \
  --scenario all --warmup 3 --repetitions 10 \
  --output benchmarks/results/2026-07-30/<git-sha>/<machine-id>/server.json
```

The output path is never overwritten. HTTP wall throughput is not interchangeable
with `gem16-bench` core-GPU throughput: root timing includes admission and slot
construction, streaming TTFT includes HTTP scheduling and prompt processing,
and the concurrent result measures contention between isolated execution slots.

### Long multimodal resident conversation

`tools/benchmark_server_long_conversation.py` owns a fresh server process and
refuses to run unless `/health` confirms exactly one slot, a 262,144-position
plan, checkpoint-recommended sampling (`temperature=1`, `top_k=64`,
`top_p=0.95`), FP8 KV, and fixed D2 rather than adaptive MTP. The root turn sends
one real image and audio recording. Later text-only turns extend that same
Responses chain toward 2K, 8K, 32K, 64K, and 128K actual input positions; no
prompt-cache reset or reconstructed root is used.

At each depth, one default warm-up and three streamed measured turns add a small
new prompt probe. The report separates:

- engine prompt milliseconds and new cache-write tokens/s;
- HTTP time to the first reasoning/text delta;
- engine decode milliseconds and target-verified output tokens/s;
- streamed delta-interval p50/p95/p99, where near-zero intervals expose tokens
  published together by an accepted MTP group;
- MTP proposal, acceptance, rejection, group, and fallback counters;
- actual rather than nominal context depth and cache hit/write counts;
- complete-run power, clocks, thermals, and VRAM;
- initial multimodal recognition and final 128K media-retrieval output.

Filler turns are part of the same conversation and now also retain their exact
bulk-prefill time and throughput. They are not mixed into the small-suffix
checkpoint distribution. This distinction avoids presenting a 70-token cached
continuation as large-batch prompt throughput.

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

## Single-run 128K and maximum-context QA characterization

`tools/benchmark_long_context_qa.py` repeats the pinned Wikipedia article body to a requested exact token count,
places three article-grounded questions at the end, forces 256 greedy output positions for a stable decode sample,
and collects GPU telemetry. The maximum-context case uses 261,888 prompt plus 256 output positions because a full
262,144-token prompt would leave no legal decode position.

| Prompt/output positions | Prefill | Prompt throughput | Decode | Decode throughput | Peak sampled GPU memory |
|---:|---:|---:|---:|---:|---:|
| 131,072 + 256 | 116.734 s | 1,122.83 tok/s | 12.833 s | 19.87 tok/s | 11,022 MiB |
| 261,888 + 256 | 410.978 s | 637.23 tok/s | 18.461 s | 13.81 tok/s | 12,244 MiB |

Both runs use checkpoint FP8 KV, complete all requested positions, report zero fallbacks and no token-loop
allocation, and retain the hybrid local-ring/global-contiguous cache. The 128K answer completes all requested
article questions before its first turn-end token. The maximum-context answer correctly covers the Dartmouth 1956
origin, Turing test, and requested applications/opportunities, but reaches the forced 256-token limit while
answering the final risk portion. These are one-run capacity, latency, and quality characterizations rather than
statistically qualified performance results.

## MTP timing

The active `batched_exact_target` scheduler executes the complete BF16 assistant, evaluates the input plus up to
four drafts in one target batch, and accepts and commits only the verified prefix on GPU. Proposed tokens are never
counted as output. Draft lengths 1, 2, and 4 report proposed, accepted, rejected, mean accepted length, target
batches, incremental VRAM, draft-length group counts, and ordinary fallback tokens.

The current qualified MTP result uses the exact 16,384-token Wikipedia workload, checkpoint-FP8 KV, three alternating
warm-up pairs, and ten alternating measured ordinary/D2 pairs. All runs emit the same 1,135 IDs. Ordinary and D2
medians are 36.788 and 54.903 effective target-verified tok/s, respectively (1.492x, +49.2%). Their 95% mean
confidence intervals are `[36.715,36.837]` and `[54.557,55.132]`. Raw runs are retained under the ignored result
path documented in the performance ledger; continuous telemetry was not captured. This result does not
make MTP universally preferable: every new workload must retain identical output semantics and report ordinary,
explicit draft, acceptance, and adaptive/fallback behavior under the same repetition policy.

External MTP characterizations use the same prompt and also include a fixed-1,135-token, ignore-EOS screen to
control output count when numerical differences change the stop point. Patched graph-vLLM reaches 57.390 tok/s in
a 3/10 stop-terminated D2 run and 57.363 tok/s in the fixed-length screen. That fixed screen reserves 0.85/0.90
GPU utilization for ordinary/MTP because MTP rejected the lower reservation. Current llama.cpp reaches 48.38 tok/s
fixed-length D2. Neither runtime's MTP output equals its own ordinary greedy output, so these values are hardware
bounds only and cannot be called exact speculative speedups or compared as quality-parity headlines. vLLM uses the
direct mixed checkpoint and FP8 KV; llama.cpp uses BF16-mapped attention and Q8_0 KV. See the corresponding
`mtp-characterization.json` files for complete disclosure.

The active competitive gate is measured only on the fixed 16,384-prompt/1,135-output workload: 50.0 exact
effective tok/s is the minimum and 55.0 tok/s is the stretch target. At the retained D2 acceptance these correspond
to at most 45.18 and 41.07 ms per verifier group. The GPU-chained path passes the minimum at 54.903 tok/s and misses
the stretch target by 0.097 tok/s. The same 3-warm-up/10-run and exact-ID policy applies.

Run the final paired qualification with the checked-in alternating orchestrator:

```bash
python tools/qualify_mtp.py \
  --workload benchmarks/results/<workload>/workload.json \
  --output benchmarks/results/<date>/<git-sha>/<machine>/mtp-qualification.json \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --assistant-model models/checkpoints/google-gemma-4-12B-it-assistant-364bd03 \
  --executable build/Windows/blackwell-release/bin/gem16-run.exe \
  --warmup-pairs 3 --measured-pairs 10
```

The tool alternates which mode runs first in each pair, retains raw per-mode runs and pair order, and fails if any
warm-up or measured output differs from the shared ordinary/MTP token sequence.
