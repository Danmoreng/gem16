# Benchmarking

## Gemma 4 26B Fast-Track benchmark policy

M20 benchmarks one frozen M17 artifact. M19, M20 and M21 may collect evidence in parallel, but all reports must reference the same artifact/binary/configuration hashes. Official Q4_0 and Unsloth remain external references; no internal Q4 backend is required. M25 compares ordinary Target and MTP using only Target-verified output tokens and reports a separate MTP context profile.


A controlled same-machine 16K D2 performance comparison is published below. It is not an exact
output/semantic-parity result. `gem16-bench decode` and `gem16-bench prefill` provide machine-readable batch-one
characterizations. The advertised `model-load`, `end-to-end`, `kernel`, `memory`, `quality`, and `mtp` subcommands
still return `not_implemented` and a non-zero exit code.

## Permanent benchmark contract

A promoted comparison uses the same GPU, power policy, prompt token IDs, output-token count, context, batch size,
sampling controls, template, cache state, and timing boundary. Any necessary difference in checkpoint, tensor
format, K/V precision, runtime feature support, or output semantics is stated beside the result. CPU weight offload,
prompt-cache asymmetry, reduced output, and hidden fallback invalidate a primary performance claim.

Unless a stricter feature plan applies, use three warm-ups and ten retained runs, retain every raw sample, and
report the median as primary plus mean, standard deviation, and a 95% confidence interval. Noisy results require
more repetitions, not selection of the best run. Record peak and steady VRAM, power, clocks, thermals, exact
commands, code/model/toolchain revisions, and actual kernel/graph dispatch. Results are stored below
`benchmarks/results/<date>/<git-sha>/<machine-id>/` and never overwritten.

Keep prompt processing, ordinary decode, speculative decode, and end-to-end timing distinct. Report TTFT and
prompt tok/s for prefill; output tok/s and median/p95/p99 ITL for decode. Speculative proposed tokens are not output
tokens. Throughput speedup is `candidate / baseline`; latency reduction is `1 - candidate / baseline`.

External llama.cpp comparisons use one of these labels:

1. **same-source closest parity**: the same immutable source with every format mapping and quality difference
   recorded;
2. **native-format/kernel baseline**: proves the relevant native Blackwell path but may differ in other tensor
   formats;
3. **fastest practical baseline**: the strongest quality-acceptable configuration an expert user would select,
   with model size and quality differences disclosed.

Do not call a converted model exact parity unless its tensor inventory and execution semantics prove that claim.
Quality and correctness evidence accompanies every promoted performance result.

## Reproducible 16K cross-engine MTP performance comparison

The public three-engine reproduction entry point is:

```bash
systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 \
  ./scripts/benchmark-cross-engine-mtp.sh
```

It runs gem16, pinned patched vLLM 0.26.0, and pinned patched llama.cpp b10240 over the checked-in exact
`benchmarks/prompts/wikipedia-summary-16k.json` token workload. Every engine receives 16,384 prompt tokens, emits
1,135 fixed greedy target tokens with D2 MTP, and uses batch one, three warm-ups, and ten measurements. The script
validates checkpoint locks, competitor versions/patches, GGUF checksums, an idle GPU, and—unless explicitly
overridden—the reference laptop's `max-power` profile plus active `nvidia-powerd` Dynamic Boost. It never overwrites
an existing result and retains per-run JSON, logs, and 200 ms GPU telemetry. The optional 48 GiB systemd scope
contains third-party JIT failure on the no-swap reference host; vLLM startup compilation is limited to four jobs
with one internal NVCC thread each and remains outside inference timing.

The corrected 2026-08-05 final-sprint comparison and limitations are recorded in
[`benchmarks/baselines/cross_engine_mtp/`](../benchmarks/baselines/cross_engine_mtp/):

| Engine | Prefill tok/s | TTFT | Effective D2 tok/s | ITL | Sampled peak VRAM |
|---|---:|---:|---:|---:|---:|
| vLLM 0.26.0 | **6,257.37** | **2,618.35 ms** | 82.25 | 12.158 ms | 15,764 MiB |
| **gem16 `a819d14c`** | 5,866.86 | 2,792.64 ms | **87.66** | **11.408 ms** | 11,746 MiB |
| llama.cpp b10240 | 3,941.23 | 4,157.08 ms | 83.89 | 11.921 ms | 10,630 MiB |

Gem16 D2 is 6.57% faster than vLLM and 4.49% faster than llama.cpp; its ITL is 6.17% and 4.30% lower. Prefill is
6.24% below vLLM and 48.86% above llama.cpp. This is a controlled performance claim for the recorded configurations,
not accepted exact parity: gem16/vLLM use direct mixed FP8/NVFP4 plus FP8 KV, llama.cpp uses patched Q8_0 attention
plus Q8_0 KV, prefill timing boundaries differ, and the three MTP output hashes differ. vLLM also records tuning
OOM fallbacks and an untuned 8K FP4 shape. Only target-verified output tokens are counted. The earlier 89.58 tok/s
gem16 row remains historical because its short-batch verifier omitted two required BF16 boundaries.

## Closed Linux short-context ordinary-decode investigation

The Linux max-power investigation at clean gem16 parent `065b68f` maps the standard `llama-bench` `pp512` and
`tg128` shapes to gem16's disclosed context-512 prefill and context-1/128-token ordinary decode counterparts.
Gem16 runs three warm-ups and ten measurements. Llama.cpp b10240 retains all 13 repetitions after its built-in
warm-up and reports repetitions 4--13.

| Engine | pp512 median tok/s | tg128 median tok/s | Aggregate ms/token |
|---|---:|---:|---:|
| **gem16** | **7,026.84** | 52.52 | 19.039 |
| llama.cpp b10240, F16 KV | 5,381.87 | **62.76** | **15.935** |

The intervals for tg128 do not overlap; llama.cpp leads by 19.48%. This is not exact parity: llama.cpp starts at
depth zero, feeds random tokens and maps source FP8 attention to Q8_0 with F16 KV, while gem16 starts after one
legal prompt token, follows its greedy Target sequence and uses direct checkpoint FP8 attention plus FP8 KV.

Nsight Systems reconciles gem16's context-1 forward to 18.135 ms/token of child kernels, an 18.507 ms graph span
and an 18.516 ms NVTX range. Final argmax plus publication accounts for only about 0.013 ms/token. The admitted
eight-warp direct-projection and exact fused-short-attention candidates both lose in their affected rejection
screens and are removed; no production change or speed claim results. The direct all-regions probe retains 4.04 GiB
CUDA-visible free with Target, assistant, fixed D2, checkpoint-recommended sampling, graphs, media workspace and
the final prompt plan resident.

The complete distributions, context matrix, profile counters, candidate outcomes, memory record, limitations and
raw-evidence path are in
[`linux-short-context-065b68f.json`](../benchmarks/baselines/llama_cpp/linux-short-context-065b68f.json).

## Windows short-context llama-bench characterization

The current Windows short-context characterization maps the standard `llama-bench` `pp512` and `tg128` tests to
`gem16-bench prefill --context 512` and `gem16-bench decode --context 1 --tokens 128`. Llama.cpp runs 13 measured
repetitions plus its built-in warm-up; the first three repetitions are discarded and the final ten are reported.
Gem16 runs three warm-ups and ten measured repetitions. The final runs start at 50/51 C for llama.cpp/gem16 with
zero GPU utilization and zero application VRAM, and both reach a maximum sampled temperature of 66 C.

| Engine | pp512 median tok/s | Prompt time | tg128 median tok/s | Aggregate ms/token | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| gem16 `cc01a05` | **6,877.29** | **74.448 ms** | 52.43 | 19.072 ms | 10,634 MiB |
| llama.cpp b10240 | 5,400.91 | 94.799 ms | **62.08** | **16.109 ms** | **9,824 MiB** |

This is a development characterization rather than a promoted parity comparison. `llama-bench` uses random
synthetic token IDs and feeds a new random token for each generation step; gem16 uses its deterministic prompt
formula and generated Target sequence. Llama.cpp `tg128` starts at position zero, while gem16 starts after the
smallest legal one-token prompt. Llama.cpp uses the patched NVFP4/Q8_0 GGUF with F16 K/V; gem16 directly loads the
mixed FP8/NVFP4 checkpoint with checkpoint-FP8 K/V. Llama.cpp exposes aggregate generation latency, while gem16
also retains the complete per-token distribution. All raw samples, commands, hashes, telemetry, and confidence
intervals are retained in
[`windows-short-context-cc01a05.json`](../benchmarks/baselines/llama_cpp/windows-short-context-cc01a05.json).

The decode command keeps one model instance resident across all runs, clears the preallocated KV cache outside
the timing boundary, performs the configured warm-ups, and retains every measured inter-token latency in JSON.
It reports median/mean throughput, standard deviation, a Student-t 95% confidence interval across runs, and pooled
p50/p95/p99 inter-token latency. Prompt IDs follow the same deterministic formula as the tracked vLLM harness. The
first token selected after prompt ingestion is excluded from decode, after which exactly `--tokens` full forward,
greedy-selection, and device-to-host intervals are timed:

```powershell
$model = python -c "from tools.hf_cache import default_target_model; print(default_target_model())"
.\build\Windows\blackwell-release\bin\gem16-bench.exe decode `
  --model $model `
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
$model = python -c "from tools.hf_cache import default_target_model; print(default_target_model())"
.\build\Windows\blackwell-release\bin\gem16-bench.exe prefill `
  --model $model `
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
`top_p=0.95`), FP8 KV, and fixed D2 rather than adaptive MTP. By default the root
turn loads the checksum-locked repository suite in `benchmarks/media/suite.json`:
three project-generated images and three public-domain LibriVox excerpts in
alternating order. That approximately 2K-token multimodal root is the empty-cache
measurement. Later text-only turns extend the same Responses chain toward 4K,
8K, 32K, 64K, and 128K actual input positions; no prompt-cache reset or
reconstructed root is used.

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
- initial multimodal recognition and final 128K retrieval of every image code,
  object count, and audio phrase.

Every repository media byte is verified against the suite manifest before the
server starts. `--image` and `--audio` may append caller-supplied stress inputs,
but are no longer required for a reproducible default run. The root disables reasoning by default so its visible 384-token budget can cover
all six labeled assets. Final quality is split into six concise resident turns,
one per asset, using the same bounded-reasoning mode as checkpoint probes; this
avoids a summary-length cap masking retrieval of the last media item.

Filler turns are part of the same conversation and now also retain their exact
bulk-prefill time and throughput. They are not mixed into the small-suffix
checkpoint distribution. This distinction avoids presenting a 70-token cached
continuation as large-batch prompt throughput.

Future results must use the timing boundaries, repetition policy, quality gates, and three llama.cpp baseline
labels defined above and any stricter active feature-plan matrix. Throughput speedup and latency reduction must be
reported separately. Exact board identity, power envelope, clocks, and thermals are recorded for every run; the
project scope remains the 16 GB CUDA target class.

Current upstream llama.cpp is pinned, but its unpatched converter rejects the locked checkpoint's mixed FP8/NVFP4
compressed-tensors groups. A tracked converter patch produces a closest-parity candidate that preserves NVFP4 MLP
tensors and maps FP8 attention weights to Q8_0. Its current inventory, full-residency probe, fixed-D2 distributions,
and continuous telemetry are recorded; native-path invocation profiling and quality acceptance remain open gates.
This candidate cannot be labeled exact format parity.

The local `benchmarks/baselines/llama_cpp/characterization.json` preserves the earlier b10210 ordinary prefill/decode
matrix through 65,536/8,192 positions. The current b10240 fixed-D2 result is instead in the cross-engine summary.
Neither is an accepted exact-parity baseline because native dispatch profiling and a current quality threshold remain
open.

The pinned SM120a competitor build is complete. Its NVFP4 object contains native block-scaled
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` instructions. This is a binary capability check only; runtime dispatch must
still be captured with the selected model before a tier-B result is valid.

## Historical 2026-07-27 cross-engine characterization

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

## Historical direct vLLM 0.25.1 development comparison

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

The active exact-Target scheduler executes the BF16 assistant, evaluates the current input plus drafts in one
causal Target batch, and commits only the verified prefix on GPU. Proposed tokens are never counted as output.
Every run reports proposed, accepted, rejected, mean accepted length, Target batches, fallback tokens, draft-length
group counts, effective verified tok/s, and incremental memory.

The current promoted comparison is the 2026-08-03 Linux max-power result described above: fixed 16K D2 reaches
89.58 effective verified tok/s and 11.163 ms ITL over three warm-ups and ten measurements. Gem16 is internally
identical to its ordinary Target sequence. External runtime hashes differ, so the vLLM and llama.cpp rows are
controlled performance references rather than exact speculative-speedup or quality-parity claims.

The historical 64.82 tok/s floor remains in `tools/qualify_mtp.py` and its schema for regression reporting and is
surpassed by the retained result. It is not a pending roadmap gate. Any future promotion must still win against an
adjacent current baseline, retain ordinary/MTP identity inside gem16, and satisfy the permanent benchmark contract.

For development-only rejection screens, use:

```bash
python tools/screen_mtp.py \
  --workload benchmarks/results/<workload>/workload.json \
  --output benchmarks/results/<date>/<git-sha>/<machine>/mtp-short-screen.json \
  --executable build/Linux/blackwell-release/bin/gem16-run
```

The default short screen is not qualification evidence. A promising candidate requires the full alternating run:

```bash
python tools/qualify_mtp.py \
  --workload benchmarks/results/<workload>/workload.json \
  --output benchmarks/results/<date>/<git-sha>/<machine>/mtp-qualification.json \
  --executable build/Linux/blackwell-release/bin/gem16-run \
  --warmup-pairs 3 --measured-pairs 10
```

The qualifier alternates mode order, retains every raw sample, and fails on any ordinary/MTP ID difference. New
results must additionally capture clocks, power, thermals, allocator accounting, graph/kernel dispatch, and the
adjacent competitor configuration. See [MTP.md](MTP.md) for execution and correctness semantics.
