# Trellis35 versus current llama.cpp, 2026-10-10

Ordinary decode completed for both engines. llama.cpp D2 also completed; gem16
Trellis35 D2 crashed before inference while instantiating its fixed-MTP CUDA
graph. The D2 cross-engine comparison is therefore incomplete. These are Linux
text-only performance characterizations, not model-quality parity or a new
product qualification.

## Results

Medians from three warmups and ten retained measurements per successful case:

| Engine / mode | Prefill tok/s | TTFT ms | Decode tok/s | Aggregate ITL ms | Sampled peak VRAM MiB | Free MiB |
|---|---:|---:|---:|---:|---:|---:|
| gem16 Trellis35 ordinary | 5,623.23 | 2,913.63 | 128.955 | 7.755 | 12,455 | 3,848 |
| llama.cpp b11552 QAT Q4_0 ordinary | 4,607.79 | 3,555.72 | 138.461 | 7.222 | 15,201 | 1,102 |
| gem16 Trellis35 fixed D2 | — | — | startup SIGSEGV | — | — | — |
| llama.cpp b11552 QAT Q4_0 D2 | 4,445.20 | 3,685.77 | 196.850 | 5.080 | 15,561 | 742 |

Trellis35 ordinary prefill is 22.04% faster; llama.cpp ordinary decode is 7.37%
faster. llama.cpp D2 is 42.17% faster than its ordinary row, with 360 MiB more
sampled peak VRAM. This D2 ratio is descriptive: its token vector differs from
ordinary starting at zero-based output index 101. Every successful case is
internally deterministic across its ten measurements. Cross-engine ordinary
outputs also differ, so none of these ratios establish equivalent model quality.

Decode sample means and 95% Student-t confidence intervals of the mean are
128.947 [128.905, 128.989], 138.586 [138.397, 138.774], and
197.021 [196.720, 197.321] tok/s, respectively. These intervals describe repeated
runs of this fixture and environment, not a general workload distribution.

The complete samples, configurations, output hashes, provenance, telemetry
summaries, commands and evidence hashes are in the new
[machine-readable record](../../../benchmarks/baselines/llama_cpp/gemma4-26b-trellis35-q4_0-refresh-2026-10-10.json).
Historical records were retained without replacing their numbers. Differences
from old rows cannot be attributed solely to llama.cpp changes: toolchain,
driver and explicit forced-output end-token masking also changed.

## Revisions, build and models

- gem16: `c01789a2b46520df8a699ee3338cb19aaccebae2`, rebuilt from current
  `main`; engine source unchanged. The benchmark harness has the recorded local
  changes described below.
- llama.cpp: upstream `master` resolved at the start of this campaign to
  [`23b0202a189c44a54625aadcb37a946dd1d6278d`](https://github.com/ggml-org/llama.cpp/commit/23b0202a189c44a54625aadcb37a946dd1d6278d),
  **0.6.0-dev / build 11552**. Unmodified source, separate
  `build/Linux/llama_cpp/current-release` directory. No old llama.cpp binary was
  executed in this campaign.
- NVIDIA RTX 5080 Laptop GPU, 16,303 MiB, driver 615.71.09, CUDA 13.4.92,
  GCC 16.2.1, SM120a Release builds. Firmware `max-power`, active
  `nvidia-powerd`; each case started at no more than 50 C with no unrelated CUDA
  process. Clocks were firmware-managed rather than locked. The telemetry
  includes observed peaks up to 179.55 W for gem16, 175.94 W for llama ordinary
  and 176.15 W for llama D2.
- Trellis35 target and native gem16 assistant: repository
  `danmoreng/gemma-4-26B-A4B-it-GEM16`, immutable revision
  `6de2a057f11332420819f8e6efd08e42d7a03bc7`, qualified target/assistant lock
  files verified before running. Trellis35 payload: 12,204,692,480 bytes,
  SHA-256 `552ace4b3f2e8e20bbc03a9d4b30887bdb6297d6da9f48d54b4a2b4e9fc803c4`.
- llama target: official Google QAT Q4_0,
  `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` revision
  `d1c082be9cf3c8a514acf63b8761f4b41935842e`, file
  `gemma-4-26B_q4_0-it.gguf`, 14,439,363,584 bytes,
  SHA-256 `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`.
  This target was already present; no new target download was required.
- llama assistant: `unsloth/gemma-4-26B-A4B-it-qat-GGUF` revision
  `7b92b5b28818151e8669af2e45e88d6086f490dd`, file
  `mtp-gemma-4-26B-A4B-it.gguf`, 251,939,328 bytes,
  SHA-256 `7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a`.
  This missing assistant was downloaded into the shared Hugging Face cache and
  verified. Both GGUF hashes were checked before the campaign.

The actual llama build used the following command; upstream maps the deprecated
`GGML_CUDA_FA_ALL_QUANTS=ON` to `GGML_CUDA_FA_QUANTS=all`. The updated repository
build script now uses the latter spelling directly.

```bash
cmake -S third_party/cache/llama.cpp -B build/Linux/llama_cpp/current-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build/Linux/llama_cpp/current-release --parallel 4 \
  --target llama-server llama-bench llama-cli
cmake --build build/Linux/blackwell-release --parallel 4 --target gem16-run
```

Both builds completed successfully. The active source pin and Linux benchmark
defaults now point to this current build rather than the old `release` directory.

## Common protocol and disclosed differences

All three successful cases use the exact `wikipedia-summary-16k.json` prompt:
16,384 token IDs, prompt hash
`d07ad4d805944f0b87869da0c5bb44d99e8c43c0eb57d05a108ad80a6abb51a8`,
greedy decoding, batch one, 1,135 forced output tokens, three warmups, ten measured
samples and a cold prompt. Decode throughput counts the same 1,134 intervals
after the first output token. No speculative proposals are counted as output.
The requested context capacity is 17,519 in both engines; llama.cpp rounds its
allocated capacity to 17,664. Actual prompt and output positions stay identical.

Both receive suppressed token IDs `[1, 50, 106, 258882, 258883]` and no stopping
token list. This explicit common mask matters: llama.cpp `ignore_eos` masks its
end-of-generation IDs, while removing gem16's stop list alone would still let
gem16 sample those IDs. The harness now masks the fixture's end tokens for every
engine in fixed-output mode. All measured runs returned exactly 1,135 tokens.

llama.cpp uses Flash Attention, Q8_0 K/V, eight threads, one slot, batch/ubatch
2048/512, all layers on CUDA0, and explicit `token_embd.weight=CUDA0` for both
target and assistant. Logs confirm 31/31 target and 5/5 assistant layer groups,
14,332.92/225.21 MiB CUDA model buffers and no retained CPU model-weight buffer.
Prompt cache RAM and context checkpoints are disabled. Every raw server response
reports `timings.cache_n=0` and the full prompt count. In current llama.cpp,
top-level `tokens_cached` is the final slot size; it is not evidence of prefix
reuse. The harness now retains and checks these fields separately.

gem16 uses its native Trellis35 W4A8 target with checkpoint FP8 K/V; its assistant
is the pinned mixed NVFP4/FP8/BF16 profile. These formats differ from llama's
Q4_0 target, Q4 assistant and Q8_0 K/V. No quantization, kernel or precision
fallback was introduced to make the configurations appear identical. All ten
gem16 ordinary results report zero fallbacks and no token-loop allocations.
Vision weights are not loaded in this text engine comparison.

gem16 creates a fresh process for each warmup/sample and excludes model loading
from inference timings. llama.cpp keeps one server resident per mode but
recomputes the entire prompt for each request. Model loading and initial plan
creation are excluded; prompt/TTFT engine and server orchestration boundaries
are disclosed rather than assumed identical. ITL is aggregate, not a token-latency
percentile. VRAM is device-wide sampling every 200 ms, including warmup and
measured activity, rather than an instrumented allocation maximum.

llama D2 uses `draft-mtp`, maximum two drafts, minimum one, `p_min=0` and backend
draft sampling. Each measurement proposed 984 tokens, accepted 642 and rejected
342 (65.24% acceptance). The harness-derived group count is 493 including the
initial output; the server reports 492 speculative verification calls. These
are different counters and must not be presented as identical measured target
forward counts.

## Trellis35 D2 startup failure

The full D2 case returned SIGSEGV during its first warmup, before producing
inference JSON or any retained measurement. A core backtrace locates the crash
inside `libcuda.so.1`, through `cuGraphInstantiateWithFlags` and
`cudaGraphInstantiate`, called by
`Gemma4Moe26BReferenceEngine::Impl::PrepareFixedMtpChainGraph`. The source builds
the fixed-D2 conditional graph in
`src/cuda/engine/detail/gemma4_26b_mtp_graphs.inc`.

Two bounded reproductions with eight output tokens, the same 16K prompt and
17,519 context capacity also returned SIGSEGV: default CUDA module loading and
`CUDA_MODULE_LOADING=EAGER`. Thus there are three observed startup failures on
this rebuilt engine/toolkit/driver combination. The backtrace identifies the
failure boundary, not whether the root cause is an engine graph issue or an
upstream driver/runtime defect. Only CUDA 13.4 is installed here.

No graph bypass, ordinary substitute, alternative precision or assistant was
used to manufacture a D2 number. Historical Trellis35 D2 measurements remain
historical and do not resolve this current failure. Completing the requested D2
pair requires isolating and fixing this graph-instantiation failure, then
rerunning Trellis35 D2 under the same protocol.

## Raw evidence, dispatch and checks

The retained local directory is
`benchmarks/results/2026-10-10/c01789a2/rtx5080-laptop-llama-current-trellis35-16k-final/`.
It contains `provenance.json`, the three successful engine JSON files with all
raw samples/responses, four exact `*.command.json` files, server logs, continuous
telemetry, build/verification/test logs, `run.py`, `summarize.py`, the crash core,
backtrace and `gem16-d2-probes.json`. The campaign ran ordinary first, then D2;
llama D2 was resumed separately after gem16 D2 aborted the campaign runner.
Run commands are recorded explicitly; reruns require a new output directory.

The earlier directories with suffixes `-16k` and `-16k-verified` retain aborted
setup attempts and are not the source of the numerical table. Separate eight-token
Nsight captures provide dispatch evidence and are excluded from performance
statistics. gem16's earlier decode-only capture used the same context capacity
but the original two-ID suppression mask; its eight output IDs match the final
run prefix exactly. It records the W4A8 projection and FP8 attention kernels.
Current llama ordinary/D2 captures used the same prompt and common mask with a
reduced requested capacity of 16,392 and cover the full request, including
prefill; they record `mul_mat_q`, Flash Attention and Q8_0 cache-write kernels.
Kernel-family dispatch was observed; no new instruction-level efficiency or
numerical-equivalence claim is made.

```bash
python -m unittest tests.python.test_benchmark_wikipedia_workload \
  tests.python.test_benchmark_gemma4_26b_wp8_text
bash -n benchmarks/baselines/llama_cpp/build.sh \
  scripts/benchmark-cross-engine-mtp.sh scripts/run-gemma4-26b-gsm8k-smoke.sh
git diff --check
```

The 18 host tests pass. They cover fixed-output masking, current/legacy cache
response semantics, rejection of short outputs and partial prompts, and retained
response evidence. Shell syntax and whitespace checks pass. No engine source,
shared CUDA implementation or numerical tolerance changed; no broad GPU quality
suite or Windows benchmark was run. No commit or push was made.
