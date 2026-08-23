# llama.cpp baseline

The current build pin is upstream build 10593, commit
`b0539c43ed13b16bf0d8a0840646faea65469702`, fetched from `master` on 2026-08-23. The nearest preceding release
tag is b10590. Historical rows below retain the exact older llama.cpp revision that produced them, including the
b10240 Linux/Windows cross-engine characterizations and the b10364 batch audit; updating the active build pin does
not relabel those results.

## Current-build Gemma 4 26B ordinary decode refresh

Build 10593 was measured against Google's immutable official QAT Q4_0 GGUF with the fixed 16,384-token Wikipedia
workload and 64 forced greedy output tokens. This is ordinary decode: speculative types and MTP are disabled. The
model uses Q8_0 K/V, Flash Attention, batch/ubatch 2048/512, one slot, eight threads, all layer groups plus the tied
embedding on CUDA0, one warm-up, and three measured repetitions.

The three decode samples are 120.059, 119.494, and 119.451 tok/s; the median is **119.494 tok/s** with 8.369 ms
average inter-token latency. Median prompt throughput is 4,366.58 tok/s. All runs emit the same 64-token sequence
with SHA-256 `9c353bfba7fde6722c86fef93d7bfdb9b30de3757b82a7493c83ee0c8c691e42`. This short refresh is a
development characterization, not the formal M20 3-warm-up/10-retained-run promotion. Exact samples, revisions,
configuration and limitations are retained in
[`gemma4-26b-a4b-qat-q4_0-current-ordinary.json`](gemma4-26b-a4b-qat-q4_0-current-ordinary.json).

## Current-build Gemma 4 12B batch/ubatch audit

The current build 10364 was screened on Windows with the same 16,384-token Wikipedia prompt and 12B patched
NVFP4/Q8_0 GGUF used by the retained Windows b10240 comparison. Every new execution in this audit used build 10364;
b10240 appears only as previously recorded historical evidence. Short 1-warm-up/3-measurement screens forced one
output token and tested `batch/ubatch` pairs `1024/512`, `1024/1024`, `2048/256`, `2048/512`, `2048/1024`, and
`4096/512`. The existing `2048/512` pair remained fastest at 4,047.92 prompt tok/s in that shortlist. In particular,
`1024/512` reached 4,004.78, `1024/1024` 3,948.29, and `2048/256` 3,748.06 tok/s.

A follow-up deliberately spent more workspace VRAM with `4096/2048`, `8192/2048`, `8192/4096`, and `8192/8192`.
Their short-screen medians and sampled peaks were 3,812.09/11,496 MiB, 3,824.13/11,496 MiB,
3,465.92/12,774 MiB, and 2,940.31 tok/s/15,430 MiB respectively. Raising only `n_batch` from 4,096 to 8,192 at
`n_ubatch=2048` did not change sampled peak VRAM materially. Raising `n_ubatch` allocated substantially larger
compute/KV buffers and made prefill progressively slower.

The selected fully resident configuration then ran the complete fixed 1,135-token D2 workload with three warm-ups
and ten measured repetitions. Target and assistant `token_embd.weight` were forced to CUDA0; verbose logs show 49/49
target and 5/5 assistant layer groups, 10,837.74/806.57 MiB CUDA model buffers, and no CPU-mapped model-weight
buffer. Q8_0 target K/V, Flash Attention, prompt-cache-off, one slot, eight threads, and all workload semantics were
retained.

| llama.cpp configuration | Prefill tok/s | TTFT | D2 tok/s | ITL | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| b10240 historical Windows, `2048/512` | 3,942.08 | 4,156.18 ms | 86.798 | 11.521 ms | 10,599 MiB |
| b10364 default embedding placement, `2048/512` | 3,989.25 | 4,107.03 ms | 87.920 | 11.374 ms | 10,586 MiB |
| b10364 VRAM-matched diagnostic, `8192/2048` | 3,804.74 | 4,306.21 ms | 82.679 | 12.095 ms | 11,512 MiB |
| **b10364 resident, `2048/512`** | **3,996.31** | **4,099.78 ms** | **88.134** | **11.346 ms** | 12,500 MiB |

The current resident result is 1.38% faster in prefill and lowers TTFT by 1.36% against the existing same-machine
b10240 row. Full residency changes current-build prefill by only +0.18% versus default placement. Batch tuning and
the upstream refresh therefore do not explain the large retained prefill gap; `2048/512` was already the right
published pair for this workload. This Windows audit does not replace the Linux cross-engine table because gem16
and vLLM were not rerun here. Full samples, hashes, telemetry, residency, raw paths, and limitations are in
[`windows-12b-batch-sweep-b10364.json`](windows-12b-batch-sweep-b10364.json).

The full `8192/2048` control was selected because its 11,512 MiB sampled peak is only 201 MiB (1.72%) below the
existing same-machine gem16 peak of 11,713 MiB. Despite using 926 MiB more than llama.cpp's default `2048/512`
control, it is 4.63% slower in prefill and raises TTFT by 4.85%. This is a diagnostic VRAM match: default llama.cpp
placement still retains 1,920 MiB of target and 512 MiB of assistant weights CPU-mapped. Conversely, the valid
fully resident `2048/512` result already uses 12,500 MiB, 787 MiB more than the historical gem16 peak, while still
remaining the best llama.cpp result. The large-batch output is internally deterministic but diverges from the
`2048/512` output at zero-based index 76, so the decode/ITL figures in that row are descriptive rather than an
exact sequence-matched comparison. Prefill remains directly comparable because all runs use the identical fixed
16,384-token prompt.

## Gemma 4 26B A4B Unsloth QAT UD-Q4_K_XL baseline

The practical 26B baseline uses
[`unsloth/gemma-4-26B-A4B-it-qat-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/tree/7b92b5b28818151e8669af2e45e88d6086f490dd)
at immutable revision `7b92b5b28818151e8669af2e45e88d6086f490dd`. The target GGUF is
`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`, 14,249,047,104 bytes, SHA-256
`a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`. The separate assistant is
`mtp-gemma-4-26B-A4B-it.gguf`, 251,939,328 bytes, SHA-256
`7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a`.

Both modes ran batch one on Windows with the same exact 16,384-token Wikipedia prompt, 1,135 forced greedy output
positions, Q8_0 K/V, Flash Attention, three warm-ups and ten measured repetitions. `token_embd.weight` was forced
to CUDA0 for both target and draft models. Verbose runtime evidence confirms 31/31 target and 5/5 assistant layer
groups on CUDA0, no retained CPU-mapped model buffer in the promoted MTP run, and 200 ms external telemetry.

| Mode | Prefill tok/s | TTFT | Decode tok/s | ITL | Peak VRAM | Free margin |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | **4,333.85** | **3,780.47 ms** | 132.683 | 7.537 ms | **14,742 MiB** | **1,561 MiB** |
| MTP D2 | 4,275.57 | 3,832.01 ms | **164.303** | **6.086 ms** | 15,108 MiB | 1,195 MiB |

MTP improves median effective decode throughput by 23.83% and lowers aggregate ITL by 19.25%, while adding 366 MiB
of sampled peak VRAM. It proposes 997 tokens, accepts 635 and rejects 362 across 500 target groups in every run.
Both modes are internally deterministic, but MTP first differs from ordinary at zero-based output index 55. This
is therefore a controlled performance characterization, not ordinary/MTP exactness or quality acceptance.

The current llama.cpp path is also not fully GPU-resident MTP. Target and assistant forwards execute through CUDA,
and draft Top-K reports backend sampling with no CPU-sampler fallback. However, the scheduler retains hidden rows in
host `std::vector<float>` storage, copies them with `std::memcpy`, constructs draft batches on the host, and performs
verification/acceptance in host loops. Describe it as **host-controlled MTP with CUDA-offloaded forwards**, not as
a CUDA-resident MTP control path. Full samples, residency facts, telemetry summaries, output hashes and limitations
are retained in
[`gemma4-26b-a4b-qat-ud-q4_k_xl-mtp-characterization.json`](gemma4-26b-a4b-qat-ud-q4_k_xl-mtp-characterization.json).

## Gemma 4 26B A4B QAT exploration

The official `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` at revision
`d1c082be9cf3c8a514acf63b8761f4b41935842e` has been checksum-verified and characterized on the reference GPU.
The 14,439,363,584-byte GGUF has SHA-256
`3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`. llama.cpp identifies 25.23B total
parameters, 128 experts with 8 active, and offloads all 31 layer groups. The benchmark additionally forces the
tied `token_embd.weight` to CUDA0; without that override llama.cpp retains a 577.50 MiB CPU-mapped copy.

With Q8_0 K/V, Flash Attention, batch one, Max Power, three discarded conditioning repetitions, and ten measured
runs, median prompt throughput is 2,334.86/5,167.55/5,025.43/4,746.58 tok/s at 128/512/2,048/8,192 tokens. Median
256-token decode throughput is 174.564/167.057/158.267 tok/s at existing contexts 128/2,048/8,192. Peak sampled
VRAM is 15,442 MiB, leaving only about 438 MiB against llama.cpp's 15,880 MiB CUDA report and therefore missing
gem16's 700 MiB safety margin. Context creation plus one token succeeds through 64K, but the 64K process reaches
15,800 MiB and is not evidence that full prefill has safe workspace.

This is an exploratory performance and fit characterization, not a quality-accepted baseline. Q4_0 kernel
dispatch has not been profiled, no quality comparison has been run, and `llama-bench` does not expose per-token
latency percentiles. Reproducible commands, ten-sample statistics, telemetry summaries, residency details, and
limitations are retained in
[`gemma4-26b-a4b-qat-q4_0-characterization.json`](gemma4-26b-a4b-qat-q4_0-characterization.json).

## Gemma 4 MTP capability

The current runtime implements the dedicated `gemma4-assistant` architecture and `draft-mtp` scheduler. The
regenerated official BF16 assistant is an 861,520,128-byte GGUF with SHA-256
`b3ab76db11dd1cfbef51925d7dfd6e234325aa86ab3edd7dea42994edb093b65`. Linux verbose logs prove all 49 target and all
5 assistant layer groups are on CUDA0; assistant layers 0/1/2 share target Layer-46 K/V and layer 3 shares
Layer-47 K/V.

The current fixed-1,135-token max-power Linux characterization reaches 3,922.61 prefill tok/s, 4,176.81 ms TTFT,
82.881 effective D2 tok/s, and 12.065 ms ITL after three warm-ups and ten measured runs. D2 proposes 1,035,
accepts 616, rejects 419, and completes 519 verification groups in every run. These are not exact format parity
results: source FP8 attention weights are Q8_0 in the patched GGUF and K/V is Q8_0. The speculative output is not
token-exact against llama.cpp ordinary decode. This remains a characterization, not an accepted performance
baseline. Full distributions and telemetry are in `../cross_engine_mtp/characterization.json`.

A historical fixed-1,135-token Wikipedia screen at the prior b10210 build evaluated llama.cpp's ordered
`ngram-mod,draft-mtp` cascade. `ngram-mod` is a higher-priority proposer, not a token-level merge: MTP runs only
when N-Gram returns no draft. Match lengths 8, 12, 16, and 24 returned no N-Gram proposals on this output and ran
at ordinary-decode speed when used alone. Aggressive match length 2 produced drafts, but N-Gram-only mean accepted
length was only about 0.12–0.13; active N-Gram/MTP screens reached 45.86–48.96 tok/s versus 50.01 tok/s for the
MTP-D2-only screen. Match 3/4 cascades likewise remained below MTP-only. Outputs changed when aggressive N-Gram
caused larger target batches, so these are external performance characterizations rather than gem16 exactness
evidence. The interrupted repeated comparison is not reported as a qualification.

`tools/benchmark_wikipedia_workload.py` exposes this matrix through `--llama-spec-types`,
`--llama-ngram-mod-n-match`, `--llama-ngram-mod-n-min`, and `--llama-ngram-mod-n-max`. It records generic aggregate
speculative counters because llama.cpp's completion response does not identify the selected proposer for each
group; source-specific server statistics require separate log instrumentation. Generated raw screens remain under
ignored `benchmarks/results/`.

## Same-source gate

The exact pinned Hugging Face checkpoint cannot currently pass upstream's converter. The converter recognizes
`Gemma4UnifiedForConditionalGeneration`, indexes the Safetensors file, and then rejects its two compressed-tensors
configuration groups:

```text
NotImplementedError: Can't handle multiple config groups for compressed-tensors yet
```

This is expected from the source: its mixed-precision shortcut only accepts multiple groups when every group is
`nvfp4-pack-quantized`; this checkpoint combines FP8 attention and NVFP4 MLP groups. See
`conversion-probe.json` for the exact command and result. Upstream tier A therefore remains blocked.

An auditable local patch now enables a separately labeled **same-source closest-parity patched** candidate. It
preserves and repacks the 144 NVFP4 MLP tensors and stores the 184 source FP8 attention weights as Q8_0 using the
current converter's `--fp8-as-q8` path. This is not exact format parity. The regenerated GGUF has 955 tensors,
size 9,366,658,208 bytes, and SHA-256
`6f90177f6a2d42406d57cfa764eae890b262bfdf71d353bd4827e0488b099896`; see
[`tensor-inventory.json`](tensor-inventory.json).

Reproduce the converter gate after preparing its documented Python requirements:

```bash
benchmarks/baselines/llama_cpp/convert.sh \
  models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  build/llama_cpp/gemma4-12b-nvfp4.gguf \
  --dry-run
```

On Windows, use the equivalent PowerShell entry point (and a venv under `Scripts\python.exe`):

```powershell
.\benchmarks\baselines\llama_cpp\convert.ps1 `
  .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  .\build\Windows\llama_cpp\gemma4-12b-nvfp4.gguf `
  --dry-run
```

Prepare and run the patched converter in a separate ignored worktree:

```bash
benchmarks/baselines/llama_cpp/prepare-patched-source.sh
benchmarks/baselines/llama_cpp/convert-patched.sh \
  models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  build/llama_cpp/gemma4-12b-nvfp4-patched.gguf
```

The patched PowerShell flow is likewise available as `prepare-patched-source.ps1` and `convert-patched.ps1`.

The checkpoint tokenizer metadata requires Transformers 5 for this probe; Transformers 4.57.6 from upstream's
legacy converter requirements fails while reading `extra_special_tokens`. The default patched command therefore
uses the pinned vLLM 0.26.0 environment with Transformers 5.14.1. This is converter tooling only, not a runtime
dependency.

`build.sh` and `build.ps1` check out the same exact clean commit and build CUDA tools specifically for SM120a, with
all Flash-Attention KV quantizations enabled. The PowerShell helper imports MSVC automatically and keeps its cache
under `build/Windows/llama_cpp`. The selected GGUF has passed structural inspection, full GPU-residency probing,
and the Linux 1/1 plus current max-power 3/10 runtime characterization; profiler-level native-instruction
invocation attribution and quality acceptance remain open.

The current build's dedicated `mmq-instance-nvfp4.cu.o` is an `sm_120a` cubin. `cuobjdump` confirms 1,792
occurrences of `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`, matching the block-scaled E2M1/UE4M3 native path. This
proves instruction availability in the binary. The candidate loads 49/49 layers on CUDA0. External telemetry samples a 10,631 MiB peak during the current
17,519-position D2 run. A profiler-level trace of the native kernel invocation remains an explicit gate.

The historical direct-runtime comparison in `quality.json` used identical chat rendering, greedy sampling, and the same three
prompts. It records 50/65 token agreement and one exact generation. The short sky answer diverges after 18 matching
tokens; the thinking trace matches 28/32 tokens. These are measurements, not an adopted tolerance. Timing results
remain characterization-only until a quality threshold and native dispatch trace are approved.

The local `characterization.json` preserves the earlier b10210 matrix and summarizes ten measured runs after three
conditioning runs for the required prefill points
through 65,536 tokens and decode points at context depths 128, 2,048, and 8,192. It retains every throughput sample,
reports the median as primary, and includes mean, sample standard deviation, range, and a 95% Student-t confidence
interval. The raw inputs remain under `benchmarks/results/`. This run did not capture profiler dispatch,
per-token latency distributions, or power/clock/thermal time series, so it is development evidence only.

The three required tiers remain separate:

1. same-source closest parity (upstream blocked; patched candidate under characterization);
2. native-NVFP4 llama.cpp (model selection and quality gate pending);
3. fastest practical quality-acceptable llama.cpp (model selection and quality gate pending).
