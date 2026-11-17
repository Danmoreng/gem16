# Current vLLM INT4 characterization — 2026-11-15

The current vLLM 0.33.0 completes the 26B A4B 16K/1,135-token Ordinary workload
with a community INT4 W4A16 checkpoint. This is external performance characterization,
not a qualified quality-equivalent or deterministic reference. No NVFP4 Target was used.

| Engine / Target | Prefill tok/s | Ordinary Decode tok/s | D2 Decode tok/s |
|---|---:|---:|---:|
| gem16 Trellis35 W4A8 | 5612.11 | 141.738 | 200.402 |
| central llama.cpp Google QAT Q4_0 | 5358.57 | 144.329 | 202.874 |
| vLLM 0.33.0 INT4 W4A16 | 6475.09 | 146.674 | 149.873 * |

D2 (*): different compiler/memory settings and strongly variable outputs; see the follow-up below.

Other engines were measured earlier the same day; this is not a new interleaved
three-engine campaign. Their full settings, bootstrap intervals, residency and dispatch
are in the [central comparison](central-llama-repeat-2026-11-15.md).

## Workload and environment

RTX 5080 Laptop GPU, firmware max-power, nvidia-powerd active, driver 620.70.06.
vLLM 0.33.0, Torch 2.14.0, Transformers 5.20.2, compressed-tensors 0.17.0;
a separate local environment leaves system packages unchanged. Same exact 16,384 prompt
IDs and 1,135 fixed greedy outputs, seed 0, batch one, prefix cache disabled, FP8 KV,
requested context 17,519, 3 warmups and 10 retained samples. Throughput counts 1,134 timed
output intervals, excluding the first token. Chunked prefill uses 1,024 tokens.
TTFT is 2530.314 ms and includes vLLM's request path;
other engines report their own prompt-processing boundaries.

The immutable Target is
`lokeshe09/gemma-4-26B-A4B-it-INT4-W4A16-channelwise@915f65bdfb1d8cd16232849897e63ba1e05614e4`.
The 14,925,293,356-byte Safetensors file has SHA-256
`9e93613028b2c4331648b8c8d0db9a93168c0bc005b67917f1e92526ec4767b1`.
It uses INT4 channel-wise weights with BF16 activations/exceptions, rather than
Trellis35 W4A8 or Google's QAT GGUF. Its payload is 3.37% larger than the llama.cpp
GGUF and 22.29% larger than the native text Target. Quantization quality has not
been requalified. Existing validation explicitly normalizes channel-wise null
`group_size` to the equivalent -1 value in memory; source files remain unchanged.

## GPU execution and memory

All per-token weights stay on the GPU; CPU weight offload is disabled. Actual
PyTorch-profiler CUDA events show 4,140 Marlin events and 690 attention events
across 24 model iterations (prefill and decode). Instrumented timings are excluded.
The separate trace and parsed families are retained in the raw results.
Whole-device sampled peak VRAM is **15447 MiB**.
The 200 ms telemetry can miss transient allocation peaks.

Automatic memory profiling could not reliably reserve the full context: cached
compilation changed its KV allocation estimate, and some configurations ran out of
memory. The final unmodified engine uses its public `kv_cache_memory_bytes=536870912`
option (512 MiB), `gpu_memory_utilization=0.98`, with the original 1,024-token
prefill chunk restored. The raw launcher injects only that documented engine option
into the unchanged project helper. All failed attempts are retained and enumerated
in the machine-readable record.

## Reproducibility and D2 limitation

**10 of 10 greedy primary outputs have distinct hashes.** Both eager-control outputs
also differ, so CUDA Graph replay alone does not explain this. The eager control
measured 6,454.26 prefill / 54.907 decode tok/s (1 warmup, 2 retained samples).
The primary is a speed characterization, not evidence of deterministic output or
quality equivalence. Within each run all fixed output positions were generated.

The initial D2 probe with this INT4 Target and the official QAT unquantized BF16 Assistant failed during
TorchInductor autotuning with a 1.38 GiB allocation request. The bounded probe preserved
full 17,519 context, used a 128-token prefill chunk and explicit 384 MiB KV budget;
it produced no inference rate. No NVFP4 Assistant substitution was made.

## Successful D2 memory follow-up

The first startup failure is not a general impossibility result. Using the public
`compilation_config={"mode":0,"cudagraph_mode":"FULL"}` option removes
TorchInductor compilation while keeping full CUDA Graph capture and the same INT4
W4A16 Target. No W4A8 or NVFP4 substitution is needed. Requested context17,519,
16,384 prompt IDs,1,135 outputs,20 retained samples,FP8 KV and all weights on GPU
are preserved. D2 uses a128-token prefill chunk and384MiB explicit KV budget,
unlike the original Ordinary1024/512MiB configuration; do not interpret the table
as a controlled MTP-only speedup. A matched1+2 Ordinary control is retained.

The noisy D2 result is repeated in two3+10 cohorts and all20 samples are pooled.
This is a variable-output characterization rather than a stable qualified rate.

D2 median **149.873 tok/s**, range
**123.766–237.519tok/s**,
peak **15521MiB**. Within-mode determinism:
**False**. This remains a characterization,
not a quality-qualified reference. The official base-model BF16 Assistant is
`google/gemma-4-26B-A4B-it-assistant@6e5aaaf4c42b98394530b8fda2e95cadd65c151c`,
SHA256`006c8713b29b046e296c0f0b7c524aa9264ae07ae8b0c3888b9bc062242297ff`.
The model card of the immutable INT4 Target identifies the matching non-QAT
`google/gemma-4-26B-A4B-it` base. Earlier QAT-Assistant runs remain diagnostics,
not the selected pairing. No result is selected by maximum throughput.

Disabling only combo-kernel autotuning reproduced the same1.38GiB allocation
failure. Explicit Humming FP8-activation W4A8 with strict no-fallback selection
fails during model construction (`RowvLLMParameter.__init__` missing `input_dim`),
before inference. The successful configuration retains16-bit activations.
Actual CUDA dispatch is recorded separately from primary timings in the
[follow-up evidence](../../../artifacts/vllm/gemma4-26b-d2-memory-followup-2026-11-15.json).

## Historical context

The same immutable INT4 Target ran on 2026-08-23 with vLLM 0.27.1 at **6,475.80
prefill / 149.348 decode tok/s**. That older test used 16K + **64** output tokens,
1 warmup and 3 retained runs. Today uses **1,135** outputs and 3 + 10. Both are
nondeterministic. The near-identical prefill result is useful context; the decode
rates do not establish a version-only regression because sequence lengths, driver,
Torch, allocation budget and runtime differ.

The exact Google QAT Q4_0 GGUF used by llama.cpp still fails loading: released
GGUF plugin 0.0.5 reports 416 unmapped parameter names. Current plugin source
`497542d12327a67f806345438238ea702cadd824` builds but fails import against vLLM
0.33.0 due to the removed `vllm.transformers_utils.configs.qwen3_5` module.
There is consequently no same-GGUF vLLM throughput claim.

## Evidence

[Public record](../../../artifacts/vllm/gemma4-26b-q4-repeat-2026-11-15.json)
contains every primary/eager sample, checkpoint identities, versions, commands,
telemetry, actual dispatch and a hashed manifest. Raw attempts and trace:
`benchmarks/results/2026-11-15/58febf4/vllm-q4-repeat/`.
The current local native binary is byte-identical to its November 5 qualified build;
remote main's two additional commits do not change runtime sources.
