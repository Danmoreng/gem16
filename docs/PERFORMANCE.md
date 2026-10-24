# Recorded performance

These are retained measurements at the revisions named below, not a new benchmark of current HEAD.
The public profiles and qualification limits are described in the [README](../README.md).

<!-- T3-GATE-RDC-AUDIT-2026-10-24 -->
## 2026-10-24 T3 GateUp production-build audit

The three existing variants also fail their all-case gate with production-matched
RDC=true; no production change. All24 actual outputs per variant and2,097,152
codec lanes are exact. These are operator diagnostics, not full-model gain/loss.

| Existing variant | Minimum | Median | Maximum throughput change |
|---|---:|---:|---:|
| Depth 2 | -0.764% | -0.155% | +0.659% |
| Depth 4 | -4.170% | +8.592% | +21.285% |
| Depth 2, register bounded | -6.809% | -5.213% | -4.358% |

[Matched-build evidence](evidence/gemma4_26b/t3-gate-rdc-audit-2026-10-24.md).
<!-- /T3-GATE-RDC-AUDIT-2026-10-24 -->

<!-- NVFP4-HEADS-2026-10-24 -->
## 2026-10-24 NVFP4 output heads

**Rejected after full qualification and end-to-end comparison.** The predeclared reproducible end-to-end gate is not met. The production Target trial is removed and the accepted parent runtime/source is restored; the Assistant variant never entered production. Correct operator results alone do not justify keeping this change.

| Mode | Parent tok/s | Candidate tok/s | Change | Bootstrap 95% |
|---|---:|---:|---:|---|
| Fixed-D2 AB | 195.967 | 196.290 | +0.1647% | [+0.0218%, +0.3158%] |
| Fixed-D2 BA | 195.881 | 196.220 | +0.1728% | [-0.0018%, +0.3470%] |
| Ordinary adjacent | 139.027 | 139.169 | +0.1024% | [+0.0604%, +0.2185%] |

Same fixed 16K / 1,135 / context 17,519 / FP8 KV / greedy workload, three warmups + ten retained
per cohort. All 60 vectors and unchanged D2 acceptance/arenas are exact. Target
row packing gains only 0.18–0.21% on repeated operators; Assistant pairing loses
all six cases and is never promoted. Native scale/mask/tie/overflow/memory,
protected profiles and actual dispatch/instruction checks pass. See
[head report](evidence/gemma4_26b/nvfp4-output-heads-2026-10-24.md) and
[record](../artifacts/trellis35/nvfp4-output-heads-2026-10-24.json).
<!-- /NVFP4-HEADS-2026-10-24 -->

## 2026-10-24 FP8 QKV/O row packing

Frozen `85fc453` versus qualified three-row native FP8 projections, same 16K
prompt / 1,135 outputs / context 17,519 / FP8 KV / greedy, 3 warmups + 10 retained per cohort:

| Mode / comparison | Parent tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|
| Fixed-D2 AB | 195.731 | 196.555 | +0.4212% |
| Fixed-D2 BA | 195.354 | 196.492 | +0.5823% |
| Ordinary adjacent | 139.133 | 139.085 | -0.0342% |

This is a small same-policy end-to-end gain. Whole-process GPU clocks differ;
there is no fixed-clock claim. All 60 vectors, D2 groups/acceptance, arenas and
precision stay exact; sampled peaks 12,455 / 12,725 MiB, process monitoring quiet.
The M1 two-entry pipeline loses all sampled cases and remains diagnostic only.
Actual dispatch/shaders, diverse numerical/sanitizer, full-memory, protected
profiles, proposals and forced continuations pass. Isolated cache-warm gains
are much larger than the whole-model effect; complete traces show QKV nearly
unchanged and O about 2.6% shorter. See the
[qualification report](evidence/gemma4_26b/fp8-projections-2026-10-24.md) and
[acceptance record](../artifacts/trellis35/fp8-projections-2026-10-24.json).
No fresh llama or Windows/media/release claim; all earlier rows remain below.

## 2026-10-24 fixed-D2 warp normalization

Frozen `c48124d` runtime versus qualified R3 local/global FP8 normalization,
identical 16K prompt / 1,135 outputs /context 17,519 / FP8 KV/greedy settings,3+10 per cohort:

| Mode / comparison | Parent decode tok/s | Candidate decode tok/s | Change |
|---|---:|---:|---:|
| Fixed-D2 AB | 193.093 | 195.666 | +1.3324% |
| Fixed-D2 BA | 195.090 | 197.774 | +1.3760% |
| Ordinary adjacent | 139.211 | 139.219 | +0.0056% |

All 80 vectors are exact; D2 groups/acceptance, all arenas and native precision
remain unchanged. Sampled peaks 12,455/13,129MiB. Operator/tail,
memcheck/synccheck, final exact-binary full-model/protected-profile, actual
shader/dispatch, draft and forced-continuation checks pass. Legacy local R2/R5
keeps its arithmetic with a shared maximum-read barrier; details and the original
under-instrumentation countercase are in the
[qualification report](evidence/gemma4_26b/d2-attention-warp-normalization-2026-10-24.md)
and [acceptance record](../artifacts/trellis35/d2-attention-warp-normalization-qualified-2026-10-24.json).
No fresh llama comparison or Windows/media/release claim; prior rows stay below.

## 2026-10-24 bounded T3 GateUp pipeline rejection

Depth2/depth4 resident-payload pipelines and one counter-justified register-bounded
variant are bit exact but fail the predeclared per-case operator gate. Positive
median microbenchmarks do not establish a whole-model gain; no full-model candidate
benchmark or production change is claimed. The
[negative evidence](evidence/gemma4_26b/t3-gate-pipeline-rejected-2026-10-24.md)
and [record](../artifacts/trellis35/t3-gate-pipeline-rejected-2026-10-24.json)
preserve all variants and countercases. The qualified M1 rates below remain.

## 2026-10-24 M1 GateUp codec windows

Frozen `26c6840` parent versus the qualified native/vector M1 GateUp candidate,
same RTX 5080 Laptop/max-power/Dynamic Boost, 16,384 prompt IDs, 1,135 outputs,
context 17,519, batch one, greedy, FP8 KV, three warmups and ten retained runs:

| Mode / comparison | Parent decode tok/s | Candidate decode tok/s | Change |
|---|---:|---:|---:|
| Ordinary AB | 136.481 | 139.031 | +1.8682% |
| Ordinary BA | 136.303 | 138.992 | +1.9723% |
| D2 adjacent pair | 192.600 | 193.025 | +0.2205% |

All 60 retained output vectors match exactly. D2 acceptance and all arenas stay
unchanged; no fallback/repack/token-loop allocation. Numerical, sanitizer,
protected-profile, actual-dispatch and exact draft/forced-case checks pass. D2
is an adjacent non-regression pair, not a speed claim. See the
[qualification report](evidence/gemma4_26b/m1-gate-window-2026-10-24.md) and
[acceptance record](../artifacts/trellis35/m1-gate-window-2026-10-24.json).
Fresh same-day controls are used rather than yesterday's rates. No new llama
comparison or Windows/media/release qualification is claimed; historical rows remain.

## 2026-10-23 T3 Down codec/payload pipeline

Frozen GateUp parent `629bbac` versus the separately qualified native/vector Down
candidate on the same Linux RTX 5080 Laptop GPU, max-power with Dynamic Boost,
16,384 prompt IDs, 1,135 outputs, context 17,519, batch one, greedy and FP8 KV.
Each cohort uses three warmups plus ten retained runs:

| Mode / comparison | Parent decode tok/s | Candidate decode tok/s | Change | Sampled peak VRAM, both |
|---|---:|---:|---:|---:|
| Fixed-D2 AB | 193.418 | 194.854 | +0.7426% | 12,725 MiB |
| Fixed-D2 BA | 193.499 | 194.716 | +0.6286% | 12,725 MiB |
| Ordinary adjacent pair | 137.672 | 137.659 | -0.0097% | 12,455 MiB |

All 60 retained runs reproduce the exact 1,135 IDs; all 40 D2 runs retain 502
groups and 632/1,004 accepted proposals, identical arenas and no fallback,
runtime repack or token-loop allocation. Full
numerical, sanitizer, protected-profile, actual-dispatch, proposal and forced-case
gates pass. Ordinary is an adjacent non-regression pair, not an ordinary speed
claim. The [qualification report](evidence/gemma4_26b/t3-down-window-2026-10-23.md)
and [acceptance record](../artifacts/trellis35/t3-down-window-2026-10-23.json)
retain all samples, commands, source identity, telemetry, positive/negative
operator evidence and limitations. No fresh llama or Windows/media/release claim;
all earlier measurements remain below.

## 2026-10-23 T3 GateUp codec-window optimization

The accepted native/vector GateUp change is measured against frozen parent
`6895ed2`, preserving the previous Down/M1 optimizations, precision and cache
semantics. Linux RTX 5080 Laptop GPU, max-power profile with Dynamic Boost,
16,384 prompt IDs, 1,135 output IDs, batch one, greedy, FP8 KV, three warmups
and ten retained runs per cohort:

| Mode / comparison | Parent decode tok/s | Candidate decode tok/s | Change | Sampled peak VRAM, both |
|---|---:|---:|---:|---:|
| Fixed-D2 AB | 179.103 | 193.284 | +7.9180% | 12,725 MiB |
| Fixed-D2 BA | 178.950 | 193.354 | +8.0492% | 12,725 MiB |
| Ordinary adjacent pair | 137.593 | 137.411 | -0.1323% | 12,455 MiB |

All sixty retained runs reproduce the exact output IDs. All forty D2 runs keep
502 groups and 632/1,004 accepted proposals, identical arena byte counts, and
no fallback or token-loop allocation. Ordinary intervals overlap; the adjacent
pair does not establish an ordinary speed improvement. Operator/reference,
sanitizer, exact draft/forced-acceptance and protected-profile checks pass.
The [qualification report](evidence/gemma4_26b/t3-gate-window-2026-10-23.md)
and [acceptance record](../artifacts/trellis35/t3-gate-window-2026-10-23.json)
retain source identity, binaries, commands, raw samples and counter evidence.
No new llama.cpp measurement or Windows/media/release qualification is claimed;
earlier campaigns below retain their original workloads and results.

## 2026-10-11 restored Trellis35 D2 comparison

The conditional CUDA graph fix restores native D2 on the installed 615.71.09 /
CUDA 13.4 stack. New 3+10 measurements with the identical 16K prompt and 1,135
forced greedy outputs give Trellis35 **5,665.11 prefill / 129.087 ordinary
decode tok/s**, and **172.314 D2 tok/s**. Ordinary/D2 outputs are exactly equal
in all retained runs. Sampled peak VRAM is **12,455 / 12,725 MiB**.

The completed October 10 current llama.cpp b11552 rows remain **138.461 ordinary
/ 196.850 D2 tok/s**, with **15,201 / 15,561 MiB** peak VRAM. These llama rows
are reused measurements, not a second campaign. Formats, KV precision and
cross-engine output sequences differ; llama's ordinary/D2 outputs also differ.
The [follow-up report](evidence/gemma4_26b/trellis35-d2-cuda-graph-fix-2026-10-11.md)
and [raw record](../benchmarks/baselines/llama_cpp/gemma4-26b-trellis35-q4_0-d2-followup-2026-10-11.json)
document graph isolation, system updates, the separate ring-atomic correction,
sanitizer and three-profile regression checks. Earlier evidence is retained below.

## 2026-10-10 current 26B comparison refresh

Current gem16 Trellis35 at `c01789a2` and newly built llama.cpp b11552 at
`23b0202a1` were measured on Linux with the same 16K prompt, 1,135 forced greedy
output tokens, three warmups and ten retained runs per successful mode.
Trellis35 ordinary reaches **5,623.23 prefill / 128.955 decode tok/s**;
llama.cpp official QAT Q4_0 reaches **4,607.79 / 138.461 tok/s** and
**196.850 tok/s with D2**. Trellis35 D2 fails during CUDA graph instantiation
before inference, reproduced three times, so the new D2 comparison is incomplete.
Formats/KV precision and output sequences differ; llama D2 also differs from its
ordinary output. The [refresh report](evidence/gemma4_26b/llama-cpp-trellis35-refresh-2026-10-10.md)
records the measured memory, complete protocol, current environment and failure
evidence. The tables below retain their historical measurements.

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

