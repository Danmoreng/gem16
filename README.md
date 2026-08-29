<p align="center">
  <img src="docs/gem16_logo.svg" alt="gem16 logo" width="160">
</p>

<h1 align="center">gem16</h1>

<p align="center">
  Run Gemma 4 12B or 26B locally on a single 16 GB NVIDIA GPU.<br>
  Native desktop app, OpenAI Agent Core v1 server, and a purpose-built CUDA inference engine.
</p>

<p align="center">
  <a href="https://github.com/Danmoreng/gem16/actions/workflows/ci.yml"><img src="https://github.com/Danmoreng/gem16/actions/workflows/ci.yml/badge.svg" alt="Linux and Windows CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2ea44f" alt="Apache 2.0 license"></a>
  <img src="https://img.shields.io/badge/status-development%20preview-e3b341" alt="Development preview">
</p>

<p align="center">
  <img src="docs/images/gem16-chat.png" alt="gem16 desktop app running a multimodal local Gemma 4 chat" width="1200">
</p>

gem16 is a local inference stack built specifically for Gemma 4 on Blackwell GPUs with about 16 GB of VRAM. Gemma 4
12B Unified and Gemma 4 26B A4B are equal, user-selectable product profiles that can be installed side by side. The
native C++ desktop app is the primary entry point: it starts or attaches to `gem16-server`, manages the selected
profile, and provides local streamed chat. The 12B profile supports text, image, and audio; 26B is text-only and may
use its separately pinned fixed-D2 Assistant.

The 12B profile loads its pinned mixed FP8/NVFP4 Safetensors checkpoint directly. The 26B profile consumes its
offline-compiled, immutable GEM16 Target artifact and optional separately compiled Assistant.

> [!IMPORTANT]
> gem16 is a development preview, not a release-qualified general-purpose runtime. The optimized CUDA backend
> currently targets Blackwell SM120/SM120a, and the supported model revisions are pinned deliberately.

## Specialized project

gem16 is developed primarily with AI coding agents. It explores model-specific execution plans and
Blackwell-optimized CUDA kernels for the qualified Gemma 4 12B and 26B profiles within 16 GB of VRAM.

## Gemma 4 26B sampled performance

On the RTX 5080 Laptop GPU, the qualified text-only 26B path completes the fixed 16,384-token Wikipedia workload
with Google's recommended sampling controls as follows:

| Mode | Decode tok/s | Output tokens |
|---|---:|---:|
| Ordinary | 148.293 | 942 |
| **Fixed D2 MTP** | **203.842** | 942 |

This is a batch-one, checkpoint-FP8-KV characterization with seed 0, `temperature=1`, `top_k=64`, `top_p=0.95`,
three paired warm-ups and ten retained alternating pairs. D2 is 37.46% faster than ordinary, has a 2,355.225 ms
median TTFT and 6,970.995 ms median end-to-end inference time, and peaks at 15,024 MiB. Every retained run produces
the same output hash; D2 accepts 523 of 836 proposals (62.56%). Assistant precision and acceptance were frozen during
this execution-only optimization. See the
[compact evidence](artifacts/m25/decode-optimization-freeze-2026-08-28.json).

## Current 16K performance

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

See the [full methodology](benchmarks/baselines/cross_engine_mtp/README.md),
[Linux data](benchmarks/baselines/cross_engine_mtp/characterization-a819d14c.json), and
[Windows data](benchmarks/baselines/cross_engine_mtp/windows-characterization-35a57bb.json).

## Current short-context performance

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
[tracked Linux summary](benchmarks/baselines/llama_cpp/linux-short-context-065b68f.json).

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
[complete samples, commands, and telemetry](benchmarks/baselines/llama_cpp/windows-short-context-cc01a05.json).

## Architecture

[![gem16 runtime architecture](docs/gem16-architecture.svg)](https://raw.githubusercontent.com/Danmoreng/gem16/main/docs/gem16-architecture.svg)

The diagram covers checkpoint loading, runtime ownership, GPU execution, memory, decode, and transactional MTP.
It is exported from the [editable tldraw source](docs/gem16-overview.tldraw).

## What works today

### Desktop app

- On a fresh installation, choose either qualified profile and install its separately pinned Target and Assistant;
  install both profiles side by side if desired.
- Resume interrupted downloads, verify every locked file with SHA-256, and check required disk space before download.
- Reuse the standard Hugging Face Hub cache through `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME`.
- Start, stop, restart, inspect, or attach to a local `gem16-server` process.
- Stream Markdown answers with separate, collapsed-by-default reasoning.
- Copy complete answers or individual fenced code and HTML blocks with one click.
- Configure the system prompt, context, sampling, MTP draft budget, reasoning budget, and appearance.

### Runtime

| Area | Current implementation |
|---|---|
| Models | Pinned Gemma 4 12B Unified and Gemma 4 26B A4B profiles |
| Platforms | Windows x64 and Linux x86-64; optimized CUDA path for SM120/SM120a |
| Execution | Batch-one prefill and decode with resident weights and conversation KV state |
| Precision | FP8 attention, packed NVFP4 MLPs, BF16 embeddings, FP8 or BF16 KV cache |
| Decode | Whole-model CUDA Graph replay and native T=1 projection plans |
| MTP | Optional pinned assistant with D1/D2/D4 verification and GPU acceptance/commit |
| Inputs | Text, images, and audio; video is not implemented |
| Interfaces | Desktop app, interactive CLI, and bounded OpenAI Agent Core v1 HTTP/SSE server |
| Validation | Host/CUDA tests plus operator, layer, logit, generation, and long-context checks |

Unsupported precision paths fail visibly. A missing native NVFP4 kernel is never hidden behind a silent
higher-precision fallback.

## Run the desktop app from source

### Prerequisites

- A Blackwell NVIDIA GPU with approximately 16 GB of VRAM for the optimized inference path
- The pinned CUDA toolkit and CUTLASS submodule
- CMake 3.28 or newer, Ninja, and a C++20 compiler
- `curl` on `PATH` for resumable downloads in the native Models screen (included with current Windows)
- OpenGL/X11 or Wayland development libraries on Linux; the Windows client uses Direct3D 11
- No account or token is required for the public repositories pinned by the current Studio catalog

The exact reference environment is recorded in
[`toolchains/blackwell16gb.lock`](toolchains/blackwell16gb.lock).

Clone the repository and initialize its pinned dependencies:

```bash
git clone --recurse-submodules https://github.com/Danmoreng/gem16.git
cd gem16
```

### Windows

From PowerShell:

```powershell
.\scripts\build.ps1 -Cuda -Test
.\scripts\run-studio.ps1
```

The development launcher incrementally rebuilds and uses the workspace CUDA
server. Pass `-SkipServerBuild` only when that native binary is already current;
persisted settings pointing at an installed server do not override development
launches.

### Linux

```bash
./scripts/build.sh --cuda --test
./scripts/run-studio.sh
```

The Linux launcher provides the equivalent `--skip-server-build` opt-out.

On first launch, Studio opens **Models** and presents 12B and 26B without a preselected default. Install either profile
or both, then select a verified profile and verify the compiled path on **Server**. Studio reuses already populated Hub
blobs, resumes incomplete downloads, and never copies checkpoints into the application archive. The 12B Target spans
two existing upstream repositories, so Studio creates a hardlink-only runtime view inside the same Hub cache; it does
not publish a mirror or duplicate the payload bytes.

To build a native portable archive for the current platform after the release server exists:

```powershell
# Windows
.\scripts\package-studio.ps1
```

The Windows script writes `build/packages/gem16-windows-x64.zip`. The portable archive includes the CUDA-enabled
server; checkpoints remain external.

```bash
# Linux
./scripts/package-studio.sh
```

See [`docs/STUDIO.md`](docs/STUDIO.md) for native architecture, packaging, protocol details, and security
notes.

## Download models without the app

The same locked model set can be populated from Python:

```bash
python tools/fetch_model.py
python tools/fetch_model.py --lock models/gemma4-12b-mtp-assistant.lock.json
```

The downloader verifies exact revisions, file sizes, and SHA-256 checksums. Benchmark, validation, and test tools
resolve these cache snapshots automatically; `--model` remains available as an explicit override.

## Server and CLI

`gem16-server` exposes `/health`, `/metrics`, `/v1/models`, `/v1/chat/completions`, and `/v1/responses`. Its published
[OpenAI Agent Core v1 contract](docs/OPENAI_AGENT_CORE_V1.md) supports streaming SSE, reasoning deltas, cancellation,
client-executed function tools, resident sessions, and ordered multimodal content where the selected profile permits
it. This is a bounded compatibility subset, not complete OpenAI platform emulation.

```powershell
$model = python -c "from tools.hf_cache import default_target_model; print(default_target_model())"
$assistant = python -c "from tools.hf_cache import default_assistant_model; print(default_assistant_model())"

.\build\Windows\blackwell-release\bin\gem16-server.exe `
  --model $model `
  --assistant-model $assistant `
  --mtp-draft-tokens 2 `
  --model-name gem16 `
  --host 127.0.0.1 `
  --port 8080 `
  --max-context 32768
```

For terminal chat, replace `gem16-server.exe` with `gem16-chat.exe`. The CLI supports resident multi-turn chat,
reasoning budgets, sampling, images, audio, local function-tool schemas, MTP, and per-turn statistics. Full request
examples live in [`docs/SERVER.md`](docs/SERVER.md), [`docs/VISION.md`](docs/VISION.md), and
[`docs/AUDIO.md`](docs/AUDIO.md).

| Binary | Purpose |
|---|---|
| `gem16-server` | Local OpenAI Agent Core v1 Chat Completions and Responses server |
| `gem16-chat` | Interactive or one-shot terminal chat |
| `gem16-run` | Inference, MTP, teacher forcing, state dumps, and capability reporting |
| `gem16-inspect` | Checkpoint validation and tensor/quantization inventory |
| `gem16-bench` | Model-load, memory, kernel, prefill, decode, and end-to-end benchmarks |

## Why a specialized engine?

gem16 follows the actual Gemma 4 architecture and checkpoint metadata instead of routing the model through a
generic graph framework:

- FP8 attention projections and packed NVFP4 MLP tensors are consumed according to the checkpoint schema.
- Safetensors are memory-mapped, validated, and streamed into final GPU allocations without a converted checkpoint.
- Gemma's local/global attention pattern, proportional RoPE, K=V semantics, logit softcap, and chat template are
  implemented explicitly.
- The token loop reuses resident weights, KV state, activation arenas, and captured CUDA Graphs without model-memory
  allocation.
- Reference paths remain beside optimized kernels so performance changes can be checked against numerical evidence.

Architecture and memory details are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/CHECKPOINT_FORMAT.md`](docs/CHECKPOINT_FORMAT.md), and [`docs/MEMORY.md`](docs/MEMORY.md).

## Performance and correctness

gem16 reports prefill, ordinary decode, MTP, memory, and end-to-end measurements separately. The comparison above
uses controlled same-machine Linux and Windows profiles; it does not replace ordinary-decode, task-quality, or
long-context gates. Cross-runtime format, cache, output, timing-boundary, and fallback differences remain visible.

Hardware, checkpoint, context, output count, sampling, quality, VRAM, clocks, power, and baseline tensor formats
must accompany any performance claim. Reproduction commands, raw-evidence policy, caveats, and current results
live in [`benchmarks/baselines/cross_engine_mtp/`](benchmarks/baselines/cross_engine_mtp/),
[`docs/PERFORMANCE_LEDGER.md`](docs/PERFORMANCE_LEDGER.md), and
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Current limitations

- Gemma 4 12B Unified and 26B A4B are equal product choices with different capabilities: 12B is multimodal; 26B is
  text-only, single-slot, and optionally uses its separately pinned Assistant.
- Full-model download and clean-machine onboarding still need release qualification on both Windows and Linux.
- Inference is batch one; continuous batching is not implemented.
- The optimized CUDA backend requires Blackwell SM120/SM120a.
- Video input, response branching, and persistent prompt-cache files are not implemented.
- Numerical validation and wider task-quality qualification remain active work.

## Documentation

- [`docs/PRODUCT_CONTRACT.md`](docs/PRODUCT_CONTRACT.md) — equal platforms, model profiles, GUI, and release boundary
- [`docs/OPENAI_AGENT_CORE_V1.md`](docs/OPENAI_AGENT_CORE_V1.md) — exact coding-agent API subset and qualification gate
- [`docs/STUDIO.md`](docs/STUDIO.md) — native C++ desktop app, chat, packaging, and managed server
- [`docs/SERVER.md`](docs/SERVER.md) — HTTP APIs, streaming, sessions, tools, and media
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — runtime and kernel architecture
- [`docs/CORRECTNESS.md`](docs/CORRECTNESS.md) — numerical validation and current gates
- [`docs/MTP.md`](docs/MTP.md) — assistant model and exact speculative-decode contract
- [`docs/AUDIO.md`](docs/AUDIO.md) and [`docs/VISION.md`](docs/VISION.md) — implemented media-input contracts
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — benchmark methodology
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — C++, CUDA, testing, dependency, and security rules
- [`docs/PERFORMANCE_IMPROVEMENT_PLAN.md`](docs/PERFORMANCE_IMPROVEMENT_PLAN.md) — completed bounded 12B performance sprint
- [`docs/GEMMA4_26B.md`](docs/GEMMA4_26B.md) — qualified 26B artifact and product contract
- [`docs/GEMMA4_26B_HUGGING_FACE.md`](docs/GEMMA4_26B_HUGGING_FACE.md) — immutable Target/Assistant publication and download contract
- [`docs/plans/gemma4-26b/START_HERE_CODEX.md`](docs/plans/gemma4-26b/START_HERE_CODEX.md) — active 26B M00 entry point
- [`docs/PERFORMANCE_LEDGER.md`](docs/PERFORMANCE_LEDGER.md) — retained measurements and profiling evidence
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — active tracks and deferred work

## License

gem16 is licensed under the [Apache License 2.0](LICENSE). The vendored CUTLASS dependency retains its BSD-3-Clause
license. Model weights and tokenizer assets are distributed separately under their respective terms.
