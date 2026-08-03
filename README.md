<p align="center">
  <img src="docs/gem16_logo.svg" alt="gem16 logo" width="160">
</p>

<h1 align="center">gem16</h1>

<p align="center">
  Run Gemma 4 12B locally on a single 16 GB NVIDIA GPU.<br>
  Desktop app, OpenAI-compatible server, and a purpose-built CUDA inference engine.
</p>

<p align="center">
  <a href="https://github.com/Danmoreng/gem16/actions/workflows/ci.yml"><img src="https://github.com/Danmoreng/gem16/actions/workflows/ci.yml/badge.svg" alt="Linux and Windows CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2ea44f" alt="Apache 2.0 license"></a>
  <img src="https://img.shields.io/badge/status-development%20preview-e3b341" alt="Development preview">
</p>

<p align="center">
  <img src="docs/images/gem16-chat.png" alt="gem16 desktop app running a multimodal local Gemma 4 chat" width="1200">
</p>

gem16 is a local inference stack built specifically for Gemma 4 12B on Blackwell GPUs with about 16 GB of VRAM.
The desktop app is the primary entry point: it downloads the pinned model set into the shared Hugging Face cache,
starts or attaches to `gem16-server`, and provides a compact multimodal chat UI.

The engine loads the original mixed FP8/NVFP4 Safetensors checkpoint directly. It does not require GGUF conversion,
TensorRT engine generation, offline requantization, or a second persistent copy of the weights.

> [!IMPORTANT]
> gem16 is a development preview, not a release-qualified general-purpose runtime. The optimized CUDA backend
> currently targets Blackwell SM120/SM120a, and the supported model revisions are pinned deliberately.

## Experimental AI-developed engine

gem16 is programmed primarily with AI coding agents and is intended as an experimental engineering project. It
examines how direct checkpoint loading, model-specific execution plans, and Blackwell-specific CUDA kernels perform
for Gemma 4 12B within 16 GB of VRAM.

### Current 16K controlled performance comparison

#### Linux

On an RTX 5080 Laptop GPU at its firmware-managed 175 W ceiling, gem16 commit `8e86cb38`, vLLM 0.26.0, and
llama.cpp b10240 produce the following same-machine result:

| Engine | Checkpoint / KV | Prefill tok/s | TTFT | Effective D2 MTP tok/s | ITL | Sampled peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| vLLM 0.26.0 | direct FP8/NVFP4 / FP8 | **6,247.55** | **2,622.47 ms** | 81.95 | 12.202 ms | 15,465 MiB |
| **gem16** | direct FP8/NVFP4 / FP8 | 5,863.59 | 2,794.19 ms | **89.58** | **11.163 ms** | 11,867 MiB |
| llama.cpp b10240 | patched NVFP4+Q8_0 GGUF / Q8_0 | 3,922.61 | 4,176.81 ms | 82.88 | 12.065 ms | 10,631 MiB |

#### Windows (retained)

On Windows 11 x64, the same RTX 5080 Laptop GPU running gem16 commit `b9a73c2` in Lenovo Max Power mode with
CUDA 13.3 and driver 596.49 produced the following latest qualified Windows component results. The llama.cpp row is
the retained same-machine reference from the prior Windows characterization:

| Engine | Checkpoint / KV | Prefill tok/s | TTFT | Effective D2 MTP tok/s | ITL |
|---|---|---:|---:|---:|---:|
| **gem16** | direct FP8/NVFP4 / FP8 | **6,045.67** | **2,710.04 ms** | **91.46** | **10.933 ms** |
| llama.cpp 10210 | patched NVFP4+Q8_0 GGUF / Q8_0 | 3,937.64 | 4,160.87 ms | 86.00 | 11.628 ms |

The retained Windows gem16 entry combines two qualified screens. Prefill and TTFT come from that commit's serial
16,384-token synthetic one-output benchmark; the MTP throughput and ITL are retained from the exact
16,384-token Wikipedia / 1,135-output qualification at commit `2be75d7`. Both use batch one, three warm-ups, and
ten measured runs. The intervening changes affect only prefill; a same-head decode screen retains the prior
output checksum. The Linux comparison and retained Windows llama.cpp qualification use the exact Wikipedia prompt,
1,135 fixed greedy output positions, fixed D2 MTP, and count only target-verified output tokens.

In the Linux run, gem16 decode is **9.31% faster than vLLM** and **8.08% faster than llama.cpp**; its median ITL
is 8.51% and 7.48% lower, respectively. Gem16 prefill is 6.15% below vLLM and 49.48% above llama.cpp. This is a
controlled performance comparison, not exact output/semantic parity: all engines use the same prompt IDs, output
budget, batch, D2 policy, warm-ups, and repetitions, but their output hashes and prefill timing boundaries differ.
Against the retained Windows llama.cpp row, retained Windows gem16 decode is 6.35% faster and prefill is 53.54%
faster. vLLM is omitted from the Windows table because the pinned native runtime is not supported on Windows.

The retained Windows gem16 row uses Lenovo Max Power; the retained llama.cpp row used the Balanced OS power scheme,
although that earlier run dynamically reached approximately 176 W. The Windows ratio is therefore a same-machine orientation,
not a new adjacent cross-engine run. Both outputs are deterministic within each engine, but their token hashes
differ. The earlier Windows distributions, MTP counters, configurations, and filtered telemetry summary are retained
in the [machine-readable Windows characterization](benchmarks/baselines/cross_engine_mtp/windows-characterization.json);
the current gem16 prefill qualification and its 240 MiB workspace reduction are recorded in
[`docs/PERFORMANCE_LEDGER.md`](docs/PERFORMANCE_LEDGER.md).

Reproduce the complete three-engine run after preparing the pinned competitor environments:

```bash
sudo systemctl enable --now nvidia-powerd.service
echo max-power | sudo tee /sys/firmware/acpi/platform_profile
systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 \
  ./scripts/benchmark-cross-engine-mtp.sh
```

The script validates models, competitor versions, patches, GGUF checksums, power state, and an idle GPU. It stores
raw JSON, logs, and power/clock/thermal telemetry in a new result directory. See the
[full result and setup instructions](benchmarks/baselines/cross_engine_mtp/README.md) and the
[machine-readable characterization](benchmarks/baselines/cross_engine_mtp/characterization.json).

Comparison scope: gem16 and vLLM use the direct mixed FP8/NVFP4 checkpoint with FP8 KV. llama.cpp uses the patched
NVFP4+Q8_0 GGUF and Q8_0 KV. The engines produce different token hashes, and their prefill timing boundaries are
not identical. gem16 fixed-D2 is additionally checked for exact identity with its ordinary Target output.

## Architecture

[![gem16 runtime architecture](docs/gem16-architecture.svg)](https://raw.githubusercontent.com/Danmoreng/gem16/main/docs/gem16-architecture.svg)

The diagram covers checkpoint loading, runtime ownership, prefill, ordinary decode, transactional MTP verification,
fixed GPU memory, and the correctness/performance qualification gates. The SVG is exported from the
[editable tldraw source](docs/gem16-overview.tldraw).

## What works today

### Desktop app

- Download and verify the target model, the official MTP draft assistant, and Google's tokenizer configuration.
- Reuse the standard Hugging Face Hub cache through `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME`.
- Start, stop, restart, inspect, or attach to a local `gem16-server` process.
- Stream Markdown answers with separate, collapsed-by-default reasoning.
- Copy complete answers or individual fenced code and HTML blocks with one click.
- Send text, PNG/JPEG/BMP images, and WAV/FLAC/MP3 audio.
- Attach media with the file picker, drag and drop, microphone recording, or image paste with `Ctrl+V`/`Cmd+V`.
- See live prefill, decode, token, and throughput statistics while a response is streaming.
- Configure context, sampling, MTP draft budget, reasoning budget, and light/dark appearance.

### Runtime

| Area | Current implementation |
|---|---|
| Model | Pinned Gemma 4 12B Unified mixed FP8/NVFP4 checkpoint |
| Platforms | Windows x64 and Linux x86-64; optimized CUDA path for SM120/SM120a |
| Execution | Batch-one prefill and decode with resident weights and conversation KV state |
| Precision | FP8 attention, packed NVFP4 MLPs, BF16 embeddings, FP8 or BF16 KV cache |
| Decode | Whole-model CUDA Graph replay and native T=1 projection plans |
| MTP | Optional pinned assistant with D1/D2/D4 verification and GPU acceptance/commit |
| Inputs | Text, images, and audio; video is not implemented |
| Interfaces | Desktop app, interactive CLI, and OpenAI-compatible HTTP/SSE server |
| Validation | Host/CUDA tests plus operator, layer, logit, generation, and long-context checks |

Unsupported precision paths fail visibly. A missing native NVFP4 kernel is never hidden behind a silent
higher-precision fallback.

## Run the desktop app from source

### Prerequisites

- A Blackwell NVIDIA GPU with approximately 16 GB of VRAM for the optimized inference path
- The pinned CUDA toolkit and CUTLASS submodule
- CMake 3.28 or newer, Ninja, and a C++20 compiler
- JDK 21 for the Compose Desktop application
- A Hugging Face account with access to the gated Google Gemma repositories

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

### Linux

```bash
./scripts/build.sh --cuda --test
./scripts/run-studio.sh
```

On first launch, open **Models**, provide a Hugging Face token if necessary, and download the pinned model set.
Files are stored in the shared content-addressed Hugging Face cache and are not copied into the application.
After verification, the managed server can start and the Chat screen is ready to use.

To build a native installer for the current platform after the release server exists:

```powershell
# Windows
.\scripts\package-studio.ps1
```

The Windows script always rebuilds the MSI and writes it to
`studioApp/build/compose/binaries/main/msi/`. The installer deploys gem16 per machine under
`C:\Program Files\gem16`; the CUDA-enabled server is embedded at
`app\resources\bin\gem16-server.exe`. Checkpoints remain outside the installation in the shared Hugging Face cache.

Pushing a `v*` tag, or manually dispatching the **Windows Release** workflow, builds the pinned CUDA 13.3 SM120a
server on GitHub Actions, verifies native NVFP4/FP8 instructions, packages the MSI, publishes its SHA-256 checksum,
and attaches both files to the GitHub Release.

```bash
# Linux
./scripts/package-studio.sh
```

See [`docs/STUDIO.md`](docs/STUDIO.md) for application packaging, cache behavior, protocol details, and security
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

`gem16-server` exposes `/health`, `/metrics`, `/v1/models`, `/v1/chat/completions`, and `/v1/responses`. It supports
streaming SSE, reasoning deltas, cancellation, structured tools, resident sessions, and ordered multimodal content.

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
| `gem16-server` | OpenAI-compatible local Chat Completions and Responses server |
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

gem16 reports prefill, ordinary decode, MTP, memory, and end-to-end measurements separately. The current prominent
comparison above is a fixed-output greedy D2 MTP characterization under a controlled Linux max-power profile; it
does not replace ordinary-decode, task-quality, or long-context gates. Cross-runtime token identity is not expected,
but every engine must remain deterministic and all format, cache, timing-boundary, and fallback differences stay
visible.

Hardware, checkpoint, context, output count, sampling, quality, VRAM, clocks, power, and baseline tensor formats
must accompany any performance claim. Reproduction commands, raw-evidence policy, caveats, and historical results
live in [`benchmarks/baselines/cross_engine_mtp/`](benchmarks/baselines/cross_engine_mtp/),
[`docs/PERFORMANCE_LEDGER.md`](docs/PERFORMANCE_LEDGER.md), and
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Current limitations

- Only the pinned Gemma 4 12B Unified checkpoint family is supported.
- Inference is batch one; continuous batching is not implemented.
- The optimized CUDA backend requires Blackwell SM120/SM120a.
- Video input, response branching, and persistent prompt-cache files are not implemented.
- Numerical validation and wider task-quality qualification remain active work.

## Documentation

- [`docs/STUDIO.md`](docs/STUDIO.md) — desktop app, model downloads, and managed server
- [`docs/SERVER.md`](docs/SERVER.md) — HTTP APIs, streaming, sessions, tools, and media
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — runtime and kernel architecture
- [`docs/CORRECTNESS.md`](docs/CORRECTNESS.md) — numerical validation and current gates
- [`docs/MTP.md`](docs/MTP.md) — assistant model and speculative decode design
- [`docs/DECODE_OPTIMIZATION_PLAN.md`](docs/DECODE_OPTIMIZATION_PLAN.md) — active ordinary-decode and MTP performance plan
- [`docs/LINUX_DECODE_HANDOFF_2026-07-31.md`](docs/LINUX_DECODE_HANDOFF_2026-07-31.md) — reproducible Linux continuation state
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — benchmark methodology
- [`docs/PERFORMANCE_LEDGER.md`](docs/PERFORMANCE_LEDGER.md) — retained measurements and profiling evidence
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — remaining milestones

## License

gem16 is licensed under the [Apache License 2.0](LICENSE). The vendored CUTLASS dependency retains its BSD-3-Clause
license. Model weights and tokenizer assets are distributed separately under their respective terms.
