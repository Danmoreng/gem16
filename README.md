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

gem16 reports prefill, decode, MTP, memory, and end-to-end measurements separately. The current retained Wikipedia
16K greedy qualification on the Windows reference setup measures 54.903 output tok/s with fixed D2 MTP versus
36.788 tok/s ordinary decode (1.492x). A separate Linux sampled qualification reaches 46.234 tok/s versus
31.450 tok/s (1.470x), with ordinary/MTP-identical outputs for all retained runs at the qualified seed and profile.

These numbers are characterization results, not a universal speed claim. Hardware, checkpoint, context, timing
boundaries, sampling, quality, VRAM use, and baseline tensor formats must all be disclosed. Reproduction commands,
raw evidence, caveats, and comparison rules live in
[`docs/PERFORMANCE_LEDGER.md`](docs/PERFORMANCE_LEDGER.md) and
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
