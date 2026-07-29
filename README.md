<p align="center">
  <img src="docs/gem16_logo.svg" alt="gem16 logo" width="180">
</p>

<h1 align="center">gem16</h1>

<p align="center">
  A specialized C++20/CUDA inference engine for running Gemma 4 12B on a single 16 GB NVIDIA GPU.
</p>

`gem16` loads the mixed FP8/NVFP4 Hugging Face checkpoint
[`unsloth/gemma-4-12b-it-NVFP4`](https://huggingface.co/unsloth/gemma-4-12b-it-NVFP4) directly. It does not
require GGUF conversion, TensorRT engine generation, requantization, or a second persistent copy of the model
weights. The first optimized backend targets Blackwell SM120/SM120a and batch-one interactive text generation.

> **Project status:** development preview. End-to-end inference, native tokenization, streaming chat, prefill,
> decode, KV caching, CUDA Graphs, and Windows/Linux builds work on the reference setup. Correctness and performance
> are actively characterized, but this is not yet a general-purpose or release-qualified inference runtime.

## Current capabilities

| Area | Current state |
|---|---|
| Checkpoint | Direct loading of the pinned Gemma 4 12B Unified mixed FP8/NVFP4 Safetensors checkpoint |
| Platforms | Linux x86-64 and Windows x64 builds; native CUDA path targets Blackwell SM120/SM120a |
| Inference | Text-only, batch one, greedy or seeded GPU sampling with resident model and conversation KV state |
| CLI | Native tokenizer and chat template, UTF-8 output, token streaming, multi-turn chat |
| Prefill | Native variable-length prefill with CUTLASS FP8/NVFP4 Tensor Core projections |
| Decode | Native T=1 projection plans, FP8 KV cache, and whole-model CUDA Graph replay |
| MTP | Optional official BF16 assistant with exact batched D1/D2/D4 verification, GPU acceptance/commit, and adaptive fallback |
| Memory | Direct source layout, unified tensor loading, deterministic arenas, no CPU weight offload |
| Tooling | Checkpoint inspection, memory planning, correctness probes, profiling, and prefill/decode benchmarks |
| Validation | Host and CUDA tests plus operator, layer, logit, greedy-generation, and long-context checks |

The implementation is intentionally narrow. It is built around the actual Gemma 4 12B architecture and
checkpoint metadata rather than a generic graph framework. Unsupported precision paths fail visibly; native
NVFP4 execution never silently falls back to a higher-precision implementation.

## Design highlights

- Mixed-precision execution follows the checkpoint schema: FP8 attention projections, packed NVFP4 MLP weights,
  dynamic activation quantization, hierarchical block/global scales, and tied BF16 embeddings.
- Safetensors files are memory-mapped and validated. Weight and scale data are streamed into their final GPU arena
  without creating a converted checkpoint on disk.
- Local and global Gemma attention, proportional RoPE, K=V semantics, logit softcap, stop-token handling, and the
  checkpoint chat template are implemented explicitly.
- The token loop performs no model-memory allocations. Decode reuses resident weights, KV state, activation arenas,
  and a captured CUDA Graph.
- Diagnostic and reference paths remain available beside optimized kernels so numerical changes can be measured.

## Requirements

- CMake 3.28 or newer
- Ninja
- A C++20 compiler
- The pinned CUDA toolkit and CUTLASS submodule for CUDA builds
- A Blackwell GPU with compute capability 12.0 for the optimized inference path
- Approximately 16 GB of VRAM for the primary checkpoint and supported context profiles

The exact reference environment is recorded in
[`toolchains/blackwell16gb.lock`](toolchains/blackwell16gb.lock). Host-only inspection and parser tests do not
require CUDA.

## Get the model

The repository pins the model and tokenizer sources in
[`models/gemma4-12b-nvfp4.lock.json`](models/gemma4-12b-nvfp4.lock.json). Download the locked files with:

```bash
python tools/fetch_model.py \
  --destination models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497
```

The downloader verifies file sizes and SHA-256 checksums. It requires a Hugging Face token with access to the
gated Google Gemma model.

## Build and test

Initialize the pinned dependencies first:

```bash
git submodule update --init --recursive
```

### Linux

```bash
./scripts/build.sh --cuda --test
```

For a host-only build:

```bash
./scripts/build.sh --host --test
```

### Windows

Run from PowerShell. The build script discovers Visual Studio, Ninja, and the configured CUDA toolkit:

```powershell
.\scripts\build.ps1 -Cuda -Test
```

For a host-only build:

```powershell
.\scripts\build.ps1 -Test
```

Warnings are reported normally and are not promoted to errors by default.

## Chat

Linux:

```bash
build/Linux/blackwell-release/bin/gem16-chat \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --max-context 8192
```

Windows PowerShell:

```powershell
.\build\Windows\blackwell-release\bin\gem16-chat.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --assistant-model .\models\checkpoints\google-gemma-4-12B-it-assistant-364bd03 `
  --mtp-draft-tokens 2 `
  --max-context 8192 `
  --stats
```

The model and exact conversation prefix remain resident between turns. Generated text is decoded and written to
the terminal incrementally. The CLI now consumes the protocol-neutral `ChatSession` request/event API in
`include/gem16/chat.h`; future HTTP/OpenAI adapters do not need terminal or CUDA-prefix knowledge. The optional
official assistant also remains resident. Chat defaults to medium thinking (4,096 reasoning tokens maximum) and
the pinned Google generation profile (`temperature=1.0`, `top_k=64`, `top_p=0.95`). Use
`--thinking-budget off|small|medium|high` for 0/1,024/4,096/8,192-token reasoning caps and `--greedy` for explicit greedy
decoding. Fixed D2 uses the GPU-chained conditional graph for greedy and sampled generation. During bounded
reasoning, the same graph routes safe full groups through MTP and only exact marker/budget/tail boundaries through
ordinary Target decode; transitions and MTP resumption remain GPU-controlled without a blocking host roundtrip. With no
`--max-tokens`, a turn runs until a checkpoint stop token or the remaining `--max-context` capacity; pass
`--max-tokens N` for a stricter per-turn limit. `--stats` prints per-turn throughput, proposal/acceptance counts,
verifier groups, and whether GPU chaining was active. Streaming output labels the model's private reasoning and
visible response as separate `--- thinking ---` and `--- answer ---` sections. `--hide-thinking` suppresses the
reasoning body while retaining the answer header; `--show-thinking` is the default. Enter `/quit` to leave the
session.

Resident chat also accepts media without restarting the process. `/image <path>` and `/audio <path>` add files to
the next user message, `/media` lists the pending queue, and `/clear-media` discards it. Files are decoded only when
the next ordinary text message is submitted. Once sent, their projected prompt remains part of the exact resident
conversation prefix, so later text turns and tool continuations can refer back to the media.

Resident chat can expose local function tools. Repeat `--tool` with a name, description, and JSON Schema file. When
Gemma requests a function, the CLI prints its validated JSON arguments, prompts for the external result, appends that
result to the resident KV prefix, and continues generation automatically:

```powershell
.\build\Windows\blackwell-release\bin\gem16-chat.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --tool get_weather "Get current weather" .\examples\tools\get_weather.schema.json `
  --no-thinking --greedy --max-context 1024
```

The CLI deliberately asks the user to execute the tool; gem16 does not run arbitrary functions itself.

Audio input is available in one-shot and resident chat. The unified checkpoint's audio and
vision tensors are always loaded with the text weights; there is no modality
residency switch. WAV, FLAC, and MP3 input is decoded by the pinned miniaudio
single-header library and converted to the model's mono float32 16-kHz frame
contract:

```powershell
.\build\Windows\blackwell-release\bin\gem16-chat.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --message "Transkribiere das Audio wortgetreu." `
  --audio C:\path\sample.wav `
  --no-thinking --max-context 1024 --max-tokens 128
```

`--audio` and `--image` may be repeated in one one-shot message; media remain in command-line order after the text
part. Image resolution is selected automatically from source dimensions, media count, audio length, output reserve,
and `--max-context`, capped at 280 tokens. The equivalent resident commands may also be repeated before submitting
the message. See [docs/AUDIO.md](docs/AUDIO.md) for the qualified audio contract. PNG, JPEG, and BMP files are decoded
to RGB, processed into the checkpoint's native merged patches, and projected
by the complete encoder-free vision embedder on the GPU:

```powershell
.\build\Windows\blackwell-release\bin\gem16-chat.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --message "Beschreibe das Bild und lies sichtbaren Text." `
  --image C:\path\image.png `
  --no-thinking --max-context 1024 --max-tokens 128
```

See [docs/VISION.md](docs/VISION.md) for preprocessing, attention semantics,
current format limits, and the verification record.

## OpenAI-compatible server

`gem16-server` exposes `/health`, `/metrics`, `/v1/models`,
`/v1/chat/completions`, and `/v1/responses` (including cancellation), with HTTP
chunked SSE, usage records, structured function calls/results, reasoning
deltas, and ordered multimodal content:

```powershell
.\build\Windows\blackwell-release\bin\gem16-server.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --model-name gem16 `
  --host 127.0.0.1 --port 8080 --max-context 8192 --max-sessions 2
```

Text strings and `text`/`input_text` parts are accepted. Images use OpenAI
`image_url` parts with inline `data:image/png|jpeg|bmp;base64,...` URLs; audio
uses `input_audio` with Base64 `wav`, `mp3`, or `flac`. Repeated parts preserve
their JSON order and share the automatic image-token budget. Function tools,
assistant `tool_calls`, and `tool` result messages map directly onto the native
Gemma tool protocol. See [docs/SERVER.md](docs/SERVER.md) for requests, SSE
events, bounded multi-session semantics, and visible unsupported fields.
The official OpenAI Python SDK agent gate is
`tools/validate_openai_agent.py`, pinned by
`tools/requirements-openai-sdk.txt`.

## Command-line tools

| Tool | Purpose |
|---|---|
| `gem16-chat` | Interactive or single-message chat with native tokenization and streaming output |
| `gem16-server` | Bounded multi-session OpenAI-compatible Chat/Responses server |
| `gem16-run` | Greedy or sampled inference, MTP, teacher forcing, state dumps, and kernel capability reporting |
| `gem16-inspect` | Validate and inventory checkpoint tensors and quantization metadata |
| `gem16-bench` | Model-load, memory, kernel, prefill, decode, and end-to-end characterization |

For example:

```bash
gem16-inspect --model /path/to/checkpoint --validate
gem16-run --print-kernel-capabilities
gem16-bench prefill --model /path/to/checkpoint --context 8192
gem16-bench decode --model /path/to/checkpoint --context 8192 --tokens 256
```

## Correctness and performance status

The engine passes its host and CUDA test suites and has real-checkpoint coverage from low-level codecs through
complete 48-layer generation. Teacher-forced logits agree closely with the reference implementations, but they are
not bit-identical; small numerical differences can still cause later greedy tokens to diverge autoregressively.

Prefill and decode are benchmarked separately at long context. On the Linux RTX 5080 Laptop reference machine,
the current 8K/256 decode median is 33.545 tok/s and the retained direct-vLLM characterization is 38.056 tok/s;
gem16 reaches 88.1% under disclosed non-parity timing boundaries. It remains ahead of the patched closest-parity
`llama.cpp` candidate. These results are characterization, not a headline parity claim: timing boundaries differ,
and the direct mixed checkpoint and available GGUF baseline differ in some tensor and KV-cache formats. Commands, caveats, and historical measurements live in
[`docs/PERFORMANCE_LEDGER.md`](docs/PERFORMANCE_LEDGER.md), while the required comparison rules are documented in
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Known limitations

- Only the pinned Gemma 4 12B Unified checkpoint family is supported.
- Inference is currently batch one. Text, image, and audio input are supported; video input is not yet implemented.
- The server bounds resident isolated execution slots with `--max-sessions` and
  shares immutable target/assistant weights. Chat clients retain
  `X-Gem16-Session-Id`; Responses clients continue through
  `previous_response_id`. Continuous batching is not implemented.
- Generation supports unchanged fused greedy selection and explicit seeded GPU sampling with temperature, exact
  top-k/top-p/min-p filtering, and full-history repetition penalty. The initial sampled path uses a preallocated
  full-vocabulary radix sort and probability scan inside the whole-model decode CUDA Graph.
- The optimized CUDA backend requires Blackwell SM120/SM120a. Other NVIDIA architectures are not performance
  targets yet.
- Optional MTP supports exact batched target verification at D1/D2/D4, device-resident drafts,
  GPU-side acceptance/commit, greedy generation, and same-seed target-exact sampling. Sampled verification commits
  only the emitted RNG/repetition prefix; fixed D2 keeps stop/tail handling and mapped-pinned streaming inside the
  GPU conditional graph. `--mtp-adaptive` remains available through the direct D1/D2/D4 paths. On the exact
  Wikipedia 16K workload, a GPU-chained fixed-D2 conditional graph preserves all 1,135 ordinary IDs and measures
  54.903 tok/s median versus 36.788 ordinary (1.492x, +49.2%) in the final three-warm-up/ten-run qualification at
  batch one with checkpoint-FP8 KV. GPU stop, final ordinary tails, and a mapped-pinned asynchronous callback ring
  are included. The Windows greedy 50 tok/s gate is passed; the 55 tok/s stretch target is missed by 0.097 tok/s.
  A separate Linux 3/10 run reaches 47.117 greedy MTP versus 31.634 ordinary. Under Google's recommended sampling
  profile at seed 42, all 26 Linux warm-up/measured outputs are ordinary/MTP-identical; sampled D2 reaches 46.234
  tok/s versus 31.450 ordinary (1.470x). This qualifies sampled correctness and benchmark reproducibility, but does
  not meet the existing 50 tok/s performance target. Greedy and sampled MTP are available in resident multi-turn
  chat.
- Continuous batching, response branching, and persistent prompt-cache files are not yet implemented.
- Full benchmark qualification, wider quality evaluation, and additional long-context validation remain ongoing.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — runtime and kernel architecture
- [`docs/CHECKPOINT_FORMAT.md`](docs/CHECKPOINT_FORMAT.md) — source tensors and quantization schema
- [`docs/CORRECTNESS.md`](docs/CORRECTNESS.md) — numerical validation strategy and current gates
- [`docs/MEMORY.md`](docs/MEMORY.md) — device arenas, KV cache, and context profiles
- [`docs/MTP.md`](docs/MTP.md) — pinned assistant, feasibility evidence, and implementation plan
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — benchmark methodology and comparison contract
- [`docs/PERFORMANCE_LEDGER.md`](docs/PERFORMANCE_LEDGER.md) — detailed measurements and profiling evidence
- [`docs/SERVER.md`](docs/SERVER.md) — OpenAI Chat Completions, SSE, tools, and media transport
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — remaining milestones

## License

gem16 is licensed under the [Apache License 2.0](LICENSE). The vendored CUTLASS dependency retains its
BSD-3-Clause license. Model weights and tokenizer assets are distributed separately under their respective terms.
