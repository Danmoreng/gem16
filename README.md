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
| Memory | Direct source layout, text-only tensor loading, deterministic arenas, no CPU weight offload |
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
verifier groups, and whether GPU chaining was active. Enter `/quit` to leave the session.

## Command-line tools

| Tool | Purpose |
|---|---|
| `gem16-chat` | Interactive or single-message chat with native tokenization and streaming output |
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
- Inference is currently text-only and batch one; image, audio, and video tensors are not loaded onto the GPU.
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
- Continuous batching, a server API, and persistent prompt-cache files are out of scope for the current runtime.
- Full benchmark qualification, wider quality evaluation, and additional long-context validation remain ongoing.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — runtime and kernel architecture
- [`docs/CHECKPOINT_FORMAT.md`](docs/CHECKPOINT_FORMAT.md) — source tensors and quantization schema
- [`docs/CORRECTNESS.md`](docs/CORRECTNESS.md) — numerical validation strategy and current gates
- [`docs/MEMORY.md`](docs/MEMORY.md) — device arenas, KV cache, and context profiles
- [`docs/MTP.md`](docs/MTP.md) — pinned assistant, feasibility evidence, and implementation plan
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — benchmark methodology and comparison contract
- [`docs/PERFORMANCE_LEDGER.md`](docs/PERFORMANCE_LEDGER.md) — detailed measurements and profiling evidence
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — remaining milestones

## License

gem16 is licensed under the [Apache License 2.0](LICENSE). The vendored CUTLASS dependency retains its
BSD-3-Clause license. Model weights and tokenizer assets are distributed separately under their respective terms.
