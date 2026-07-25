# gem16gb

`gem16gb` is an early-stage C++20/CUDA inference engine for high-performance inference on NVIDIA GPUs with
approximately 16 GB of VRAM. The first model is the mixed FP8/NVFP4
`unsloth/gemma-4-12b-it-NVFP4` checkpoint, and the first optimized backend is Blackwell SM120/SM120a.

## What works

- The model repository is pinned to commit `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`, including file sizes,
  SHA-256 digests, Git object IDs, and Xet/LFS identities.
- `gem16gb-inspect` memory-maps a single Safetensors file or indexed shards, validates offsets and byte lengths,
  parses the compressed-tensors quantization schema, classifies all 1,389 tensors in the pinned snapshot, and
  exports JSON.
- Host parser tests work on Linux and Windows without CUDA. Linux also has opt-in ASan/UBSan builds.
- `gem16gb-bench memory` builds an aligned, deterministic base arena from the real text-only tensor inventory and
  reports the required separate K/V cache plus the one-state diagnostic lower bound for every context profile.
- The exact host NVFP4 codec covers E2M1, E4M3FN, dynamic-local activation quantization, compressed-tensors global
  divisors, and a binary64 projection oracle with pinned-checkpoint byte fixtures.
- The CUDA build contains an explicit correctness-only W4A4 projection and an experimental direct-source SM120a
  projection. A complete real-checkpoint Layer-0 characterization now composes FP8 local attention and the NVFP4
  MLP without a host roundtrip or persistent weight repack. It is a correctness characterization, not yet a
  trusted-hidden-state or performance qualification.
- A direct-source packed-NVFP4 SIMT/GEMV alternative and a closed SM120a Gate/Up/GELU operator are available for
  controlled characterization. The SIMT path lost decisively to native MMA, while Gate/Up fusion remains opt-in
  through `--enable-fused-gate-up` because its small isolated gain did not produce a stable Windows end-to-end
  median improvement.
- A CUDA-only, batch-one greedy characterization now loads the complete text model into one weight arena, executes
  all 48 layers with separate K/V caches, applies the tied BF16 output head and exact logit softcap, and selects the
  token on the GPU. The default applies the checkpoint's static E4M3 FP8 K/V scales; an explicit BF16 correctness
  mode remains available. Both modes run with zero fallbacks and no allocation in the token loop. Checkpoint EOS
  and suppressed-token controls are applied explicitly.
- `gem16gb-chat` is a pure C++ application. It loads the checkpoint's `tokenizer.json`, performs native
  byte-fallback BPE encode/decode, enforces the pinned `chat_template.jinja` contract, and sources EOS/suppressed
  tokens from `generation_config.json`.

Sampling, persistent chat sessions, and benchmark-qualified inference do **not** work yet. Normal greedy decode
now replays the complete 48-layer forward pass as one CUDA Graph. A pinned host control record supplies the current
token, position, and suppression count to device-side RoPE, KV append, ring selection, attention, and argmax logic;
graph capture and instantiation remain outside the token loop. The 96 position-independent per-layer graphs remain
available to diagnostic paths that request hidden states or logits. The default cache stores one physical E4M3FN
byte per K/V value and dequantizes with the checkpoint's per-layer BF16 scales during attention. It is still an
unfused correctness kernel rather than a performance result. The exact-blue greedy gate passes, while the longer
sky gate currently diverges from vLLM/llama.cpp at its third generated token. Unsupported modes fail visibly and
never fall back to a higher precision path.

## Build on Linux

CMake 3.28+, Ninja, and a C++20 compiler are required.

```bash
./scripts/build.sh --host --test
```

For the CUDA capability probe with the pinned local toolkit:

```bash
./scripts/build.sh --cuda --test
build/Linux/blackwell-release/bin/gem16gb-run --print-kernel-capabilities
```

The CUDA build targets only `120a`. Its experimental projection disassembles to
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`, but the capability report deliberately remains
`native_nvfp4_kernels=false` until real-shape, layer, logit, and benchmark gates pass.

## Greedy inference characterization

The first end-to-end path accepts token IDs so tokenizer behavior cannot hide model errors. It is deliberately
reported as `status=characterization` and `benchmark_qualified=false`:

```bash
build/Linux/blackwell-release/bin/gem16gb-run \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --input-token-ids 2,105,2364,107,40654,607,7121,506,3658,3730,236761,106,107,105,4368,107,100,45518,107,101 \
  --max-tokens 2 \
  --max-context 32 \
  --greedy
```

The checkpoint-declared FP8 K/V semantics are the default. Use `--kv-cache bf16` only for the explicitly labeled
BF16 correctness comparison. `--projection-path reference` selects the slow CUDA scalar projections; it is never
an automatic fallback. `--enable-fused-gate-up` enables the experimental closed native Gate/Up/GELU operator and
is reported explicitly as `fused_gate_up` in result JSON. Native prefill fuses causal score, softmax, and value
reduction by default without changing their arithmetic order. `--disable-fused-prefill-attention` retains the
three-kernel reference path for correctness and A/B performance checks; JSON records the selected path.

Reproduce the committed-token gate without copying token IDs manually:

```bash
python3 tools/validate_inference.py \
  --run build/Linux/blackwell-release/bin/gem16gb-run \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497
```

For a broader, position-aligned comparison, `gem16gb-run` supports diagnostic
teacher forcing. The preceding reference token is fed at every decode
position, while the engine's unmodified greedy prediction and full logits are
recorded. This prevents one early argmax difference from changing all later
inputs:

```bash
python3 tools/teacher_forced_compare.py \
  --run build/Linux/blackwell-release/bin/gem16gb-run \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --golden tests/golden/vllm-gemma4-12b-nvfp4-correctness-v1-fp8.json \
  --kv-cache fp8 \
  --output build/teacher-forced-fp8.json
```

The committed suite contains 12 chats and 127 FP8-reference positions. The
current engine agrees with vLLM FP8 at 118/127 Top-1 positions (92.9%); the
vLLM Top-1 is in the engine Top-5 at all positions. With both engines using
BF16 K/V, agreement rises to 127/131 (96.9%). These are diagnostic
measurements, not an accepted correctness tolerance or performance result.

## Command-line chat characterization

Run the native C++ application directly:

```bash
build/Linux/blackwell-release/bin/gem16gb-chat \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497
```

Generated text streams token-by-token in interactive mode. Enter `/quit` to exit. Add `--thinking` to enable the
checkpoint template's thinking form. The current characterization reloads the model and reprocesses the full
conversation on every turn; persistent sessions follow after the model-wide correctness gate. For an auditable
one-turn result:

```bash
build/Linux/blackwell-release/bin/gem16gb-chat \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --message "Reply with exactly the word blue." --max-tokens 8 --max-context 64 --json
```

`--render-only --json` validates prompt rendering and token IDs without loading CUDA weights. No Python interpreter,
Transformers installation, server, or subprocess is involved in the chat application.

For prompt-derived layer diagnostics, add `--dump-state <file> --dump-state-position <position>`. The capture
preallocates pinned storage and writes only after generation. `tools/dump_vllm_states.py` emits the same
self-describing format from the pinned reference runtime, and `tools/compare_states.py` compares attention context,
layer intermediates, final hidden states, K, and V.

## Build on Windows

Use PowerShell from a regular terminal; the helper discovers Visual Studio 2022 Build Tools with `vswhere`, imports
the x64 MSVC environment, and uses Ninja. CMake 3.28+, Ninja, and the Visual Studio C++ workload are required.

```powershell
.\scripts\build.ps1 -Test
```

For the CUDA capability probe, install the pinned CUDA toolkit. The helper uses `CUDA_PATH` when set and otherwise
discovers toolkits installed in NVIDIA's standard Windows location:

```powershell
.\scripts\build.ps1 -Cuda -Test
.\build\Windows\blackwell-release\bin\gem16gb-run.exe --print-kernel-capabilities
```

The target layout is the same on both operating systems, while CMake caches stay isolated under `build/Linux` and
`build/Windows`. The `host-sanitize` preset is Linux-only because it currently uses GCC/Clang ASan and UBSan.
The validated Windows development toolchain is recorded in `toolchains/windows-blackwell16gb.lock`; the Linux
reference remains in `toolchains/blackwell16gb.lock`.

## Download the pinned checkpoint

The checkpoint is about 9.34 GB. `HF_TOKEN` is optional for this public repository.

```bash
python3 tools/fetch_model.py \
  --destination models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497
```

On Windows, use `python` instead of `python3`; all Python tools use `pathlib` and accept native Windows paths.
The trusted vLLM reference runtime remains Linux-only because upstream vLLM has no supported native Windows
runtime; run those reference-generation and characterization commands on Linux rather than changing their
semantics.

The downloader resumes partial files, verifies sizes and SHA-256 digests, and never imports or executes code
from the model repository.

## Inspect

```bash
build/Linux/host-debug/bin/gem16gb-inspect \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --validate \
  --json manifest.json
```

The equivalent Windows command is:

```powershell
.\build\Windows\host-debug\bin\gem16gb-inspect.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --validate `
  --json .\manifest.json
```

## Plan memory

Final K and V states require separate storage because their normalization and RoPE paths differ. Select it
explicitly; the JSON result retains the one-state byte count only as an audit lower bound:

```powershell
.\build\Windows\host-debug\bin\gem16gb-bench.exe memory `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --profile long `
  --kv-storage separate
```

The current base plan covers immutable text weights, scales, and KV payload. Activation, graph, sampling, kernel,
and prefill workspaces remain explicitly marked as unplanned rather than being estimated without an execution plan.

## Characterize decode

The CUDA benchmark keeps the model loaded across warm-ups and measured repetitions and emits raw inter-token
latencies plus summary statistics as JSON:

```powershell
.\build\Windows\blackwell-release\bin\gem16gb-bench.exe decode `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --context 128 `
  --tokens 256 `
  --warmups 3 `
  --repetitions 10 `
  --enable-fused-gate-up
```

This is currently a development characterization, not an accepted competitive result. The hybrid KV cache keeps
1,024 positions in each local-attention ring and grows the eight global-attention layers through the requested
context, up to the checkpoint's 262,144-position contract. A complete greedy-forward CUDA Graph and retained static
per-layer diagnostic graphs are prepared before the token loop and enabled by default. Use
`--disable-decode-graphs` for a direct arithmetic and performance A/B; result JSON records both the selected path
and measured graph-associated device-memory growth.

On Windows, a short Nsight Systems prefill/decode capture with NVTX phase ranges can be collected and summarized
without opening the GUI:

```powershell
.\tools\profile_windows.ps1 `
  -Model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  -Context 128 `
  -DecodeTokens 16
```

Reports and exported SQLite databases are written below the selected build directory's `profiles` folder and stay
outside version control.

Prompt ingestion uses a native 32-token chunk plan by default. Its FP8 attention and NVFP4 MLP projections use real
16-row by 8-column MMA tiles that reuse checkpoint weight fragments across prompt tokens while retaining causal
local/global attention. Its causal attention combines score, softmax, and value phases into one block while
preserving the reference reduction order. Pass `--disable-fused-prefill-attention` for the unfused native A/B path,
or `--serial-prefill` to
`gem16gb-run` or `gem16gb-bench decode` for the retained one-token correctness bridge.

## Validate real-checkpoint layer assembly

After a Blackwell CUDA build, run all three real-checkpoint characterization gates with one cross-platform tool:

```powershell
python .\tools\validate_layer_checkpoint.py `
  --bench .\build\Windows\blackwell-release\bin\gem16gb-bench.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --output .\build\Windows\blackwell-release\layer-checkpoint-validation.json
```

This executes Layer-0 local attention, Layer-5 full attention, and the complete Layer-0 decoder characterization.
It rejects fallbacks, persistent repacks, missing layer-scalar execution, host roundtrips between sublayers, and
divergent NVFP4 activation bytes. It deliberately does not invent a model-quality tolerance before the trusted
hidden-state distribution exists.

## Hardware and limitations

Blackwell is the first implementation target, not a permanent board-specific product boundary. Every benchmark
records the exact board, power, clocks, driver, and VRAM independently, while engine planning is expressed in terms
of the 16 GB CUDA hardware class. No engine performance claim exists yet.

Project code is Apache-2.0. Checkpoint use is governed separately by its model card and linked Gemma terms.
