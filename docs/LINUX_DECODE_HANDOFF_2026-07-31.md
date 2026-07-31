# Linux decode optimization handoff — 2026-07-31

Purpose: continue the current Windows conversation in a fresh Linux Codex session on the same RTX 5080 Laptop,
reproduce the latest gem16/llama.cpp screen, add vLLM, establish a qualified Linux parent, and then execute
`DECODE_OPTIMIZATION_PLAN.md`.

This handoff is documentation only. It does not authorize a commit, push, engine change, benchmark run, model
download, or deletion by itself.

## User decisions that must be preserved

- Use the local RTX 5080 Laptop GPU with approximately 16 GB VRAM.
- Run the optimization program under Linux rather than continuing performance work under Windows.
- Keep gem16 on its direct mixed FP8/NVFP4 checkpoint and checkpoint FP8 KV cache.
- Use Q8_0 K and V cache for llama.cpp; do not use Q4 KV.
- Do not use the Reddit user's model. The Reddit command was context only.
- The primary comparison is the best valid current llama.cpp configuration with a model comparable in size and
  quality to gem16's checkpoint.
- Add vLLM under Linux as a direct-checkpoint performance/headroom reference.
- Optimize ordinary decode first, especially its long-context degradation, then optimize MTP.
- The hard fixed-D2 16K target is at least 64.82 effective verified output token/s; matching a stale, slower
  baseline is not sufficient.
- Gem16 MTP must remain exactly identical to gem16 ordinary Target output. Proposed tokens are never throughput.
- Prefill optimization resumes after decode gates pass; prefill must not be silently regressed.

## Repository state at handoff

Windows repository root was `C:\Development\gem16gb`.

Base HEAD before the documentation changes:

```text
5501b522e6f4600368f2afde2196bb12dfbc7963
Refresh README for desktop app, server, and multimodal workflow
```

Documentation changes expected to be committed and pushed by the user before reboot:

```text
README.md
docs/BENCHMARKING.md
docs/DECISIONS.md
docs/DECODE_OPTIMIZATION_PLAN.md
docs/LINUX_DECODE_HANDOFF_2026-07-31.md
docs/MTP.md
docs/ROADMAP.md
```

In the Linux session, start from the user's resulting commit, record its full SHA, and require a clean worktree
before generating a parent benchmark. Do not assume the SHA remains `5501b52` after the documentation commit.

Read, in order:

1. `AGENTS.md`;
2. this handoff;
3. `docs/DECODE_OPTIMIZATION_PLAN.md`;
4. `docs/BENCHMARKING.md`;
5. `docs/MTP.md`;
6. relevant recent entries in `docs/DECISIONS.md` and `docs/PERFORMANCE_LEDGER.md`.

## Reference hardware, checkpoints, and toolchain

The same physical machine is recorded in `toolchains/blackwell16gb.lock`:

```text
GPU: NVIDIA GeForce RTX 5080 Laptop GPU
UUID: GPU-93070293-2184-eb69-555b-1d856a4fdbb8
Compute capability: 12.0
VRAM: 16,303 MiB
Linux: Arch Linux rolling
CUDA: 13.3.73
Driver: 610.43.03
CUTLASS: 4.5.2, db1c288993354c88e551c40c19a8fb93a774a241
```

Verify rather than assume these values after boot. Record kernel, driver, CUDA runtime/toolkit, compiler, CMake,
Ninja, GPU UUID, VBIOS, power limit, clocks, display ownership, temperature, and free VRAM in the new result tree.

Expected checkpoint directories:

```text
models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497
models/checkpoints/google-gemma-4-12B-it-assistant-364bd03
```

Pinned revisions:

```text
Target: b1f649734b34aa5575b03d186abd1b9be3d0d5c4
Assistant: 364bd03c9952e5b7da73665ee30c9eccfc408345
```

Use offline/local files after verifying the locks and checksums. Do not resolve either model from `main`.

## Windows screen to reproduce under Linux

The Windows screen used one warm-up and one measured run. It is an orientation snapshot, not a qualified parent
and not a cross-OS optimization baseline.

### Ordinary decode, 256 output tokens

| Existing context | llama.cpp tok/s | gem16 tok/s | gem16/llama.cpp |
|---:|---:|---:|---:|
| 128 | 40.54 | 38.748 | 0.956x |
| 512 | 46.60 | 39.625 | 0.850x |
| 2,048 | 46.68 | 39.218 | 0.840x |
| 8,192 | 46.14 | 38.259 | 0.829x |
| 16,384 | 44.32 | 36.504 | 0.824x |
| 32,768 | 42.04 | 34.200 | 0.814x |
| 65,536 | 37.25 | 29.862 | 0.802x |

### Prefill screen

| Prompt tokens | llama.cpp tok/s | gem16 tok/s |
|---:|---:|---:|
| 128 | 2,471.52 | 2,421.67 |
| 512 | 3,205.99 | 3,583.31 |
| 2,048 | 2,511.92 | 2,904.81 |
| 8,192 | 3,248.24 | 4,161.34 |
| 16,384 | 3,031.02 | 3,749.51 |
| 32,768 | 2,559.71 | 2,908.81 |
| 65,536 | 1,944.89 | 2,023.18 |

### Fixed Wikipedia 16K D2 MTP, fixed 1,135 output tokens

| Engine | Effective tok/s | Target batches | Accepted/rejected | Mean accepted |
|---|---:|---:|---:|---:|
| llama.cpp | 64.8216 | 519 | 616/419 | 1.1869 |
| gem16 | 53.6426 | 502 | 632/372 | 1.2590 |

Approximate Target-group times are 33.71 ms for llama.cpp and 42.11 ms for gem16. At the current gem16 acceptance
and 502 batches, reaching 64.82 token/s requires at most approximately 34.85 ms/group.

Windows raw MTP artifacts were retained in the ignored path:

```text
benchmarks/results/2026-07-31/5501b52/blackwell16gb-windows-llama-q8-screen/
```

Do not require those ignored files to exist under Linux. The disclosed numbers and hashes below are sufficient to
identify the intended candidate.

## Exact current llama.cpp candidate used on Windows

This is newer than the repository's tracked `benchmarks/baselines/llama_cpp/commit.txt`. Do not accidentally run
the older `846e991...` pin and label it a reproduction.

```text
llama.cpp commit: 000547513f1530346ecd163db8b3e13962949961
Upstream build/version: 10210
CUDA architecture: 120a-real
CMAKE_BUILD_TYPE: Release
GGML_CUDA: ON
GGML_CUDA_FA_ALL_QUANTS: ON
GGML_NATIVE: OFF
```

The source checkpoint's mixed compressed-tensors groups still required a local port of the tracked converter
patch. On Windows the only modified upstream file was `conversion/base.py`. The port:

- accepted mixed `float-quantized` plus `nvfp4-pack-quantized` groups;
- preserved the 144 packed NVFP4 MLP tensors and their block scales;
- dequantized the 184 source FP8 attention weights through the supported converter path;
- marked those FP8-dequantized weights for current upstream's `--fp8-as-q8` output;
- identified NVFP4 tensors by packed `uint8` weight storage rather than scale rank;
- treated any group containing `nvfp4-pack-quantized` as the NVFP4 mixed model.

The tracked patch at
`benchmarks/baselines/llama_cpp/patches/0001-support-mixed-fp8-nvfp4-compressed-tensors.patch` is the semantic
starting point, but it targets the older pin and produces BF16 attention unless the current `--fp8-as-q8` flow is
ported. Apply it manually to a separate current-commit worktree, record the resulting patch as an artifact, and do
not modify a supposedly clean upstream tree invisibly.

Windows target GGUF:

```text
Filename: gemma4-12b-mixed-q8-nvfp4.gguf
Size: 9,366,658,112 bytes
SHA-256: 0fc3dce6d631d1ee5ab5398f621b4bfe50591d01d08339659d554eb91e23091d
Tensor count: 955
NVFP4: 144
Q8_0: 184
F32: 626
BF16: 1
```

Windows assistant GGUF:

```text
Filename: gemma4-12b-it-assistant-bf16.gguf
Size: 861,520,160 bytes
SHA-256: 7b82a9f31fa365fb8ce533424cfad6c5106086f40b3eade4d91d8c5bb63d8224
```

Regenerate both under Linux from the locked checkpoints. Matching hashes are expected for a deterministic
conversion; if they differ, stop and compare converter commit, patch, arguments, metadata, and tensor inventory
before benchmarking.

The screened llama.cpp runtime configuration was:

```text
full GPU residency
split mode: none
flash attention: on
K/V cache: q8_0/q8_0
batch/ubatch: 2048/512
threads: 8
poll: 100
priority: 2
one parallel sequence
offline, no prompt cache
```

The checked-in Wikipedia runner already supplies full GPU layers, no split, Flash Attention, Q8 KV, parallel 1,
batch 2,048, and ubatch 512. It does not currently encode every screened thread/poll/priority choice. Make the
actual Linux command machine-readable. First reproduce the configuration, then perform a bounded llama.cpp
settings sweep; the final competitor must be the fastest stable quality-acceptable setting, not an intentionally
weak reproduction.

## vLLM reference under Linux

Use the repository's pinned direct-checkpoint environment and read
`benchmarks/baselines/vllm/README.md` before running. The retained reference is vLLM 0.25.1, but verify and record
the installed version and patches rather than assuming it.

For the new comparison:

- load the exact target checkpoint directly;
- use FP8 KV cache for the requested comparable long-context run;
- batch one, no CPU offload, no prefix caching, identical token IDs;
- record CUDA Graph/eager mode, memory utilization, chosen FP8/NVFP4 kernels, and any autotuning fallback;
- run ordinary decode and fixed D2 with the official assistant where memory permits;
- treat vLLM MTP as a performance/headroom characterization unless it equals its own ordinary Target output.

vLLM does not replace llama.cpp as the hard competitive gate. It provides a direct-format reference that can show
whether gem16's remaining gap is in its custom kernels or inherent to the workload/format.

## Required Linux continuation sequence

### 1. Establish a clean and recorded environment

```bash
git status --short
git rev-parse HEAD
nvidia-smi
nvcc --version
cmake --version
ninja --version
```

Close unrelated GPU workloads. Record display ownership, power mode/limit, clocks, temperature, driver persistence,
and whether clocks can be locked. Use the same conditions and engine order policy for every adjacent comparison.

### 2. Build and validate current gem16

```bash
cmake --preset blackwell-release
cmake --build --preset blackwell-release --parallel
ctest --preset blackwell-release
build/Linux/blackwell-release/bin/gem16-run --print-kernel-capabilities
```

Also run the required SASS/native-path verification documented in `docs/CORRECTNESS.md`. Do not benchmark if SM120a
FP8/NVFP4 paths are missing or a fallback is reported.

### 3. Build the exact current llama.cpp candidate

Create a separate ignored worktree at commit `000547513...`, configure `120a-real`, CUDA, all-quant Flash Attention,
and native-off as listed above, then build at least `llama-server`, `llama-bench`, `llama-cli`, and GGUF inspection
tools. Save the full configure command, cache, build log, version, and SASS/native NVFP4 proof.

Port and save the mixed converter patch, convert with current upstream's `--fp8-as-q8`, regenerate the assistant,
and export the tensor inventory. Reject CPU offload or any target tensor type inconsistent with the 144/184 split.

### 4. Recreate the exact 16K workload

Use the checked-in preparation command from
`benchmarks/baselines/wikipedia_summary_16k/README.md`. Verify 16,384 prompt IDs and this little-endian uint32 hash:

```text
d07ad4d805944f0b87869da0c5bb44d99e8c43c0eb57d05a108ad80a6abb51a8
```

Use the same generated workload JSON for gem16, llama.cpp, and vLLM.

### 5. Repeat the short Linux overview before profiling

Run one warm-up and one measured run for the prefill and ordinary-decode matrices plus fixed 16K D2. Add vLLM at
every context it can support without offload under the disclosed FP8-KV policy. This is the direct Linux analogue
of the Windows overview and should finish before any kernel work.

Store under a new path such as:

```text
benchmarks/results/<date>/<linux-git-sha>/blackwell16gb-linux-decode-refresh/
```

Retain `system.json`, commands, model inventories/checksums, raw JSONL, output hashes, server logs, and a concise
screen summary. Never overwrite the Windows result or an older Linux result.

### 6. Establish the qualified Linux parent

If the screen is correct and stable, run 3 warm-ups and 10 measured repetitions for:

- gem16 and llama.cpp ordinary decode at 128/512/2K/8K/16K/32K/64K, 256 output tokens;
- gem16 and llama.cpp prefill at the same prompt points, for regression tracking;
- fixed 16K/1,135-output D2 for gem16 and llama.cpp;
- vLLM at the common fitting points as a separately labeled reference.

Alternate engine order where practical and capture continuous power, clocks, temperature, peak/steady VRAM, TTFT,
median/p95/p99 ITL, accepted/rejected/proposed drafts, Target batches, and output hashes.

The current `tools/qualify_mtp.py` still encodes the historical 50/55 token/s status fields. Before using it for
the final new gate, update those fields/tests/schema deliberately to 64.82 and the moving llama.cpp parity rule.
Do not mislabel a 50-token/s pass as completion. That code change belongs to the Linux implementation session, not
this documentation-only handoff.

### 7. Profile before changing kernels

Follow Phase 0 of `DECODE_OPTIMIZATION_PLAN.md`. The current production CUDA Graph must remain the throughput path,
but profiling must expose child-kernel costs. Attribute at least 90% of ordinary token and D2 Target-group GPU time
at 512, 16K, and 64K. Capture the actual llama.cpp attention and MMQ dispatches and the vLLM direct-format kernels.

Only after the qualified parent and attribution exist should implementation begin with global decode attention.

## Known diagnosis and constraints

- Gem16's ordinary ratio worsens monotonically with context; the strongest first hypothesis is the eight global
  attention layers, not MTP acceptance.
- Gem16 global attention uses 512-token splits, FP32 partial output/LSE, and a separate merge. It has 32 splits at
  16K and 128 at 64K per active query group.
- The forty local layers cap at a 1,024-token window but create a large constant floor.
- Current MTP accepts more than llama.cpp yet loses, proving per-Target-group execution is the immediate issue.
- Historical profiles suggest attention plus projections dominate; graph/control-only changes have been small.
- A Windows Nsight capture saw outer graph launches but failed to attribute current child kernels. Do not profile
  prefill accidentally and call it decode evidence.

Do not repeat unchanged rejected experiments: naive BF16/TF32 T=1 Tensor-Core attention, ordinary fused Gate/Up,
rounded projection epilogues, combined residual/MLP quantization, verifier-suffix graph, or persistent duplicate
weights. `DECODE_OPTIMIZATION_PLAN.md` records the conditions for a materially new retry.

## Expected first Linux-session deliverable

Before implementing an optimization, report:

1. exact Linux repository/toolchain/GPU state;
2. gem16 build/test/native-path status;
3. exact llama.cpp commit, build flags, converter patch, GGUF hashes/inventory, and residency;
4. verified shared workload hash;
5. the one-warm-up/one-run gem16/llama.cpp/vLLM overview;
6. differences from the Windows orientation screen, labeled as OS/runtime observations rather than code gains;
7. the proposed qualified-run and profiling commands.

Then proceed automatically to the qualified parent and Phase-0 profiling unless correctness, conversion, residency,
or permission blocks the work.

## Suggested opening request for the new Codex session

```text
Lies AGENTS.md und docs/LINUX_DECODE_HANDOFF_2026-07-31.md vollständig und führe die dort beschriebene Linux-
Übergabe aus. Baue und validiere zuerst gem16 und den exakt gepinnten aktuellen llama.cpp-Kandidaten, reproduziere
danach den kurzen 1/1-Benchmark von Windows unter Linux und ergänze vLLM mit dem direkten Checkpoint. Verwende für
llama.cpp ausschließlich Q8_0 KV, für gem16 den Checkpoint-FP8-KV-Pfad, dieselben Token-IDs und keine CPU-Offloads.
Implementiere noch keine Kerneloptimierung, bevor der qualifizierte Linux-Parent und Phase-0-Profiling belastbar
vorliegen. Halte alle Format-, Output- und Timingunterschiede explizit fest.
```
