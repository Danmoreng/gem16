# llama.cpp baseline

Upstream is pinned to release b10240, commit `0b14b87d7c20cb753b94b96854dd7b45306fc696`, published on
2026-08-03 (version 10240). A Linux cross-engine characterization has been captured; no exact-format or
quality-accepted baseline exists yet.

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
