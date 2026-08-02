# Linux max-power cross-engine D2 MTP characterization

This directory retains the machine-readable summary behind the prominent README result. It is a reproducible
development characterization, not a claim of exact tensor-format or output-quality parity.

## Result

On commit `4b237b16366b1a9ee2cd339f0549e06a7cfc69aa`, one RTX 5080 Laptop GPU ran the same exact 16,384-token
Wikipedia prompt followed by 1,135 fixed greedy output positions. Every engine used batch one, fixed D2 MTP, three
warm-ups, and ten measured repetitions. Linux used the firmware `max-power` profile with `nvidia-powerd` active;
the GPU dynamically reached its 175 W ceiling.

| Engine | Prefill tok/s | TTFT | Effective D2 MTP tok/s | ITL |
|---|---:|---:|---:|---:|
| vLLM 0.25.1 | **6,308.53** | **2,597.12 ms** | 81.96 | 12.201 ms |
| gem16 | 5,315.11 | 3,082.53 ms | **85.26** | **11.729 ms** |
| llama.cpp 10210 | 3,947.45 | 4,150.53 ms | 84.18 | 11.879 ms |

The full medians, means, distributions, MTP counters, telemetry summary, configuration, and limitations are in
[`characterization.json`](characterization.json). Raw JSON, console logs, commands, system state, and 200 ms
telemetry are retained under
`benchmarks/results/2026-08-02/4b237b1/blackwell16gb-linux-maxpower-cross-engine-mtp-prefill-refresh/`.

## Reproduce

Prepare gem16 and both locked checkpoints:

```bash
./scripts/build.sh --cuda --test
python tools/fetch_model.py
python tools/fetch_model.py --lock models/gemma4-12b-mtp-assistant.lock.json
```

Prepare the pinned vLLM 0.25.1 environment described in
[`../vllm/README.md`](../vllm/README.md), including its graph-safe Gemma 4 suppression patch. The benchmark script
rejects an environment unless it has the recorded vLLM, Torch, Transformers, and compressed-tensors versions and
the patch.

Build and convert the pinned llama.cpp candidate:

```bash
./benchmarks/baselines/llama_cpp/build.sh
./benchmarks/baselines/llama_cpp/prepare-patched-source.sh

TARGET=$(python -c 'from tools.hf_cache import default_target_model; print(default_target_model())')
ASSISTANT=$(python -c 'from tools.hf_cache import default_assistant_model; print(default_assistant_model())')

./benchmarks/baselines/llama_cpp/convert-patched.sh \
  "$TARGET" build/Linux/llama_cpp/gemma4-12b-mixed-q8-nvfp4.gguf \
  --fp8-as-q8

LLAMA_CPP_SOURCE=third_party/cache/llama.cpp \
LLAMA_CPP_CONVERT_PYTHON=third_party/cache/unsloth-nvfp4-env/bin/python \
./benchmarks/baselines/llama_cpp/convert.sh \
  "$ASSISTANT" build/Linux/llama_cpp/gemma4-12b-it-assistant-bf16.gguf \
  --outtype bf16
```

On the reference Lenovo laptop, enable Dynamic Boost and the benchmark power profile:

```bash
sudo systemctl enable --now nvidia-powerd.service
echo max-power | sudo tee /sys/firmware/acpi/platform_profile
```

Then run the complete comparison:

```bash
./scripts/benchmark-cross-engine-mtp.sh
```

The script refuses to overwrite a prior result. It validates checkpoint locks, vLLM versions and patch, llama.cpp
version and GGUF checksums, power state, and absence of unrelated CUDA work. It writes raw per-run JSON, console
logs, external 200 ms GPU telemetry, system information, and a concise `summary.json` below
`benchmarks/results/<date>/<git-sha>/`.

Use `--allow-uncontrolled-power` on hardware without Linux `platform_profile`/`nvidia-powerd`, but do not compare
that output directly with this 175 W reference result without disclosing the power difference.

## Comparison limitations

- gem16 and vLLM load the direct mixed FP8/NVFP4 checkpoint and use FP8 KV. The patched llama.cpp GGUF preserves
  NVFP4 MLP tensors but maps source FP8 attention tensors to Q8_0 and uses Q8_0 KV.
- Prefill timing boundaries differ. In particular, llama.cpp reports a narrower native prompt-processing boundary.
- All outputs are deterministic within each engine, but their token hashes differ. Prior ordinary-target gates show
  that external vLLM and llama.cpp MTP do not retain their own ordinary greedy sequence; only gem16 enforces exact
  ordinary-Target identity in this comparison path.
- Proposed assistant tokens are never counted as output throughput. The table reports only target-verified output.
