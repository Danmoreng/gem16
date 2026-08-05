# Cross-engine D2 MTP performance comparisons

This directory retains the machine-readable summary behind the prominent README result. It is a reproducible,
controlled same-machine performance comparison, not a claim of exact tensor-format or output/semantic parity.

## Final-sprint Linux result

The corrected gem16 engine at commit `a819d14c57c54b85b8159be465bea1da0adc5388`, vLLM 0.26.0, and llama.cpp
b10240 (`0b14b87d7c20cb753b94b96854dd7b45306fc696`) ran on one RTX 5080 Laptop GPU. Every engine received the same
exact 16,384-token Wikipedia prompt followed by 1,135 fixed greedy output positions, with batch one, fixed D2 MTP,
three warm-ups, and ten measured repetitions. Linux used the firmware `max-power` profile with `nvidia-powerd`
active; the GPU dynamically reached its 175 W ceiling.

| Engine | Prefill tok/s | TTFT | Effective D2 MTP tok/s | ITL | Sampled peak VRAM |
|---|---:|---:|---:|---:|---:|
| vLLM 0.26.0 | **6,257.37** | **2,618.35 ms** | 82.25 | 12.158 ms | 15,764 MiB |
| **gem16** | 5,866.86 | 2,792.64 ms | **87.66** | **11.408 ms** | 11,746 MiB |
| llama.cpp b10240 | 3,941.23 | 4,157.08 ms | 83.89 | 11.921 ms | 10,630 MiB |

Gem16 decode is 6.57% faster than vLLM and 4.49% faster than llama.cpp. Its median ITL is 6.17% lower than vLLM
and 4.30% lower than llama.cpp. Gem16 prefill is 6.24% below vLLM and 48.86% above llama.cpp. The exact alternating
gem16-only qualification separately produced 47.760 ordinary and 87.423 fixed-D2 tok/s; all 26 warm-up/measured
runs retained one Target hash. Machine-readable data are in
[`characterization-a819d14c.json`](characterization-a819d14c.json) and
[`gem16-qualification-a819d14c.json`](gem16-qualification-a819d14c.json).

The prior `8e86cb38` Linux row reached 89.58 D2 tok/s but lacked two required short-batch verifier BF16 boundaries.
It remains in [`characterization.json`](characterization.json) as historical evidence and is not the corrected
final performance result.

## Windows result

A fresh adjacent Windows 11 run at gem16 commit `1ffabc4` and the same llama.cpp b10240 pin uses Lenovo Max Power,
CUDA 13.3, driver 596.49, and the same 16K/1,135-token fixed-D2 3/10 workload. The engines ran serially after the
GPU cooled to 50 C:

| Engine | Prefill tok/s | TTFT | Effective D2 MTP tok/s | ITL | Sampled peak VRAM |
|---|---:|---:|---:|---:|---:|
| **gem16** | **6,047.04** | **2,709.43 ms** | **90.95** | **10.995 ms** | 11,820 MiB |
| llama.cpp b10240 | 3,940.28 | 4,158.08 ms | 86.77 | 11.524 ms | **10,586 MiB** |

Gem16 is 53.47% faster in prefill and 4.81% faster in decode; TTFT is 34.84% lower and median ITL is 4.59% lower.
Both engines reached approximately 175 W. All measured outputs are deterministic within each engine, and both
output hashes match the corresponding historical `8e86cb38` Linux results. Full distributions, Windows-specific GGUF hashes,
MTP counters, and 200 ms telemetry summaries are in
[`windows-characterization.json`](windows-characterization.json).

The corrected final distributions, MTP counters, telemetry summary, configuration, runtime pins, and limitations
are in [`characterization-a819d14c.json`](characterization-a819d14c.json). Raw JSON, console/server logs, commands,
system state, and 200 ms telemetry are retained under
`benchmarks/results/2026-08-05/a819d14c/blackwell16gb-linux-maxpower-12b-sprint/S08-final/cross-engine/`.
The historical `8e86cb38` data and its disclosed dirty benchmark-pin/script entries remain unchanged in
[`characterization.json`](characterization.json) and the 2026-08-03 raw directory.

## Reproduce

Prepare gem16 and both locked checkpoints:

```bash
./scripts/build.sh --cuda --test
python tools/fetch_model.py
python tools/fetch_model.py --lock models/gemma4-12b-mtp-assistant.lock.json
```

Install the pinned vLLM wheel and apply the audited graph-safe Gemma 4 suppression patch:

```bash
./benchmarks/baselines/vllm/build.sh
```

A cold vLLM 0.26.0 start JIT-builds memory-heavy NVFP4 CUTLASS variants. The benchmark harness limits this to four
compiler jobs with one internal NVCC thread each. On the 64 GiB reference host this replaced an unsafe 10+ process
cold start with a measured maximum of four compilers. Inference timing begins after engine initialization.

Build and convert the pinned llama.cpp candidate:

```bash
./benchmarks/baselines/llama_cpp/build.sh
./benchmarks/baselines/llama_cpp/prepare-patched-source.sh

TARGET=$(python -c 'from tools.hf_cache import default_target_model; print(default_target_model())')
ASSISTANT=$(python -c 'from tools.hf_cache import default_assistant_model; print(default_assistant_model())')

LLAMA_CPP_CONVERT_PYTHON=third_party/cache/vllm-0.26.0-env/bin/python \
./benchmarks/baselines/llama_cpp/convert-patched.sh \
  "$TARGET" build/Linux/llama_cpp/gemma4-12b-mixed-q8-nvfp4.gguf \
  --fp8-as-q8

LLAMA_CPP_SOURCE=third_party/cache/llama.cpp \
LLAMA_CPP_CONVERT_PYTHON=third_party/cache/vllm-0.26.0-env/bin/python \
./benchmarks/baselines/llama_cpp/convert.sh \
  "$ASSISTANT" build/Linux/llama_cpp/gemma4-12b-it-assistant-bf16.gguf \
  --outtype bf16
```

On the reference Lenovo laptop, enable Dynamic Boost and the benchmark power profile:

```bash
sudo systemctl enable --now nvidia-powerd.service
echo max-power | sudo tee /sys/firmware/acpi/platform_profile
```

Then run the complete comparison. A systemd memory scope is recommended on machines without swap so a failed
third-party JIT cannot evict the desktop session:

```bash
systemd-run --user --scope \
  -p MemoryMax=48G -p MemorySwapMax=0 \
  ./scripts/benchmark-cross-engine-mtp.sh
```

Running `./scripts/benchmark-cross-engine-mtp.sh` directly is also supported. The script refuses to overwrite a
prior result. It validates checkpoint locks, vLLM versions and patch, llama.cpp version and GGUF checksums, power
state, and absence of unrelated CUDA work. It writes raw per-run JSON, console logs, external GPU telemetry,
system information, and `summary.json` below `benchmarks/results/<date>/<git-sha>/`.

Use `--allow-uncontrolled-power` on hardware without Linux `platform_profile`/`nvidia-powerd`, but do not compare
that output directly with this 175 W reference result without disclosing the power difference.

## Comparison limitations

- Gem16 and vLLM load the direct mixed FP8/NVFP4 checkpoint and use FP8 KV. The patched llama.cpp GGUF preserves
  NVFP4 MLP tensors but maps source FP8 attention tensors to Q8_0 and uses Q8_0 KV.
- Prefill timing boundaries differ. In particular, llama.cpp reports a narrower native prompt-processing boundary.
- All outputs are deterministic within each engine, but their token hashes differ. External vLLM and llama.cpp MTP
  do not retain their own ordinary greedy sequence; only gem16 enforces exact ordinary-Target identity here.
- vLLM FlashInfer autotuning reported GPU-OOM fallbacks, and its 8,192-token NVFP4 prefill shape remained outside
  the tuned bucket range. These fallbacks are visible in the retained log.
- Proposed assistant tokens are never counted as output throughput. The table reports only target-verified output.
