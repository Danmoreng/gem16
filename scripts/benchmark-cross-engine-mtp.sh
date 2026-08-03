#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

warmups=3
repetitions=10
output_dir=""
allow_uncontrolled_power=false
skip_model_verification=false
# A cold vLLM 0.26 start can launch one memory-heavy NVFP4 CUTLASS compile
# per CPU concurrently. Four jobs keep observed peak host memory safely below
# 64 GiB; internal NVCC parallelism stays at one thread per job. This does not
# alter measured inference once the kernels have been built.
vllm_compile_jobs=4
vllm_nvcc_threads=1

usage() {
  cat <<'EOF'
Usage: scripts/benchmark-cross-engine-mtp.sh [options]

Reproduce the Linux 16K fixed-D2 cross-engine characterization for gem16,
vLLM, and llama.cpp. All engines receive the same 16,384 prompt token IDs,
generate exactly 1,135 target tokens, and use 3 warmups plus 10 measured runs
by default.

Options:
  --warmups N                  Warmup runs per engine (default: 3)
  --repetitions N              Measured runs per engine (default: 10)
  --output DIR                 Unique result directory
  --allow-uncontrolled-power   Do not require max-power + nvidia-powerd
  --skip-model-verification    Skip lock-file checksum verification
  -h, --help                   Show this help

Environment overrides:
  GEM16_MODEL                  Direct target checkpoint directory
  GEM16_ASSISTANT_MODEL        Direct assistant checkpoint directory
  VLLM_PYTHON                  Python containing the pinned patched vLLM
  GEM16_LLAMA_SERVER           Pinned llama-server executable
  GEM16_LLAMA_GGUF             Patched target GGUF
  GEM16_LLAMA_ASSISTANT_GGUF   Official-assistant BF16 GGUF
EOF
}

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: expected a positive integer, got: $1" >&2
    exit 64
  }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --warmups)
      [[ $# -ge 2 ]] || { echo "error: --warmups requires a value" >&2; exit 64; }
      [[ "$2" =~ ^[0-9]+$ ]] || { echo "error: invalid warmup count: $2" >&2; exit 64; }
      warmups="$2"
      shift 2
      ;;
    --repetitions)
      [[ $# -ge 2 ]] || { echo "error: --repetitions requires a value" >&2; exit 64; }
      positive_integer "$2"
      repetitions="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { echo "error: --output requires a value" >&2; exit 64; }
      output_dir="$2"
      shift 2
      ;;
    --allow-uncontrolled-power)
      allow_uncontrolled_power=true
      shift
      ;;
    --skip-model-verification)
      skip_model_verification=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

command -v nvidia-smi >/dev/null || { echo "error: nvidia-smi is required" >&2; exit 2; }
command -v python >/dev/null || { echo "error: python is required" >&2; exit 2; }

commit="$(git rev-parse HEAD)"
short_commit="$(git rev-parse --short=8 HEAD)"
if [[ -z "$output_dir" ]]; then
  output_dir="benchmarks/results/$(date +%F)/${short_commit}/blackwell16gb-linux-maxpower-cross-engine-mtp-$(date +%H%M%S)"
fi
if [[ -e "$output_dir" ]]; then
  echo "error: refusing to overwrite result directory: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"
output_dir="$(realpath "$output_dir")"

workload="$repo_root/benchmarks/prompts/wikipedia-summary-16k.json"
gem16_run="$repo_root/build/Linux/blackwell-release/bin/gem16-run"
vllm_python="${VLLM_PYTHON:-$repo_root/third_party/cache/vllm-0.26.0-env/bin/python}"
llama_server="${GEM16_LLAMA_SERVER:-$repo_root/build/Linux/llama_cpp/release/bin/llama-server}"
llama_gguf="${GEM16_LLAMA_GGUF:-$repo_root/build/Linux/llama_cpp/gemma4-12b-mixed-q8-nvfp4.gguf}"
llama_assistant_gguf="${GEM16_LLAMA_ASSISTANT_GGUF:-$repo_root/build/Linux/llama_cpp/gemma4-12b-it-assistant-bf16.gguf}"

for file in "$workload" "$gem16_run" "$vllm_python" "$llama_server" "$llama_gguf" "$llama_assistant_gguf"; do
  [[ -e "$file" ]] || { echo "error: required benchmark artifact is missing: $file" >&2; exit 2; }
done
[[ -x "$gem16_run" ]] || { echo "error: gem16-run is not executable: $gem16_run" >&2; exit 2; }
[[ -x "$vllm_python" ]] || { echo "error: vLLM Python is not executable: $vllm_python" >&2; exit 2; }
[[ -x "$llama_server" ]] || { echo "error: llama-server is not executable: $llama_server" >&2; exit 2; }

if [[ "$allow_uncontrolled_power" != true ]]; then
  profile_file=/sys/firmware/acpi/platform_profile
  [[ -r "$profile_file" ]] || {
    echo "error: platform profile is unavailable; pass --allow-uncontrolled-power to proceed visibly" >&2
    exit 2
  }
  [[ "$(<"$profile_file")" == "max-power" ]] || {
    echo "error: max-power is required; run: echo max-power | sudo tee $profile_file" >&2
    exit 2
  }
  systemctl is-active --quiet nvidia-powerd.service || {
    echo "error: nvidia-powerd is inactive; run: sudo systemctl enable --now nvidia-powerd.service" >&2
    exit 2
  }
fi

active_compute="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -n "$active_compute" ]]; then
  echo "error: unrelated CUDA compute processes are active:" >&2
  echo "$active_compute" >&2
  exit 2
fi

model="${GEM16_MODEL:-$(python -c 'from tools.hf_cache import default_target_model; print(default_target_model())')}"
assistant_model="${GEM16_ASSISTANT_MODEL:-$(python -c 'from tools.hf_cache import default_assistant_model; print(default_assistant_model())')}"
[[ -f "$model/config.json" ]] || { echo "error: target checkpoint is missing: $model" >&2; exit 2; }
[[ -f "$assistant_model/config.json" ]] || { echo "error: assistant checkpoint is missing: $assistant_model" >&2; exit 2; }

if [[ "$skip_model_verification" != true ]]; then
  python tools/fetch_model.py --verify-only
  python tools/fetch_model.py --lock models/gemma4-12b-mtp-assistant.lock.json --verify-only
fi

"$vllm_python" - <<'PY'
import importlib.metadata
from pathlib import Path
expected = {
    "vllm": "0.26.0",
    "torch": "2.11.0",
    "transformers": "5.14.1",
    "compressed-tensors": "0.17.0",
    "setuptools": "80.10.2",
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"pinned vLLM environment mismatch: expected {expected}, got {actual}")
source = Path(importlib.metadata.distribution("vllm").locate_file(
    "vllm/model_executor/models/gemma4_mtp.py"
))
text = source.read_text(encoding="utf-8")
if "for token_id in self._suppress_token_ids:" not in text:
    raise SystemExit("vLLM Gemma 4 graph-safe suppression patch is not applied")
PY

expected_llama_commit="$(tr -d '[:space:]' < benchmarks/baselines/llama_cpp/commit.txt)"
"$llama_server" --version 2>&1 | grep -q "${expected_llama_commit:0:9}" || {
  echo "error: llama-server is not the pinned ${expected_llama_commit} build" >&2
  exit 2
}
printf '%s  %s\n' \
  '6f90177f6a2d42406d57cfa764eae890b262bfdf71d353bd4827e0488b099896' "$llama_gguf" \
  'b3ab76db11dd1cfbef51925d7dfd6e234325aa86ab3edd7dea42994edb093b65' "$llama_assistant_gguf" \
  | sha256sum --check --status || {
    echo "error: llama.cpp GGUF checksum mismatch" >&2
    exit 2
  }

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "git_commit=$commit"
  echo "git_dirty_entries=$(git status --porcelain | wc -l)"
  echo "platform_profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unavailable)"
  echo "nvidia_powerd=$(systemctl is-active nvidia-powerd.service 2>/dev/null || true)"
  echo "kernel=$(uname -srmo)"
  echo "cuda=$(nvcc --version | tail -1)"
  echo "vllm_max_jobs=$vllm_compile_jobs"
  echo "vllm_torchinductor_compile_threads=$vllm_compile_jobs"
  echo "vllm_flashinfer_nvcc_threads=$vllm_nvcc_threads"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,temperature.gpu,utilization.gpu,pstate --format=csv,noheader
  nvidia-smi -q -d POWER | grep -E 'Current Power Limit|Default Power Limit|Max Power Limit' | head -8
} > "$output_dir/system.txt"
printf 'invocation=' > "$output_dir/commands.txt"
printf '%q ' "$0" --warmups "$warmups" --repetitions "$repetitions" --output "$output_dir" >> "$output_dir/commands.txt"
printf '\n' >> "$output_dir/commands.txt"

monitor_pid=""
cleanup_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}
trap cleanup_monitor EXIT INT TERM

wait_for_cool_gpu() {
  local deadline=$((SECONDS + 180))
  while (( SECONDS < deadline )); do
    local temperature
    temperature="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "$temperature" =~ ^[0-9]+$ ]] && (( temperature <= 50 )); then
      return
    fi
    sleep 2
  done
  echo "warning: GPU did not cool to 50 C before the next engine" >&2
}

run_with_telemetry() {
  local name="$1"
  shift
  wait_for_cool_gpu
  local telemetry="$output_dir/${name}-telemetry.csv"
  printf 'timestamp,power_w,sm_mhz,memory_mhz,temp_c,util_pct,memory_used_mib,pstate\n' > "$telemetry"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,power.draw,clocks.current.graphics,clocks.current.memory,temperature.gpu,utilization.gpu,memory.used,pstate \
        --format=csv,noheader,nounits >> "$telemetry" 2>/dev/null || true
      sleep 0.2
    done
  ) &
  monitor_pid=$!
  set +e
  "$@" 2>&1 | tee "$output_dir/${name}.console.log"
  local status=${PIPESTATUS[0]}
  set -e
  cleanup_monitor
  if (( status != 0 )); then
    echo "error: ${name} benchmark failed with status ${status}" >&2
    exit "$status"
  fi
}

common=(
  --workload "$workload"
  --fixed-output-tokens 1135
  --mtp-draft-tokens 2
  --warmups "$warmups"
  --repetitions "$repetitions"
)

run_with_telemetry gem16 \
  python tools/benchmark_wikipedia_workload.py \
  --engine gem16 --output "$output_dir/gem16.json" \
  --model "$model" --assistant-model "$assistant_model" \
  --executable "$gem16_run" "${common[@]}"

run_with_telemetry vllm \
  env PATH="$(dirname "$vllm_python"):$PATH" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
  MAX_JOBS="$vllm_compile_jobs" NVCC_THREADS="$vllm_nvcc_threads" \
  FLASHINFER_NVCC_THREADS="$vllm_nvcc_threads" \
  TORCHINDUCTOR_COMPILE_THREADS="$vllm_compile_jobs" \
  "$vllm_python" tools/benchmark_wikipedia_workload.py \
  --engine vllm --output "$output_dir/vllm.json" \
  --model "$model" --assistant-model "$assistant_model" \
  --gpu-memory-utilization 0.92 --vllm-kv-cache-dtype fp8 \
  "${common[@]}"

run_with_telemetry llama-cpp \
  python tools/benchmark_wikipedia_workload.py \
  --engine llama-cpp --output "$output_dir/llama-cpp.json" \
  --executable "$llama_server" --gguf "$llama_gguf" \
  --assistant-model "$llama_assistant_gguf" \
  --llama-kv-cache-type q8_0 --llama-threads 8 --llama-poll 100 --llama-prio 0 \
  "${common[@]}"

OUTPUT_DIR="$output_dir" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
engines = {}
for name in ("gem16", "vllm", "llama-cpp"):
    document = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
    summary = document["summary"]
    speculation = summary.get("mtp") or summary.get("speculative")
    engines[name] = {
        "configuration": document["configuration"],
        "prompt_tokens_per_second": summary["prompt_tokens_per_second"],
        "time_to_first_token_ms": summary["time_to_first_token_ms"],
        "decode_tokens_per_second": summary["decode_tokens_per_second"],
        "average_inter_token_latency_ms": summary["average_inter_token_latency_ms"],
        "deterministic_outputs": summary["deterministic_outputs"],
        "output_token_sha256_values": summary["output_token_sha256_values"],
        "mtp": {
            field: speculation[field]["median"]
            for field in (
                "proposed_tokens", "accepted_tokens", "rejected_tokens",
                "mean_accepted_length", "target_batches"
            )
        },
    }

result = {
    "schema_version": 1,
    "status": "controlled_performance_comparison",
    "scope": "same-machine 16K fixed-output greedy D2 MTP comparison",
    "workload": {
        "prompt_tokens": 16384,
        "fixed_output_tokens": 1135,
        "batch_size": 1,
        "mtp_draft_tokens": 2,
        "prompt_token_ids_sha256": "d07ad4d805944f0b87869da0c5bb44d99e8c43c0eb57d05a108ad80a6abb51a8",
    },
    "engines": engines,
    "limitations": [
        "Prefill timing boundaries differ between the three runtimes.",
        "gem16/vLLM use direct FP8/NVFP4 plus FP8 KV; llama.cpp uses patched Q8_0 attention plus Q8_0 KV.",
        "External MTP output is not exact against each runtime's ordinary target output.",
        "Telemetry is retained separately and spans startup, warmups, and measurements.",
    ],
}
(root / "summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

print("\n| Engine | Prefill tok/s | TTFT ms | D2 MTP tok/s | ITL ms |")
print("|---|---:|---:|---:|---:|")
for name in ("vllm", "gem16", "llama-cpp"):
    value = engines[name]
    print(
        f"| {name} | {value['prompt_tokens_per_second']['median']:.2f} | "
        f"{value['time_to_first_token_ms']['median']:.2f} | "
        f"{value['decode_tokens_per_second']['median']:.2f} | "
        f"{value['average_inter_token_latency_ms']['median']:.3f} |"
    )
print(f"\nResults: {root}")
PY
