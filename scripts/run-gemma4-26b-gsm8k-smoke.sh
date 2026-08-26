#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

if [[ ${1:-} == "--help" ]]; then
  cat <<'EOF'
Run a paired Gemma 4 26B GSM8K benchmark (20-question smoke by default).

The script starts the pinned Google QAT Q4_0 reference in llama.cpp, runs the
quality harness, stops it, then repeats the same questions with gem16 and emits
a paired comparison.

Environment overrides:
  OUTPUT_DIR            Result directory (default: date/revision scoped)
  QUALITY_PYTHON        Existing Python 3.10-3.13 environment with sgl-eval
  QUALITY_BOOTSTRAP_PYTHON  Python used to create .venv-quality
  LLAMA_SERVER          Pinned llama-server executable
  Q4_GGUF               Official Google QAT Q4_0 GGUF
  GEM16_SERVER          gem16-server executable
  GEM16_MODEL           Compiled gem16 26B Target directory
  SERVER_PORT           Loopback port reused sequentially (default: 18080)
  STARTUP_TIMEOUT       Server startup timeout in seconds (default: 300)
  NUM_EXAMPLES          Number of GSM8K questions (default: 20; full: 1319)
EOF
  exit 0
fi

if [[ $# -ne 0 ]]; then
  echo "error: unexpected arguments; use --help" >&2
  exit 64
fi

llama_server=${LLAMA_SERVER:-$repo_root/build/Linux/llama_cpp/release/bin/llama-server}
q4_gguf=${Q4_GGUF:-$repo_root/models/checkpoints/google-gemma-4-26b-a4b-it-qat-q4_0-gguf-d1c082b/gemma-4-26B_q4_0-it.gguf}
gem16_server=${GEM16_SERVER:-$repo_root/build/Linux/blackwell-release/bin/gem16-server}
gem16_model=${GEM16_MODEL:-$repo_root/artifacts/raw/m08/qat-hybrid-clean-1}
server_port=${SERVER_PORT:-18080}
startup_timeout=${STARTUP_TIMEOUT:-300}
num_examples=${NUM_EXAMPLES:-20}
if [[ ! $num_examples =~ ^[1-9][0-9]*$ || $num_examples -gt 1319 ]]; then
  echo "error: NUM_EXAMPLES must be an integer in [1, 1319]" >&2
  exit 64
fi
revision=$(git rev-parse --short=12 HEAD)
run_date=$(date -u +%F)
if [[ $num_examples -eq 20 ]]; then
  default_run_name=local-gsm8k-smoke20
elif [[ $num_examples -eq 1319 ]]; then
  default_run_name=local-gsm8k-full1319
else
  default_run_name=local-gsm8k-${num_examples}
fi
output_dir=${OUTPUT_DIR:-$repo_root/benchmarks/results/$run_date/$revision/$default_run_name}
reference_output=$output_dir/reference-q4-gsm8k${num_examples}.json
candidate_output=$output_dir/gem16-gsm8k${num_examples}.json
comparison_output=$output_dir/comparison-gsm8k${num_examples}.json

for required in "$llama_server" "$q4_gguf" "$gem16_server" "$gem16_model"; do
  if [[ ! -e $required ]]; then
    echo "error: required path does not exist: $required" >&2
    exit 1
  fi
done
if [[ ! -x $llama_server || ! -x $gem16_server ]]; then
  echo "error: server binaries must be executable" >&2
  exit 1
fi
if ! "$llama_server" --version 2>&1 | grep -q '0b14b87d7'; then
  echo "error: LLAMA_SERVER is not the pinned M19 llama.cpp revision 0b14b87d7" >&2
  exit 1
fi

quality_python=${QUALITY_PYTHON:-$repo_root/.venv-quality/bin/python}
if [[ ! -x $quality_python ]]; then
  bootstrap_python=${QUALITY_BOOTSTRAP_PYTHON:-}
  if [[ -z $bootstrap_python ]]; then
    for candidate in \
      python3.13 python3.12 python3.11 python3.10 \
      /home/sebastian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3; do
      if command -v "$candidate" >/dev/null 2>&1; then
        bootstrap_python=$(command -v "$candidate")
        break
      fi
    done
  fi
  if [[ -z $bootstrap_python || ! -x $bootstrap_python ]]; then
    echo "error: Python 3.10-3.13 is required; set QUALITY_BOOTSTRAP_PYTHON" >&2
    exit 1
  fi
  echo "Creating pinned quality environment with $bootstrap_python"
  "$bootstrap_python" -m venv "$repo_root/.venv-quality"
  "$repo_root/.venv-quality/bin/python" -m pip install \
    -r "$repo_root/tools/requirements-quality.txt"
  quality_python=$repo_root/.venv-quality/bin/python
fi

mkdir -p "$output_dir"
server_pid=

stop_server() {
  if [[ -z ${server_pid:-} ]]; then
    return
  fi
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 100); do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL "$server_pid" 2>/dev/null || true
    fi
  fi
  wait "$server_pid" 2>/dev/null || true
  server_pid=
}

trap stop_server EXIT INT TERM

wait_for_server() {
  local log_path=$1
  local deadline=$((SECONDS + startup_timeout))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "error: server exited during startup; log tail:" >&2
      tail -80 "$log_path" >&2 || true
      return 1
    fi
    if curl --silent --show-error --fail --max-time 2 \
      "http://127.0.0.1:$server_port/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "error: server did not become healthy within ${startup_timeout}s; log tail:" >&2
  tail -80 "$log_path" >&2 || true
  return 1
}

result_complete() {
  local path=$1
  [[ -f $path ]] && "$quality_python" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["status"] == "complete" else 1)' \
    "$path"
}

run_quality() {
  local backend=$1
  local model_name=$2
  local output_path=$3
  "$quality_python" "$repo_root/tools/benchmark_quality.py" \
    --benchmark gsm8k \
    --backend "$backend" \
    --base-url "http://127.0.0.1:$server_port/v1" \
    --model "$model_name" \
    --num-examples "$num_examples" \
    --repeats 1 \
    --threads 1 \
    --reasoning none \
    --generation checkpoint \
    --max-tokens 512 \
    --seed 0 \
    --resume \
    --output "$output_path"
}

if ! result_complete "$reference_output"; then
  reference_log=$output_dir/reference-server-$(date -u +%H%M%S).log
  echo "Starting pinned llama.cpp Q4 reference; log: $reference_log"
  "$llama_server" \
    -m "$q4_gguf" \
    -ngl 99 \
    -ot token_embd.weight=CUDA0 \
    -fa on \
    -ctk q8_0 \
    -ctv q8_0 \
    -c 32768 \
    -np 1 \
    -a google-gemma-4-26b-qat-q4_0 \
    --host 127.0.0.1 \
    --port "$server_port" \
    >"$reference_log" 2>&1 &
  server_pid=$!
  wait_for_server "$reference_log"
  echo "Running $num_examples GSM8K questions against the Q4 reference"
  run_quality openai google-gemma-4-26b-qat-q4_0 "$reference_output"
  stop_server
else
  echo "Reference result is already complete: $reference_output"
fi

if ! result_complete "$candidate_output"; then
  candidate_log=$output_dir/gem16-server-$(date -u +%H%M%S).log
  echo "Starting gem16 26B Target; log: $candidate_log"
  "$gem16_server" \
    --model "$gem16_model" \
    --model-name gem16-gemma4-26b-a4b \
    --host 127.0.0.1 \
    --port "$server_port" \
    --max-context 32768 \
    --max-sessions 1 \
    >"$candidate_log" 2>&1 &
  server_pid=$!
  wait_for_server "$candidate_log"
  echo "Running the same $num_examples GSM8K questions against gem16"
  run_quality gem16 gem16-gemma4-26b-a4b "$candidate_output"
  stop_server
else
  echo "gem16 result is already complete: $candidate_output"
fi

if ! result_complete "$comparison_output"; then
  if [[ -e $comparison_output ]]; then
    echo "error: existing comparison is not complete: $comparison_output" >&2
    exit 1
  fi
  "$quality_python" "$repo_root/tools/compare_quality_benchmarks.py" \
    --reference "$reference_output" \
    --candidate "$candidate_output" \
    --output "$comparison_output"
else
  echo "Comparison is already complete: $comparison_output"
fi

echo "Paired GSM8K benchmark complete"
echo "Results: $output_dir"
echo "Comparison: $comparison_output"
