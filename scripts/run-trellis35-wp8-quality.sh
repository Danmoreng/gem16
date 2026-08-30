#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
quality_python=${QUALITY_PYTHON:-$repo_root/.venv-quality/bin/python}
server=${GEM16_SERVER:-$repo_root/build/blackwell-release/bin/gem16-server}
candidate=${GEM16_MODEL:-$repo_root/models/checkpoints/google-gemma-4-26b-a4b-it-trellis35-w4a8-wp2}
control=${GEM16_CONTROL_MODEL:-$repo_root/artifacts/raw/m25/hf-qualified-checkpoints-final-test-2026-08-27/target}
assistant=${GEM16_ASSISTANT_MODEL:-$repo_root/artifacts/raw/m25/hf-qualified-checkpoints-final-test-2026-08-27/assistant}
output_dir=${OUTPUT_DIR:-$repo_root/artifacts/raw/trellis35/wp8/quality}
port=${SERVER_PORT:-18080}
base_url=http://127.0.0.1:$port/v1
server_pid=

if [[ ${1:-} == "--help" ]]; then
  echo "Run the frozen Trellis35 WP8 task suite: full GSM8K, full AIME 2026,"
  echo "GPQA Diamond first 20, and a paired NVFP4-control GPQA first 20."
  echo "Results are resumable under OUTPUT_DIR (default: artifacts/raw/trellis35/wp8/quality)."
  echo "Overrides: QUALITY_PYTHON GEM16_SERVER GEM16_MODEL GEM16_CONTROL_MODEL"
  echo "           GEM16_ASSISTANT_MODEL OUTPUT_DIR SERVER_PORT"
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "error: unexpected arguments; use --help" >&2
  exit 64
fi
for required in "$quality_python" "$server"; do
  if [[ ! -x $required ]]; then
    echo "error: executable not found: $required" >&2
    exit 2
  fi
done
for required in "$candidate" "$control" "$assistant"; do
  if [[ ! -d $required ]]; then
    echo "error: checkpoint directory not found: $required" >&2
    exit 2
  fi
done
mkdir -p "$output_dir"

stop_server() {
  if [[ -n $server_pid ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -INT "$server_pid"
    wait "$server_pid" || true
  fi
  server_pid=
}
trap stop_server EXIT INT TERM

start_server() {
  local model=$1
  local name=$2
  local log=$3
  stop_server
  "$server" \
    --model "$model" \
    --assistant-model "$assistant" \
    --mtp-draft-tokens 2 \
    --model-name "$name" \
    --host 127.0.0.1 \
    --port "$port" \
    --max-context 32768 \
    --max-sessions 1 \
    --seed 0 \
    --log-level info >"$log" 2>&1 &
  server_pid=$!
  for ((attempt = 0; attempt < 600; ++attempt)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      wait "$server_pid" || true
      echo "error: server exited during startup; inspect $log" >&2
      exit 2
    fi
    if curl --silent --fail "http://127.0.0.1:$port/ready" >/dev/null; then
      return
    fi
    sleep 1
  done
  echo "error: server did not become ready; inspect $log" >&2
  exit 2
}

run_quality() {
  local benchmark=$1
  local examples=$2
  local output=$3
  "$quality_python" "$repo_root/tools/benchmark_quality.py" \
    --benchmark "$benchmark" \
    --backend gem16 \
    --base-url "$base_url" \
    --num-examples "$examples" \
    --seed 0 \
    --repeats 1 \
    --threads 1 \
    --resume \
    --output "$output"
}

start_server "$candidate" gem16-trellis35-wp8 \
  "$output_dir/candidate-server.log"
run_quality gsm8k 1319 "$output_dir/candidate-gsm8k1319.json"
run_quality aime26 30 "$output_dir/candidate-aime2630.json"
run_quality gpqa 20 "$output_dir/candidate-gpqa20.json"

start_server "$control" gem16-nvfp4-wp8-control \
  "$output_dir/control-server.log"
run_quality gpqa 20 "$output_dir/control-gpqa20.json"
stop_server

"$quality_python" "$repo_root/tools/compare_quality_benchmarks.py" \
  --reference "$repo_root/benchmarks/results/m19-quality-suite/overnight-2026-08-26/01-gsm8k/gem16-gsm8k1319.json" \
  --candidate "$output_dir/candidate-gsm8k1319.json" \
  --output "$output_dir/comparison-gsm8k1319.json"
"$quality_python" "$repo_root/tools/compare_quality_benchmarks.py" \
  --reference "$repo_root/benchmarks/results/m19-quality-suite/overnight-2026-08-26/02-aime26/gem16-aime2630.json" \
  --candidate "$output_dir/candidate-aime2630.json" \
  --output "$output_dir/comparison-aime2630.json"
"$quality_python" "$repo_root/tools/compare_quality_benchmarks.py" \
  --reference "$output_dir/control-gpqa20.json" \
  --candidate "$output_dir/candidate-gpqa20.json" \
  --output "$output_dir/comparison-gpqa20.json"

echo "WP8 task-quality suite complete: $output_dir"
