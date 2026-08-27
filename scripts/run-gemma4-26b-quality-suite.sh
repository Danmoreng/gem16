#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
paired_runner=$script_dir/run-gemma4-26b-gsm8k-smoke.sh

if [[ ${1:-} == "--help" ]]; then
  cat <<'EOF'
Run the complete paired Gemma 4 26B task-quality suite in this order:

  1. GSM8K: llama.cpp Q4, then gem16 (1,319 questions)
  2. AIME 2026: llama.cpp Q4, then gem16 (30 questions)
  3. GPQA Diamond: llama.cpp Q4, then gem16 (198 questions)

Every question is journaled durably. Rerun the same command to skip completed
engines/tasks and resume the interrupted engine at its next unfinished question.

Environment overrides:
  OUTPUT_DIR            Stable suite result root (default: revision scoped)
  QUALITY_PYTHON        Existing Python 3.10-3.13 environment with sgl-eval
  LLAMA_SERVER          Pinned llama-server executable
  Q4_GGUF               Official Google QAT Q4_0 GGUF
  GEM16_SERVER          gem16-server executable
  GEM16_MODEL           Compiled gem16 26B Target directory
  GEM16_ASSISTANT_MODEL Compiled 26B MTP Assistant directory
  SERVER_PORT           Loopback port reused sequentially (default: 18080)
  STARTUP_TIMEOUT       Server startup timeout in seconds (default: 300)
EOF
  exit 0
fi

if [[ $# -ne 0 ]]; then
  echo "error: unexpected arguments; use --help" >&2
  exit 64
fi

revision=$(git -C "$repo_root" rev-parse --short=12 HEAD)
suite_dir=${OUTPUT_DIR:-$repo_root/benchmarks/results/m19-quality-suite/$revision}
mkdir -p "$suite_dir"

run_pair() {
  local order=$1
  local benchmark=$2
  local examples=$3
  local label=$4
  local task_dir=$suite_dir/$order-$benchmark
  echo
  echo "=== $label: paired full run ($examples questions) ==="
  BENCHMARK=$benchmark \
  NUM_EXAMPLES=$examples \
  OUTPUT_DIR=$task_dir \
  GEM16_MTP_DRAFT_TOKENS=2 \
    "$paired_runner"
}

echo "Starting resumable paired quality suite"
echo "Suite results: $suite_dir"
echo "Decode profiles: llama.cpp ordinary; gem16 exact sampled fixed-D2 MTP"
run_pair 01 gsm8k 1319 "GSM8K"
run_pair 02 aime26 30 "AIME 2026"
run_pair 03 gpqa 198 "GPQA Diamond"

echo
echo "Complete paired quality suite finished"
echo "Results: $suite_dir"
