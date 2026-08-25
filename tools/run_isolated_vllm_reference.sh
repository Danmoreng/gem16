#!/usr/bin/env bash
set -euo pipefail

# Run a vLLM reference command in a bounded transient user service. The limits
# apply to the complete process tree, including Ninja, nvcc, and cicc children.
readonly memory_high="40G"
readonly memory_max="45G"
readonly max_jobs="4"
readonly torchinductor_compile_threads="4"
readonly flashinfer_nvcc_threads="1"
readonly tasks_max="256"
readonly runtime_max="3h"

usage() {
  cat <<'EOF'
Usage:
  tools/run_isolated_vllm_reference.sh [--dry-run] -- COMMAND [ARG ...]

Runs COMMAND from the repository root in a transient systemd user service with:
  MemoryHigh=40G, MemoryMax=45G, MemorySwapMax=0
  MAX_JOBS=4, TORCHINDUCTOR_COMPILE_THREADS=4, FLASHINFER_NVCC_THREADS=1
  OOMPolicy=kill, KillMode=control-group, TasksMax=256
  RuntimeMaxSec=3h

If the command exceeds its memory limit, systemd kills the complete service
process tree rather than allowing its compiler children to exhaust host RAM.
The script never downloads a model or clears a compiler cache.
EOF
}

dry_run=false
while (($# > 0)); do
  case "$1" in
    --dry-run)
      dry_run=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "error: expected -- before COMMAND, got: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if (($# == 0)); then
  echo "error: COMMAND is required" >&2
  usage >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
unit="gem16-vllm-reference-$(date -u +%Y%m%dT%H%M%SZ)-$$"

runner=(
  systemd-run
  --user
  --wait
  --collect
  --pipe
  "--unit=${unit}"
  --property=Type=exec
  "--working-directory=${repo_root}"
  --property=MemoryAccounting=yes
  "--property=MemoryHigh=${memory_high}"
  "--property=MemoryMax=${memory_max}"
  --property=MemorySwapMax=0
  # OOMPolicy=kill sets cgroup v2 memory.oom.group=1 for this service.
  --property=OOMPolicy=kill
  --property=KillMode=control-group
  --property=TasksAccounting=yes
  "--property=TasksMax=${tasks_max}"
  "--property=RuntimeMaxSec=${runtime_max}"
  --property=TimeoutStopSec=30s
  "--setenv=MAX_JOBS=${max_jobs}"
  "--setenv=TORCHINDUCTOR_COMPILE_THREADS=${torchinductor_compile_threads}"
  "--setenv=FLASHINFER_NVCC_THREADS=${flashinfer_nvcc_threads}"
  "--setenv=PATH=${PATH}"
  --expand-environment=no
)

# Forward only toolchain/cache paths needed by an offline local-checkpoint run.
# In particular, do not copy tokens or credentials into the transient unit.
for name in CUDA_HOME CUDA_PATH LD_LIBRARY_PATH HF_HOME XDG_CACHE_HOME TMPDIR; do
  if [[ -v "${name}" ]]; then
    runner+=("--setenv=${name}=${!name}")
  fi
done
runner+=(-- "$@")

if [[ "${dry_run}" == true ]]; then
  printf 'unit=%s\n' "${unit}"
  printf 'repository=%s\n' "${repo_root}"
  printf 'limits=MemoryHigh:%s MemoryMax:%s MemorySwapMax:0 TasksMax:%s RuntimeMaxSec:%s\n' \
    "${memory_high}" "${memory_max}" "${tasks_max}" "${runtime_max}"
  printf 'parallelism=MAX_JOBS:%s TORCHINDUCTOR_COMPILE_THREADS:%s FLASHINFER_NVCC_THREADS:%s\n' \
    "${max_jobs}" "${torchinductor_compile_threads}" "${flashinfer_nvcc_threads}"
  printf 'systemd-command='
  printf ' %q' "${runner[@]}"
  printf '\n'
  exit 0
fi

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "error: systemd-run is required" >&2
  exit 2
fi
if ! command -v systemctl >/dev/null 2>&1 || ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "error: a working systemd user manager is required" >&2
  exit 2
fi
if [[ ! -r /sys/fs/cgroup/cgroup.controllers ]] ||
   ! grep -qw memory /sys/fs/cgroup/cgroup.controllers; then
  echo "error: cgroup v2 memory controller is required" >&2
  exit 2
fi

started=false
stop_unit() {
  if [[ "${started}" == true ]]; then
    systemctl --user stop "${unit}.service" >/dev/null 2>&1 || true
  fi
}
on_signal() {
  trap - HUP INT TERM
  stop_unit
  exit 130
}
trap on_signal HUP INT TERM

printf 'Starting bounded vLLM reference unit %s.service\n' "${unit}" >&2
printf 'Limits: MemoryHigh=%s MemoryMax=%s MAX_JOBS=%s TORCHINDUCTOR_COMPILE_THREADS=%s FLASHINFER_NVCC_THREADS=%s\n' \
  "${memory_high}" "${memory_max}" "${max_jobs}" "${torchinductor_compile_threads}" \
  "${flashinfer_nvcc_threads}" >&2
started=true
set +e
"${runner[@]}"
status=$?
set -e
started=false
trap - HUP INT TERM
exit "${status}"
