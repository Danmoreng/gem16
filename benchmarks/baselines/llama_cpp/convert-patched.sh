#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
expected_commit="$(tr -d '[:space:]' < "${script_dir}/commit.txt")"
patched_source="${LLAMA_CPP_PATCHED_SOURCE:-${repo_root}/third_party/cache/llama.cpp-mixed}"
python_bin="${LLAMA_CPP_CONVERT_PYTHON:-${repo_root}/third_party/cache/vllm-0.26.0-env/bin/python}"
patch_file="${script_dir}/patches/0001-support-mixed-fp8-nvfp4-compressed-tensors.patch"

if [[ ! -d "${patched_source}/.git" ]]; then
  echo "error: patched converter source not found; run prepare-patched-source.sh" >&2
  exit 2
fi
if [[ "$(git -C "${patched_source}" rev-parse HEAD)" != "${expected_commit}" ]]; then
  echo "error: patched source is not based on ${expected_commit}" >&2
  exit 2
fi
if ! git -C "${patched_source}" apply --unidiff-zero --reverse --check "${patch_file}"; then
  echo "error: tracked mixed-precision patch is not applied exactly" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "error: converter Python not found at ${python_bin}" >&2
  exit 2
fi

LLAMA_CPP_SOURCE="${patched_source}" \
LLAMA_CPP_CONVERT_PYTHON="${python_bin}" \
  exec "${script_dir}/convert.sh" "$@"
