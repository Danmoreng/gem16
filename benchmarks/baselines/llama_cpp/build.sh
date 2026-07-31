#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
expected_commit="$(tr -d '[:space:]' < "${script_dir}/commit.txt")"
source_dir="${LLAMA_CPP_SOURCE:-${repo_root}/third_party/cache/llama.cpp}"
build_dir="${LLAMA_CPP_BUILD_DIR:-${repo_root}/build/Linux/llama_cpp/release}"

if [[ ! -d "${source_dir}/.git" ]]; then
  mkdir -p "$(dirname -- "${source_dir}")"
  git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "${source_dir}"
fi

git -C "${source_dir}" fetch --filter=blob:none origin "${expected_commit}"
git -C "${source_dir}" checkout --detach "${expected_commit}"

actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  echo "error: expected llama.cpp ${expected_commit}, got ${actual_commit}" >&2
  exit 2
fi
if [[ -n "$(git -C "${source_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "error: refusing to build a modified llama.cpp worktree" >&2
  exit 2
fi

cmake -S "${source_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TOOLS=ON
cmake --build "${build_dir}" --parallel --target llama-cli llama-bench llama-quantize llama-gguf llama-server

"${build_dir}/bin/llama-cli" --version
