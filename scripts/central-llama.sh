#!/usr/bin/env bash
# Source this file to resolve a current, shared CUDA server for a benchmark.
# The resolved path is stable even if the central 'current' symlink changes later.
gem16_resolve_central_llama_server() {
  local requested=${1:-} root source server commit latest version
  root=${LLAMA_CPP_CENTRAL_ROOT:-$HOME/Development/_engines/llama.cpp}
  source=$(realpath -e "$root/current") || { echo "error: central llama.cpp is missing at $root/current" >&2; return 2; }
  server=$source/build-central-cuda/bin/llama-server
  [[ -x $server ]] || { echo "error: central CUDA llama-server is missing: $server" >&2; return 2; }
  if [[ -n $requested && $(realpath -e "$requested") != "$server" ]]; then
    echo 'error: benchmark llama-server override must resolve to the current central CUDA server' >&2
    return 2
  fi
  commit=$(git -C "$source" rev-parse HEAD) || return 2
  [[ -z $(git -C "$source" status --porcelain --untracked-files=no) ]] || {
    echo 'error: central llama.cpp source contains tracked modifications' >&2; return 2;
  }
  if ! latest=$(timeout 20 git ls-remote --exit-code https://github.com/ggml-org/llama.cpp.git refs/heads/master); then
    echo 'error: cannot verify the current official llama.cpp master; benchmark not started' >&2
    return 2
  fi
  latest=${latest%%[[:space:]]*}
  if [[ $commit != "$latest" ]]; then
    echo "error: central llama.cpp $commit is behind official master $latest" >&2
    echo 'Run benchmarks/baselines/llama_cpp/build.sh to update the central build.' >&2
    return 2
  fi
  version=$("$server" --version 2>&1) || { echo 'error: central llama-server cannot run' >&2; return 2; }
  [[ $version == *"${commit:0:7}"* ]] || {
    echo 'error: central llama-server version does not match its source checkout' >&2; return 2;
  }
  printf '%s\n' "$server"
}

# Include shared libraries: the launcher itself is only a small wrapper.
gem16_central_llama_runtime_sha256() {
  (
    set -o pipefail
    cd -- "$(dirname -- "$1")"
    find . -maxdepth 1 -type f \( -name 'llama-server' -o -name 'lib*.so*' \) -print0 \
      | LC_ALL=C sort -z | xargs -0 --no-run-if-empty sha256sum | sha256sum | cut -d' ' -f1
  )
}
