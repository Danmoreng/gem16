#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
skip_server_build=false
if [[ "${1:-}" == "--skip-server-build" ]]; then
  skip_server_build=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: scripts/run-studio.sh [--skip-server-build]" >&2
  exit 64
fi
if [[ "$skip_server_build" == false ]]; then
  cmake --preset blackwell-release -S "$repo_root"
  cmake --build --preset blackwell-release --target gem16-server --parallel
fi
server="$repo_root/build/Linux/blackwell-release/bin/gem16-server"
if [[ ! -x "$server" ]]; then
  echo "The workspace server does not exist: $server. Run without --skip-server-build first." >&2
  exit 1
fi
export GEM16_REPO_ROOT="$repo_root"
export GEM16_STUDIO_SERVER_EXECUTABLE="$server"
exec "$repo_root/gradlew" -p "$repo_root" :studioApp:run
