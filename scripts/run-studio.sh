#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GEM16_REPO_ROOT="$repo_root"
exec "$repo_root/gradlew" -p "$repo_root" :studioApp:run
