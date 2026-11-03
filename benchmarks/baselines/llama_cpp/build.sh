#!/usr/bin/env bash
set -euo pipefail
root=${LLAMA_CPP_CENTRAL_ROOT:-$HOME/Development/_engines/llama.cpp}
updater=$root/update.sh
[[ -x $updater ]] || { echo "error: central llama.cpp updater not found: $updater" >&2; exit 2; }
exec "$updater" "$@"
