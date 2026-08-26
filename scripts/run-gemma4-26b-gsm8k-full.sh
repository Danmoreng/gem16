#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NUM_EXAMPLES=1319
exec "$script_dir/run-gemma4-26b-gsm8k-smoke.sh" "$@"
