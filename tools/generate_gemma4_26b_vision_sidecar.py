#!/usr/bin/env python3
"""Compile the pinned Google QAT BF16 Vision tower into a GEM16 FP8 sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gem16_compile.common import CompilerError
from tools.gem16_compile.vision_sidecar import SOURCE_LOCK_PATH, compile_vision_sidecar


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-fp8-encoder", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, default=SOURCE_LOCK_PATH)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-host-memory", type=int, default=2 * 1024 * 1024 * 1024)
    parser.add_argument("--staging-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--native-timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    try:
        result = compile_vision_sidecar(
            source_directory=args.source,
            output_directory=args.output,
            native_encoder=args.native_fp8_encoder,
            source_lock=args.source_lock,
            threads=args.threads,
            host_memory_cap_bytes=args.max_host_memory,
            staging_bytes=args.staging_bytes,
            native_timeout_seconds=args.native_timeout_seconds,
        )
    except CompilerError as error:
        print(f"vision_sidecar_error: {error}", file=sys.stderr)
        return error.exit_code
    print("vision_sidecar_ok " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
