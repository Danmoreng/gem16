#!/usr/bin/env python3
"""Verify a compiled Gemma 4 26B Vision FP8 sidecar and its immutable manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gem16_compile.common import CompilerError
from tools.gem16_compile.vision_sidecar_verify import verify_vision_sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", type=Path)
    args = parser.parse_args()
    try:
        result = verify_vision_sidecar(args.sidecar)
    except CompilerError as error:
        print(f"vision_sidecar_verify_error: {error}", file=sys.stderr)
        return error.exit_code
    print("vision_sidecar_verify_ok " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
