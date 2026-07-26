#!/usr/bin/env python3
"""Run the built gem16-inspect executable with a stable repository-relative default."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    executable = "gem16-inspect.exe" if os.name == "nt" else "gem16-inspect"
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("build") / platform.system() / "host-debug" / "bin" / executable,
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    command = [str(args.binary), "--model", str(args.model)]
    if args.json:
        command += ["--json", str(args.json)]
    if args.validate:
        command.append("--validate")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

