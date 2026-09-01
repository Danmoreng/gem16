#!/usr/bin/env python3
"""Run the local-only V11 Vision/fixed-D2 exactness laboratory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def regular_path(text: str) -> Path:
    path = Path(text).resolve()
    if not path.is_file() or path.is_symlink():
        raise argparse.ArgumentTypeError(f"not a regular file: {path}")
    return path


def directory_path(text: str) -> Path:
    path = Path(text).resolve()
    if not path.is_dir() or path.is_symlink():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=regular_path)
    parser.add_argument("--target", required=True, type=directory_path)
    parser.add_argument("--assistant", required=True, type=directory_path)
    parser.add_argument("--vision", required=True, type=directory_path)
    parser.add_argument("--image", required=True, type=regular_path)
    parser.add_argument("--budget", required=True, choices=(70, 140, 280), type=int)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()

    command = [
        str(options.binary),
        str(options.target),
        str(options.assistant),
        str(options.vision),
        str(options.image),
        str(options.budget),
    ]
    gate_closed_environment = os.environ.copy()
    gate_closed_environment.pop("GEM16_VISION_D2_DIAGNOSTIC", None)
    gate_closed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=gate_closed_environment,
    )
    if gate_closed.returncode != 64:
        print(
            "V11 driver did not fail closed without the diagnostic gate: "
            f"exit={gate_closed.returncode}",
            file=os.sys.stderr,
        )
        return 2
    environment = os.environ.copy()
    # The committed evidence runner always executes the complete budget slice,
    # even when a developer previously used the binary's local short filter.
    environment.pop("GEM16_V11_MATRIX_FILTER", None)
    environment["GEM16_VISION_D2_DIAGNOSTIC"] = "1"
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.stderr:
        print(completed.stderr, end="", file=os.sys.stderr)
    try:
        artifact = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        print(f"V11 driver returned invalid JSON: {error}", file=os.sys.stderr)
        return completed.returncode or 2
    artifact["driver_exit_code"] = completed.returncode
    artifact["command"] = command
    artifact["default_gate_exit_code"] = gate_closed.returncode
    artifact["default_gate_fail_closed"] = True

    output = options.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as temporary:
        json.dump(artifact, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o644)
    temporary_path.replace(output)
    print(output)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
