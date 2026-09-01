#!/usr/bin/env python3
"""Capture the repository-owned V10 Vision tower timing fixtures."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


CASES = (
    (70, "budget-70-wide.bmp"),
    (140, "budget-140-wide.bmp"),
    (280, "budget-280-wide.bmp"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--e2e-benchmark", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--vision-module", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for budget, fixture in CASES:
        command = [
            str(args.benchmark.resolve()),
            str(args.vision_module.resolve()),
            str((args.fixtures / fixture).resolve()),
            str(budget),
            "--warmups",
            str(args.warmups),
            "--repetitions",
            str(args.repetitions),
        ]
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        payload = json.loads(completed.stdout)
        output = args.output_directory / f"tower-budget-{budget}.json"
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {output}")
        if args.e2e_benchmark is not None:
            if args.model is None:
                raise SystemExit("--model is required with --e2e-benchmark")
            e2e_command = [
                str(args.e2e_benchmark.resolve()),
                str(args.model.resolve()),
                str(args.vision_module.resolve()),
                str((args.fixtures / fixture).resolve()),
                str(budget),
                str(args.warmups),
                str(args.repetitions),
            ]
            e2e_completed = subprocess.run(
                e2e_command, check=True, capture_output=True, text=True
            )
            e2e_payload = json.loads(e2e_completed.stdout)
            e2e_output = args.output_directory / f"e2e-budget-{budget}.json"
            e2e_output.write_text(
                json.dumps(e2e_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {e2e_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
