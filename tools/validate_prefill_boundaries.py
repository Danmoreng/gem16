#!/usr/bin/env python3
"""Validate deterministic generation across native prefill chunk boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


CASES = {
    129: [236770, 236779, 236770, 236770, 236770, 236770, 236770, 236761],
    257: [236761, 236779, 107, 1, 107, 1, 107, 138],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    return parser.parse_args()


def prompt_tokens(count: int) -> list[int]:
    return [1000 + ((index * 7919) % 9000) for index in range(count)]


def run_case(executable: Path, model: Path, count: int) -> dict[str, object]:
    expected = CASES[count]
    command = [
        str(executable),
        "--model",
        str(model),
        "--input-token-ids",
        ",".join(str(token) for token in prompt_tokens(count)),
        "--max-tokens",
        str(len(expected)),
        "--max-context",
        str(count + len(expected)),
        "--greedy",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{count}-token inference failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    result = json.loads(completed.stdout)
    actual = result.get("output_token_ids")
    return {
        "prompt_tokens": count,
        "expected": expected,
        "actual": actual,
        "exact_match": actual == expected,
        "prefill_chunk_tokens": result.get("prefill_chunk_tokens"),
        "fallbacks": result.get("fallbacks"),
        "token_loop_allocations": result.get("token_loop_allocations"),
    }


def main() -> int:
    args = parse_args()
    try:
        executable = args.run.resolve(strict=True)
        model = args.model.resolve(strict=True)
        results = [run_case(executable, model, count) for count in CASES]
    except (json.JSONDecodeError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    document = {
        "schema_version": 1,
        "status": "ok" if all(item["exact_match"] for item in results) else "mismatch",
        "cases": results,
    }
    print(json.dumps(document, sort_keys=True))
    return 0 if document["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
