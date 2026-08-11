#!/usr/bin/env python3
"""Require two Gemma 4 26B BF16 captures to match at deterministic boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def stable_execution(document: dict[str, object]) -> object:
    execution = document.get("execution")
    if not isinstance(execution, dict):
        return execution
    stable = dict(execution)
    stable.pop("max_rss_kib", None)
    return stable


def compare(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    compared = (
        "checkpoint",
        "software",
        "execution",
        "prompt",
        "captures",
        "final_logits",
        "generated_token_ids",
    )
    mismatches = [
        field
        for field in compared
        if (
            stable_execution(first) != stable_execution(second)
            if field == "execution"
            else first.get(field) != second.get(field)
        )
    ]
    first_captures = first.get("captures")
    capture_count = len(first_captures) if isinstance(first_captures, dict) else 0
    return {
        "schema_version": 1,
        "status": "pass" if not mismatches else "fail",
        "compared_fields": list(compared),
        "mismatched_fields": mismatches,
        "capture_count": capture_count,
        "final_logits_sha256": (
            first.get("final_logits", {}).get("sha256")
            if isinstance(first.get("final_logits"), dict)
            else None
        ),
        "generated_token_ids": first.get("generated_token_ids"),
        "execution_metadata_excluded": ["max_rss_kib"],
    }


def main() -> int:
    args = parse_args()
    first = json.loads(args.first.read_text(encoding="utf-8"))
    second = json.loads(args.second.read_text(encoding="utf-8"))
    report = compare(first, second)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
