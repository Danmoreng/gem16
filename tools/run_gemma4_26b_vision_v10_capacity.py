#!/usr/bin/env python3
"""Measure V10 Target+Vision(+Assistant) admission as fresh processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--assistant-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def measure(base: list[str], context: int) -> dict[str, object]:
    command = [*base, "--max-context", str(context), "--print-model-report"]
    begin = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    elapsed = time.monotonic() - begin
    record: dict[str, object] = {
        "context_tokens": context,
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
    }
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if stdout:
        try:
            record["model_report"] = json.loads(stdout)
        except json.JSONDecodeError:
            record["stdout"] = stdout
    if stderr:
        record["stderr"] = stderr
    return record


def main() -> int:
    args = parse_args()
    previous: dict[tuple[str, int], dict[str, object]] = {}
    if args.output.exists():
        prior_payload = json.loads(args.output.read_text(encoding="utf-8"))
        for scenario in prior_payload.get("scenarios", []):
            for run in scenario.get("runs", []):
                previous[(scenario["name"], run["context_tokens"])] = run
    common = [
        str(args.chat.resolve()),
        "--model",
        str(args.model.resolve()),
        "--vision-model",
        str(args.vision_model.resolve()),
    ]
    scenarios = []
    for name, extra, contexts in (
        (
            "target_vision",
            [],
            (
                32768,
                65536,
                86016,
                98304,
                98305,
                131072,
                163840,
                196608,
                229376,
                245760,
                262144,
            ),
        ),
        (
            "target_vision_assistant",
            ["--assistant-model", str(args.assistant_model.resolve())],
            (
                32768,
                65536,
                86016,
                98304,
                131072,
                163840,
                196608,
                221184,
                225280,
                227328,
                228352,
                228864,
                229120,
                229376,
                229377,
                245760,
                262144,
            ),
        ),
    ):
        runs = []
        for context in contexts:
            prior = previous.get((name, context))
            stale_guard = (
                name == "target_vision_assistant"
                and context > 86016
                and prior is not None
                and "M25 Assistant workspace context is invalid"
                in str(prior.get("stderr", ""))
            )
            if prior is not None and not stale_guard:
                runs.append(prior)
                continue
            print(f"measuring {name} at {context}", flush=True)
            runs.append(measure([*common, *extra], context))
        scenarios.append({"name": name, "runs": runs})
    payload = {"schema_version": 1, "scenarios": scenarios}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
