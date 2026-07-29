#!/usr/bin/env python3
"""Validate resident two-turn sampled MTP chat against ordinary chat."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(command: list[str], conversation: str) -> tuple[list[str], str]:
    completed = subprocess.run(
        command, input=conversation, text=True, capture_output=True, check=True
    )
    responses = []
    for line in completed.stdout.splitlines():
        marker = "model> "
        if marker in line:
            responses.append(line.split(marker, 1)[1])
    return responses, completed.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--assistant", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    common = [
        str(args.binary), "--model", str(args.target),
        "--max-context", str(args.max_context),
        "--max-tokens", str(args.max_tokens),
        "--seed", str(args.seed), "--stats",
    ]
    conversation = (
        "Nenne genau eine Farbe.\n"
        "Wiederhole nur diese Farbe.\n"
        "/quit\n"
    )
    ordinary, ordinary_stats = run(common, conversation)
    mtp, mtp_stats = run(
        common + ["--assistant-model", str(args.assistant),
                  "--mtp-draft-tokens", "2"],
        conversation,
    )
    result = {
        "schema_version": 1,
        "seed": args.seed,
        "sampling": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        "ordinary_responses": ordinary,
        "mtp_responses": mtp,
        "equal": ordinary == mtp,
        "turn_count": len(mtp),
        "gpu_chained_every_turn": mtp_stats.count("GPU chained yes") == 2,
        "ordinary_stats": ordinary_stats.splitlines(),
        "mtp_stats": mtp_stats.splitlines(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["equal"] and result["turn_count"] == 2 and result[
        "gpu_chained_every_turn"
    ] else 1


if __name__ == "__main__":
    raise SystemExit(main())
