#!/usr/bin/env python3
"""Validate gem16 GPU sampling against dumped full logits.

This is an end-to-end, checkpoint-backed gate. It intentionally duplicates the
specified SplitMix64 stream and processor order so changes require an explicit
golden/tool update rather than silently changing seeded output.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import subprocess
import tempfile

MASK64 = (1 << 64) - 1
VOCABULARY = 262144


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def sample_reference(
    logits: list[float], history: set[int], temperature: float, top_k: int,
    top_p: float, min_p: float, repetition_penalty: float, seed: int,
    step: int,
) -> tuple[int, int]:
    adjusted: list[tuple[float, int]] = []
    for token, value in enumerate(logits):
        if repetition_penalty != 1.0 and token in history:
            value = value * repetition_penalty if value < 0.0 else value / repetition_penalty
        adjusted.append((value / temperature, token))
    adjusted.sort(key=lambda item: (-item[0], item[1]))
    eligible = adjusted[: top_k or VOCABULARY]
    maximum = eligible[0][0]
    if min_p > 0.0:
        eligible = [item for item in eligible if math.exp(item[0] - maximum) >= min_p]
    weights = [math.exp(item[0] - maximum) for item in eligible]
    total = math.fsum(weights)
    if top_p < 1.0:
        cutoff = top_p * total
        cumulative = 0.0
        count = 0
        while count < len(weights) and (count == 0 or cumulative < cutoff):
            cumulative += weights[count]
            count += 1
        eligible = eligible[:count]
        weights = weights[:count]
        total = cumulative
    bits = splitmix64((seed ^ splitmix64(step)) & MASK64)
    target = (((bits >> 11) + 0.5) / 9007199254740992.0) * total
    cumulative = 0.0
    for item, weight in zip(eligible, weights):
        cumulative += weight
        if target < cumulative:
            return item[1], len(eligible)
    return eligible[-1][1], len(eligible)


def run_once(args: argparse.Namespace, logits_path: pathlib.Path) -> dict:
    command = [
        str(args.run), "--model", str(args.model), "--input-token-ids",
        args.input_token_ids, "--max-tokens", "1", "--max-context",
        str(args.max_context), "--sample", "--temperature", str(args.temperature),
        "--top-k", str(args.top_k), "--top-p", str(args.top_p), "--min-p",
        str(args.min_p), "--repetition-penalty", str(args.repetition_penalty),
        "--seed", str(args.seed), "--dump-logits", str(logits_path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--input-token-ids", default="2,105,2364,107")
    parser.add_argument("--max-context", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.02)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    history = {int(value) for value in args.input_token_ids.split(",")}
    with tempfile.TemporaryDirectory(prefix="gem16-sampling-") as directory:
        logits_path = pathlib.Path(directory) / "logits.bin"
        first = run_once(args, logits_path)
        data = logits_path.read_bytes()
        if len(data) != VOCABULARY * 4:
            raise RuntimeError(f"unexpected logit dump size: {len(data)}")
        logits = list(struct.unpack(f"<{VOCABULARY}f", data))
        expected, eligible = sample_reference(
            logits, history, args.temperature, args.top_k, args.top_p,
            args.min_p, args.repetition_penalty, args.seed, 0)
        second = run_once(args, pathlib.Path(directory) / "repeat.bin")
    actual = first["output_token_ids"][0]
    report = {
        "schema_version": 1,
        "status": "ok" if actual == expected and first["output_token_ids"] == second["output_token_ids"] else "failed",
        "actual_token_id": actual,
        "reference_token_id": expected,
        "eligible_tokens_after_filters": eligible,
        "seeded_repeat_match": first["output_token_ids"] == second["output_token_ids"],
        "sampling": first["sampling"],
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
