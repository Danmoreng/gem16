#!/usr/bin/env python3
"""Validate same-seed sampled MTP against ordinary target sampling."""

from __future__ import annotations

try:
    from tools.hf_cache import default_assistant_model, default_target_model
except ModuleNotFoundError:
    from hf_cache import default_assistant_model, default_target_model

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def csv_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated integers")
    return values


def run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def token_hash(tokens: list[int]) -> str:
    payload = b"".join(token.to_bytes(4, "little") for token in tokens)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=default_target_model())
    parser.add_argument("--assistant", type=Path, default=default_assistant_model())
    parser.add_argument("--input-token-ids", default="2,9259,107")
    parser.add_argument("--seeds", type=csv_ints, default=[0, 1, 42])
    parser.add_argument("--draft-lengths", type=csv_ints, default=[1, 2, 4])
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--kv-cache", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.max_context <= 0:
        parser.error("token and context limits must be positive")
    if any(length not in (1, 2, 4) for length in args.draft_lengths):
        parser.error("draft lengths must be 1, 2, or 4")

    common = [
        str(args.binary), "--model", str(args.target),
        "--input-token-ids", args.input_token_ids,
        "--max-context", str(args.max_context),
        "--max-tokens", str(args.max_tokens),
        "--kv-cache", args.kv_cache,
        "--sample", "--temperature", "1.0", "--top-k", "64",
        "--top-p", "0.95", "--repetition-penalty",
        str(args.repetition_penalty), "--suppress-token-ids", "258883,258882",
    ]
    cases: list[dict[str, Any]] = []
    for seed in args.seeds:
        ordinary_command = common + ["--seed", str(seed)]
        ordinary = run(ordinary_command)
        expected = ordinary["output_token_ids"]
        for draft_length in args.draft_lengths:
            mtp_command = ordinary_command + [
                "--assistant-model", str(args.assistant),
                "--mtp-draft-tokens", str(draft_length),
            ]
            mtp = run(mtp_command)
            actual = mtp["output_token_ids"]
            case = {
                "seed": seed,
                "draft_length": draft_length,
                "equal": actual == expected,
                "output_tokens": len(actual),
                "sha256_u32le": token_hash(actual),
                "accepted_tokens": mtp["mtp"]["accepted_tokens"],
                "rejected_tokens": mtp["mtp"]["rejected_tokens"],
                "verification_groups": mtp["mtp"]["verification_groups"],
                "gpu_chained": mtp["mtp"]["group_execution"]
                == "gpu_chained_fixed_d2_conditional_graph",
                "streaming": mtp["mtp"]["streaming"],
            }
            cases.append(case)
            print(json.dumps(case, sort_keys=True))
            if actual != expected:
                raise SystemExit(
                    f"sampled MTP differs from ordinary at seed {seed}, D{draft_length}"
                )

    summary = {
        "schema_version": 1,
        "sampling": {"temperature": 1.0, "top_k": 64, "top_p": 0.95,
                     "repetition_penalty": args.repetition_penalty},
        "kv_cache": args.kv_cache,
        "cases": cases,
        "all_equal": all(case["equal"] for case in cases),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
