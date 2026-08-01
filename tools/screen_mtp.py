#!/usr/bin/env python3
"""Run a short, fixed-output MTP development screen.

This is intentionally not a qualification tool.  It trims a pinned workload
to a requested prompt length, disables stop handling, and measures only the
requested engine modes so kernel candidates can be rejected quickly.
"""

from __future__ import annotations

try:
    from tools.hf_cache import default_assistant_model, default_target_model
except ModuleNotFoundError:
    from hf_cache import default_assistant_model, default_target_model

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_wikipedia_workload import (
    BenchmarkError,
    load_workload,
    positive_int,
    repository_state,
    run_gem16,
    summarize_runs,
    token_u32_checksum,
)


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=default_target_model())
    parser.add_argument(
        "--assistant-model", type=Path, default=default_assistant_model()
    )
    parser.add_argument("--prompt-tokens", type=positive_int, default=2048)
    parser.add_argument("--output-tokens", type=positive_int, default=256)
    parser.add_argument("--warmups", type=nonnegative_int, default=1)
    parser.add_argument("--repetitions", type=positive_int, default=2)
    parser.add_argument("--include-ordinary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        workload_path = args.workload.resolve(strict=True)
        workload, full_prompt, original_generation = load_workload(workload_path)
        if args.prompt_tokens > len(full_prompt):
            raise BenchmarkError("screen prompt exceeds source workload")
        prompt = full_prompt[: args.prompt_tokens]
        generation = {
            "max_new_tokens": args.output_tokens,
            "stop_token_ids": [],
            "suppress_token_ids": original_generation["suppress_token_ids"],
        }
        output = args.output.resolve()
        prompt_file = output.with_suffix(".prompt-token-ids.txt")
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(
            ",".join(str(token) for token in prompt), encoding="ascii"
        )
        executable = args.executable.resolve(strict=True)
        model = args.model.resolve(strict=True)
        assistant = args.assistant_model.resolve(strict=True)

        modes = ["mtp_d2"]
        if args.include_ordinary:
            modes.insert(0, "ordinary")
        runs: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
        output_tokens: dict[str, list[int]] = {}

        def run(mode: str) -> tuple[dict[str, Any], list[int]]:
            return run_gem16(
                executable,
                model,
                prompt_file,
                len(prompt),
                generation,
                assistant if mode == "mtp_d2" else None,
                2 if mode == "mtp_d2" else 0,
                False,
            )

        for index in range(args.warmups):
            for mode in modes:
                print(f"warmup {index + 1}: {mode}", flush=True)
                _, tokens = run(mode)
                output_tokens.setdefault(mode, tokens)
        for index in range(args.repetitions):
            for mode in modes:
                print(f"measured {index + 1}: {mode}", flush=True)
                result, tokens = run(mode)
                if tokens != output_tokens.setdefault(mode, tokens):
                    raise BenchmarkError(f"{mode} changed exact output")
                runs[mode].append(result)

        if "ordinary" in output_tokens and (
            output_tokens["ordinary"] != output_tokens["mtp_d2"]
        ):
            raise BenchmarkError("ordinary and MTP outputs differ")

        summary = {mode: summarize_runs(values) for mode, values in runs.items()}
        document = {
            "schema_version": 1,
            "status": "screen_passed",
            "scope": "short_mtp_development_screen_not_qualification",
            "benchmark_source": repository_state(),
            "workload": {
                "source_path": str(workload_path),
                "source_id": workload.get("id"),
                "prompt_tokens": len(prompt),
                "prompt_token_ids_sha256": token_u32_checksum(prompt),
                "fixed_output_tokens": args.output_tokens,
            },
            "configuration": {
                "kv_cache": "checkpoint_fp8",
                "mtp_draft_tokens": 2,
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                "modes": modes,
            },
            "summary": summary,
            "runs": runs,
            "representative_output_token_ids": output_tokens,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "output": str(output),
            "mtp_d2_median": summary["mtp_d2"]["decode_tokens_per_second"]["median"],
            "ordinary_median": (
                summary["ordinary"]["decode_tokens_per_second"]["median"]
                if "ordinary" in summary else None
            ),
        }, sort_keys=True))
        return 0
    except (BenchmarkError, OSError, ValueError, KeyError) as error:
        import sys

        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
