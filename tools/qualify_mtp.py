#!/usr/bin/env python3
"""Run the exact alternating ordinary/MTP Wikipedia qualification."""

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
)


MINIMUM_MTP_DECODE_TOKENS_PER_SECOND = 64.82


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=default_target_model())
    parser.add_argument(
        "--assistant-model", type=Path, default=default_assistant_model()
    )
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--warmup-pairs", type=positive_int, default=3)
    parser.add_argument("--measured-pairs", type=positive_int, default=10)
    parser.add_argument(
        "--sampled", action="store_true",
        help="qualify same-seed Google-recommended target sampling",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def qualification(args: argparse.Namespace) -> dict[str, Any]:
    workload_path = args.workload.resolve(strict=True)
    workload, prompt, generation = load_workload(workload_path)
    model = args.model.resolve(strict=True)
    assistant = args.assistant_model.resolve(strict=True)
    executable = args.executable.resolve(strict=True)
    prompt_file = args.output.with_suffix(".prompt-token-ids.txt").resolve()
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(
        ",".join(str(token) for token in prompt), encoding="ascii"
    )

    ordinary_runs: list[dict[str, Any]] = []
    mtp_runs: list[dict[str, Any]] = []
    pair_order: list[dict[str, Any]] = []
    reference_tokens: list[int] | None = None

    sampling = (
        {
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "seed": getattr(args, "seed", 0),
        }
        if getattr(args, "sampled", False)
        else None
    )

    def run_mode(mode: str) -> tuple[dict[str, Any], list[int]]:
        return run_gem16(
            executable,
            model,
            prompt_file,
            len(prompt),
            generation,
            assistant if mode == "mtp_d2" else None,
            2 if mode == "mtp_d2" else 0,
            False,
            sampling,
        )

    def run_pair(phase: str, index: int, measured: bool) -> None:
        nonlocal reference_tokens
        order = (
            ("ordinary", "mtp_d2")
            if index % 2 == 0
            else ("mtp_d2", "ordinary")
        )
        print(
            f"{phase} pair {index + 1}: {order[0]} then {order[1]}",
            flush=True,
        )
        pair_record: dict[str, Any] = {
            "phase": phase,
            "pair": index + 1,
            "order": list(order),
        }
        for mode in order:
            run, tokens = run_mode(mode)
            if reference_tokens is None:
                reference_tokens = tokens
            if tokens != reference_tokens:
                raise BenchmarkError(
                    f"{phase} pair {index + 1} {mode} changed exact output"
                )
            pair_record[mode] = {
                "decode_tokens_per_second": run["decode_tokens_per_second"],
                "output_token_sha256": run["output_token_sha256"],
            }
            if measured:
                (ordinary_runs if mode == "ordinary" else mtp_runs).append(run)
        pair_order.append(pair_record)

    for index in range(args.warmup_pairs):
        run_pair("warmup", index, False)
    for index in range(args.measured_pairs):
        run_pair("measured", index, True)

    if reference_tokens is None:
        raise BenchmarkError("qualification produced no output")
    ordinary_summary = summarize_runs(ordinary_runs)
    mtp_summary = summarize_runs(mtp_runs)
    ordinary_median = ordinary_summary["decode_tokens_per_second"]["median"]
    mtp_median = mtp_summary["decode_tokens_per_second"]["median"]
    minimum_met = mtp_median >= MINIMUM_MTP_DECODE_TOKENS_PER_SECOND
    return {
        "schema_version": 2,
        "status": "qualified" if minimum_met else "target_not_met",
        "benchmark_source": repository_state(),
        "workload": {
            "path": str(workload_path),
            "id": workload.get("id"),
            "source": workload.get("source"),
            "prompt_tokens": len(prompt),
            "prompt_token_ids_sha256": workload["prompt"]["token_ids_sha256"],
            "max_new_tokens": generation["max_new_tokens"],
            "stop_token_ids": generation["stop_token_ids"],
            "suppress_token_ids": generation["suppress_token_ids"],
        },
        "runtime": {
            "executable": str(executable),
            "checkpoint": str(model),
            "assistant_checkpoint": str(assistant),
            "prompt_token_file": str(prompt_file),
        },
        "configuration": {
            "batch_size": 1,
            "kv_cache": "checkpoint_fp8",
            "mtp_draft_tokens": 2,
            "warmup_pairs": args.warmup_pairs,
            "measured_pairs": args.measured_pairs,
            "alternating_first_mode": True,
            "primary_statistic": "median",
            "decoding_mode":
                "sampled" if getattr(args, "sampled", False) else "greedy",
            "sampling": sampling,
        },
        "qualification": {
            "ordinary_equals_mtp": True,
            "minimum_mtp_decode_tokens_per_second":
                MINIMUM_MTP_DECODE_TOKENS_PER_SECOND,
            "minimum_mtp_decode_tokens_per_second_met": minimum_met,
            "llama_cpp_parity_requires_separate_comparison": True,
            "median_speedup": mtp_median / ordinary_median,
            "median_throughput_increase": mtp_median / ordinary_median - 1.0,
        },
        "summary": {"ordinary": ordinary_summary, "mtp_d2": mtp_summary},
        "runs": {"ordinary": ordinary_runs, "mtp_d2": mtp_runs},
        "pair_order": pair_order,
        "representative_output_token_ids": reference_tokens,
        "limitations": [
            "No continuous power, clock, or thermal telemetry was captured.",
            "TTFT includes prompt processing and first-token selection.",
            "Decode throughput uses generated_tokens - 1 verified intervals.",
        ],
    }


def main() -> int:
    args = parse_args()
    try:
        document = qualification(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": document["status"],
                    "ordinary_median": document["summary"]["ordinary"][
                        "decode_tokens_per_second"
                    ]["median"],
                    "mtp_d2_median": document["summary"]["mtp_d2"][
                        "decode_tokens_per_second"
                    ]["median"],
                    "median_speedup": document["qualification"][
                        "median_speedup"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0 if document["status"] == "qualified" else 2
    except (BenchmarkError, OSError, ValueError, KeyError) as error:
        import sys

        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
