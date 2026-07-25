#!/usr/bin/env python3
"""Validate native prefill boundaries against pinned vLLM logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any

import compare_logits


class ValidationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(
            "tests/golden/vllm-gemma4-12b-nvfp4-prefill-boundaries.json"
        ),
    )
    return parser.parse_args()


def unsigned_token_list(value: Any, field: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(token, int) and token >= 0 for token in value)
    ):
        raise ValidationError(f"{field} must be a non-empty unsigned token list")
    return value


def run_case(
    executable: Path,
    model: Path,
    case: dict[str, Any],
    directory: Path,
) -> dict[str, object]:
    count = case.get("prompt_tokens")
    if not isinstance(count, int) or count <= 0:
        raise ValidationError("golden case has an invalid prompt_tokens value")
    prompt = unsigned_token_list(case.get("prompt_token_ids"), "prompt_token_ids")
    if len(prompt) != count:
        raise ValidationError(f"{count}-token golden prompt has the wrong length")
    reference_output = unsigned_token_list(
        case.get("output_token_ids"), "output_token_ids"
    )
    reference_logprobs = case.get("top_logprobs")
    if (
        not isinstance(reference_logprobs, list)
        or len(reference_logprobs) != len(reference_output)
    ):
        raise ValidationError(f"{count}-token golden top-logprobs are malformed")
    # Only the first generated position is produced by the batched prefill
    # plan. Later positions exercise decode and have their own correctness
    # suite, so they must not influence this boundary-specific gate.
    targets = reference_output[:1]
    top_logprobs = reference_logprobs[:1]
    logits_path = directory / f"prefill-{count}.f32"
    command = [
        str(executable),
        "--model",
        str(model),
        "--input-token-ids",
        ",".join(map(str, prompt)),
        "--teacher-forced-token-ids",
        ",".join(map(str, targets)),
        "--max-tokens",
        str(len(targets)),
        "--max-context",
        str(count + len(targets)),
        "--kv-cache",
        "fp8",
        "--greedy",
        "--dump-logits",
        str(logits_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ValidationError(
            f"{count}-token inference failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    inference = json.loads(completed.stdout)
    if not isinstance(inference, dict):
        raise ValidationError(f"{count}-token inference did not emit an object")
    logits = compare_logits.read_logits(logits_path, 262144)
    comparison = compare_logits.compare(
        logits,
        {
            "id": f"prefill_boundary_{count}",
            "output_token_ids": targets,
            "top_logprobs": top_logprobs,
        },
    )
    steps = comparison["steps"]
    ranks = [int(step["reference_top1_engine_rank"]) for step in steps]
    selected_deltas = [
        next(
            entry["absolute_delta"]
            for entry in step["reference_top_logprob_comparisons"]
            if entry["token_id"] == step["reference_top1_token_id"]
        )
        for step in steps
    ]
    runtime_ok = all(
        (
            inference.get("fallbacks") == 0,
            inference.get("token_loop_allocations") is False,
            inference.get("kv_cache_mode") == "checkpoint_fp8",
            inference.get("logits_dump_steps") == len(targets),
        )
    )
    first_top1 = bool(steps[0]["top1_agreement"])
    return {
        "prompt_tokens": count,
        "reference_output_token_ids": targets,
        "engine_predictions_under_teacher_forcing": inference.get(
            "output_token_ids"
        ),
        "prefill_chunk_tokens": inference.get("prefill_chunk_tokens"),
        "first_prefill_top1_agreement": first_top1,
        "top1_agreements": sum(bool(step["top1_agreement"]) for step in steps),
        "positions": len(steps),
        "maximum_reference_top1_rank": max(ranks),
        "mean_selected_logprob_absolute_delta": statistics.fmean(
            selected_deltas
        ),
        "maximum_selected_logprob_absolute_delta": max(selected_deltas),
        "runtime_invariants_ok": runtime_ok,
        "qualified": runtime_ok and first_top1,
    }


def main() -> int:
    args = parse_args()
    try:
        executable = args.run.resolve(strict=True)
        model = args.model.resolve(strict=True)
        golden = json.loads(args.golden.read_text(encoding="utf-8"))
        cases = golden.get("cases") if isinstance(golden, dict) else None
        if not isinstance(cases, list) or not cases:
            raise ValidationError("golden fixture has no cases")
        with tempfile.TemporaryDirectory(prefix="gem16gb-prefill-boundaries-") as raw:
            directory = Path(raw)
            results = [run_case(executable, model, case, directory) for case in cases]
    except (
        ValidationError,
        compare_logits.ComparisonError,
        json.JSONDecodeError,
        OSError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    qualified = all(bool(item["qualified"]) for item in results)
    document = {
        "schema_version": 2,
        "status": "ok" if qualified else "mismatch",
        "reference": {
            "checkpoint": golden.get("checkpoint"),
            "runtime": golden.get("reference_runtime"),
            "execution": golden.get("execution"),
        },
        "cases": results,
    }
    print(json.dumps(document, sort_keys=True))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
