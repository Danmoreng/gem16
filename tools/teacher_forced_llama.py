#!/usr/bin/env python3
"""Compare llama.cpp with vLLM at identical teacher-forced token positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import urllib.error
import urllib.request
from typing import Any


class LlamaComparisonError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--gguf-sha256", required=True)
    parser.add_argument("--llama-cpp-commit", required=True)
    parser.add_argument("--prompt-id", action="append")
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        document = json.load(response)
    if not isinstance(document, dict):
        raise LlamaComparisonError("llama.cpp returned a non-object response")
    return document


def select_prompts(
    document: dict[str, Any], selected_ids: list[str] | None
) -> list[dict[str, Any]]:
    prompts = document.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise LlamaComparisonError("reference fixture has no prompts")
    by_id = {
        prompt.get("id"): prompt for prompt in prompts if isinstance(prompt, dict)
    }
    if len(by_id) != len(prompts) or not all(
        isinstance(prompt_id, str) and prompt_id for prompt_id in by_id
    ):
        raise LlamaComparisonError("reference prompt IDs are invalid or duplicated")
    if selected_ids is None:
        return list(prompts)
    unknown = [prompt_id for prompt_id in selected_ids if prompt_id not in by_id]
    if unknown:
        raise LlamaComparisonError(f"unknown prompt ids: {', '.join(unknown)}")
    return [by_id[prompt_id] for prompt_id in selected_ids]


def token_list(value: Any, field: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(token, int) and token >= 0 for token in value)
    ):
        raise LlamaComparisonError(f"{field} must be a non-empty token list")
    return value


def compare_position(
    response: dict[str, Any],
    reference_entries: Any,
    reference_token: int,
    index: int,
) -> dict[str, Any]:
    probabilities = response.get("completion_probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != 1:
        raise LlamaComparisonError(
            f"llama.cpp position {index} returned no single-token probabilities"
        )
    prediction = probabilities[0]
    predicted_token = prediction.get("id") if isinstance(prediction, dict) else None
    llama_top = prediction.get("top_logprobs") if isinstance(prediction, dict) else None
    if not isinstance(predicted_token, int) or not isinstance(llama_top, list):
        raise LlamaComparisonError(f"llama.cpp position {index} response is malformed")
    if not isinstance(reference_entries, list) or not reference_entries:
        raise LlamaComparisonError(f"reference position {index} is malformed")
    reference_by_id = {
        entry.get("token_id"): entry
        for entry in reference_entries
        if isinstance(entry, dict)
    }
    llama_by_id = {
        entry.get("id"): entry for entry in llama_top if isinstance(entry, dict)
    }
    llama_reference_entry = llama_by_id.get(reference_token)
    reference_selected = reference_by_id.get(reference_token)
    if reference_selected is None:
        raise LlamaComparisonError(
            f"reference top-logprobs omit selected token at position {index}"
        )
    llama_rank = next(
        (
            rank
            for rank, entry in enumerate(llama_top, start=1)
            if isinstance(entry, dict) and entry.get("id") == reference_token
        ),
        None,
    )
    selected_delta = None
    if llama_reference_entry is not None:
        llama_logprob = llama_reference_entry.get("logprob")
        reference_logprob = reference_selected.get("logprob")
        if isinstance(llama_logprob, (int, float)) and isinstance(
            reference_logprob, (int, float)
        ):
            selected_delta = float(llama_logprob) - float(reference_logprob)
    return {
        "index": index,
        "reference_token_id": reference_token,
        "llama_cpp_top1_token_id": predicted_token,
        "top1_agreement": predicted_token == reference_token,
        "reference_top1_llama_cpp_top20_rank": llama_rank,
        "top20_overlap_count": len(set(reference_by_id) & set(llama_by_id)),
        "selected_logprob_delta": selected_delta,
        "llama_cpp_top_logprobs": llama_top,
    }


def aggregate(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [step for prompt in prompts for step in prompt["steps"]]
    if not steps:
        raise LlamaComparisonError("no positions were compared")
    agreements = sum(step["top1_agreement"] for step in steps)
    in_top20 = sum(
        step["reference_top1_llama_cpp_top20_rank"] is not None for step in steps
    )
    in_top5 = sum(
        step["reference_top1_llama_cpp_top20_rank"] is not None
        and step["reference_top1_llama_cpp_top20_rank"] <= 5
        for step in steps
    )
    deltas = [
        abs(step["selected_logprob_delta"])
        for step in steps
        if step["selected_logprob_delta"] is not None
    ]
    if not deltas:
        raise LlamaComparisonError(
            "the llama.cpp top-logprobs contain no reference-selected tokens"
        )
    return {
        "prompts_compared": len(prompts),
        "prompts_with_all_top1_agree": sum(
            all(step["top1_agreement"] for step in prompt["steps"])
            for prompt in prompts
        ),
        "positions_compared": len(steps),
        "top1_agreements": agreements,
        "top1_agreement_rate": agreements / len(steps),
        "reference_top1_in_llama_cpp_top20": in_top20,
        "reference_top1_in_llama_cpp_top20_rate": in_top20 / len(steps),
        "reference_top1_in_llama_cpp_top5": in_top5,
        "reference_top1_in_llama_cpp_top5_rate": in_top5 / len(steps),
        "mean_top20_overlap": statistics.fmean(
            step["top20_overlap_count"] for step in steps
        ),
        "selected_logprob_positions_compared": len(deltas),
        "mean_selected_logprob_absolute_delta": statistics.fmean(deltas),
        "maximum_selected_logprob_absolute_delta": max(deltas),
    }


def main() -> int:
    args = parse_args()
    try:
        reference_path = args.reference.resolve(strict=True)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if not isinstance(reference, dict) or reference.get("schema_version") != 1:
            raise LlamaComparisonError(
                "reference fixture must be a schema-version-1 object"
            )
        prompts = select_prompts(reference, args.prompt_id)
        endpoint = args.endpoint.rstrip("/") + "/completion"
        captures = []
        for prompt_index, prompt in enumerate(prompts, start=1):
            prompt_id = prompt["id"]
            print(
                f"[{prompt_index}/{len(prompts)}] {prompt_id}",
                file=sys.stderr,
                flush=True,
            )
            input_ids = token_list(
                prompt.get("prompt_token_ids"), f"{prompt_id}.prompt_token_ids"
            )
            targets = token_list(
                prompt.get("output_token_ids"), f"{prompt_id}.output_token_ids"
            )
            reference_steps = prompt.get("top_logprobs")
            if not isinstance(reference_steps, list) or len(reference_steps) != len(
                targets
            ):
                raise LlamaComparisonError(
                    f"{prompt_id} reference logprob positions are malformed"
                )
            steps = []
            for index, target in enumerate(targets):
                response = post_json(
                    endpoint,
                    {
                        "prompt": input_ids + targets[:index],
                        "temperature": 0.0,
                        "seed": reference["execution"]["seed"],
                        "n_predict": 1,
                        "n_probs": reference["execution"]["top_logprobs"],
                        "cache_prompt": True,
                        "stream": False,
                    },
                )
                steps.append(
                    compare_position(
                        response, reference_steps[index], target, index
                    )
                )
            first_divergence = next(
                (step["index"] for step in steps if not step["top1_agreement"]),
                None,
            )
            captures.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_tokens": len(input_ids),
                    "positions": len(targets),
                    "top1_agreements": sum(
                        step["top1_agreement"] for step in steps
                    ),
                    "first_top1_divergence": first_divergence,
                    "steps": steps,
                }
            )

        document = {
            "schema_version": 1,
            "status": "diagnostic",
            "benchmark_qualified": False,
            "comparison_mode": "teacher_forced",
            "model_label": args.model_label,
            "gguf_sha256": args.gguf_sha256,
            "llama_cpp_commit": args.llama_cpp_commit,
            "reference": {
                "path": str(reference_path),
                "runtime": "vLLM direct compressed checkpoint",
            },
            "execution": {
                "temperature": 0.0,
                "seed": reference["execution"]["seed"],
                "top_logprobs": reference["execution"]["top_logprobs"],
                "prompt_cache": True,
            },
            "summary": aggregate(captures),
            "prompts": captures,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(document["summary"], sort_keys=True))
        return 0
    except (
        LlamaComparisonError,
        json.JSONDecodeError,
        OSError,
        KeyError,
        TypeError,
        urllib.error.URLError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
