#!/usr/bin/env python3
"""Run position-aligned C++ inference and compare it with a vLLM fixture."""

from __future__ import annotations

try:
    from tools.hf_cache import default_target_model
except ModuleNotFoundError:
    from hf_cache import default_target_model

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any

import compare_logits


class TeacherForcedError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=default_target_model())
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-cache", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument(
        "--prompt-id",
        action="append",
        help="compare only this prompt; may be repeated",
    )
    parser.add_argument("--vocabulary", type=int, default=262144)
    parser.add_argument(
        "--keep-logits",
        type=Path,
        help="retain one raw float32 logit file per prompt in this directory",
    )
    return parser.parse_args()


def unsigned_token_list(value: Any, field: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(token, int) and token >= 0 for token in value)
    ):
        raise TeacherForcedError(f"{field} must be a non-empty unsigned token list")
    return value


def generation_controls(model: Path) -> tuple[list[int], list[int]]:
    document = json.loads(
        (model / "generation_config.json").read_text(encoding="utf-8")
    )
    stop_tokens = document.get("eos_token_id")
    if isinstance(stop_tokens, int):
        stop_tokens = [stop_tokens]
    stops = unsigned_token_list(stop_tokens, "eos_token_id")
    suppressed = document.get("suppress_tokens", [])
    if not isinstance(suppressed, list) or not all(
        isinstance(token, int) and token >= 0 for token in suppressed
    ):
        raise TeacherForcedError("suppress_tokens must be an unsigned token list")
    return stops, suppressed


def select_prompts(
    document: dict[str, Any], selected_ids: list[str] | None
) -> list[dict[str, Any]]:
    prompts = document.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise TeacherForcedError("golden fixture has no prompts")
    by_id: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        prompt_id = prompt.get("id") if isinstance(prompt, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            raise TeacherForcedError("golden fixture contains a prompt without an id")
        if prompt_id in by_id:
            raise TeacherForcedError(f"duplicate golden prompt id: {prompt_id}")
        by_id[prompt_id] = prompt
    if selected_ids is None:
        return list(by_id.values())
    unknown = [prompt_id for prompt_id in selected_ids if prompt_id not in by_id]
    if unknown:
        raise TeacherForcedError(f"unknown prompt ids: {', '.join(unknown)}")
    return [by_id[prompt_id] for prompt_id in selected_ids]


def validate_reference_cache(document: dict[str, Any], engine_cache: str) -> None:
    execution = document.get("execution")
    reference_cache = (
        execution.get("kv_cache_dtype") if isinstance(execution, dict) else None
    )
    expected = {"fp8": "fp8", "bfloat16": "bf16"}.get(reference_cache)
    if expected is not None and expected != engine_cache:
        raise TeacherForcedError(
            f"reference K/V cache is {reference_cache}, but engine cache is {engine_cache}"
        )


def validate_inference(
    inference: dict[str, Any],
    targets: list[int],
    cache_mode: str,
) -> None:
    expected_mode = (
        "checkpoint_fp8" if cache_mode == "fp8" else "bf16_correctness"
    )
    expected_storage = (
        "uint8_e4m3fn" if cache_mode == "fp8" else "float32_bf16_semantics"
    )
    required = {
        "status": "characterization",
        "benchmark_qualified": False,
        "fallbacks": 0,
        "packed_weight_source_layout_direct": False,
        "weight_layout": "sm120_row8_k64",
        "weight_scale_layout": "sm120_row8_k64",
        "load_time_weight_swizzle": True,
        "load_time_scale_swizzle": True,
        "persistent_repack_bytes": 0,
        "fp8_prefill_tile": "cutlass_m128n128k64",
        "fp8_prefill_output": "scaled_bf16",
        "fp8_prefill_storage": "physical_bf16",
        "fp8_prefill_pipeline_stages": 0,
        "fp8_prefill_schedule": "cutlass_auto",
        "local_prefill_query_heads_per_cta": 2,
        "global_prefill_query_heads_per_cta": 4,
        "local_prefill_fp8_staging": "async_fp8x16_fp8x4_bf16x2",
        "global_prefill_fp8_staging": "async_contiguous_fp8x16_fp8x4_bf16x2",
        "prefill_chunk_tokens": 8192 if cache_mode == "fp8" else 1024,
        "grouped_qkv_prefill": False,
        "fused_rmsnorm_boundaries": True,
        "fused_prefill_rmsnorm_fp8_quantization": True,
        "fused_prefill_rmsnorm_nvfp4_quantization": True,
        "fused_prefill_gated_gelu_nvfp4_quantization": True,
        "fused_prefill_qk_rmsnorm_rope": True,
        "prefill_rope_table": "precomputed_exact_max_context",
        "token_loop_allocations": False,
        "decoding_mode": "teacher_forced",
        "kv_cache_mode": expected_mode,
        "kv_cache_storage": expected_storage,
        "teacher_forced_token_ids": targets,
        "logits_dumped": True,
        "logits_dump_steps": len(targets),
    }
    for field, expected in required.items():
        if inference.get(field) != expected:
            raise TeacherForcedError(
                f"inference field {field} is {inference.get(field)!r}, "
                f"expected {expected!r}"
            )
    predictions = unsigned_token_list(
        inference.get("output_token_ids"), "output_token_ids"
    )
    if len(predictions) != len(targets):
        raise TeacherForcedError("inference returned the wrong number of predictions")
    matches = sum(left == right for left, right in zip(predictions, targets))
    if inference.get("teacher_forced_matches") != matches:
        raise TeacherForcedError("inference teacher-forced match count is inconsistent")


def aggregate_comparisons(prompts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    prompt_list = list(prompts)
    steps = [
        step
        for prompt in prompt_list
        for step in prompt["comparison"]["steps"]
    ]
    if not steps:
        raise TeacherForcedError("no comparison steps were produced")
    deltas = [
        entry["absolute_delta"]
        for step in steps
        for entry in step["reference_top_logprob_comparisons"]
    ]
    selected_deltas = [
        next(
            entry["absolute_delta"]
            for entry in step["reference_top_logprob_comparisons"]
            if entry["token_id"] == step["reference_top1_token_id"]
        )
        for step in steps
    ]
    agreements = sum(bool(step["top1_agreement"]) for step in steps)
    ranks = [int(step["reference_top1_engine_rank"]) for step in steps]
    overlaps = [int(step["reference_top20_overlap"]) for step in steps]
    exact_prompts = sum(
        all(step["top1_agreement"] for step in prompt["comparison"]["steps"])
        for prompt in prompt_list
    )
    return {
        "prompts_compared": len(prompt_list),
        "prompts_with_all_top1_agree": exact_prompts,
        "positions_compared": len(steps),
        "top1_agreements": agreements,
        "top1_agreement_rate": agreements / len(steps),
        "reference_top1_in_engine_top5": sum(rank <= 5 for rank in ranks),
        "reference_top1_in_engine_top5_rate": sum(rank <= 5 for rank in ranks)
        / len(ranks),
        "reference_top1_in_engine_top20": sum(rank <= 20 for rank in ranks),
        "reference_top1_in_engine_top20_rate": sum(rank <= 20 for rank in ranks)
        / len(ranks),
        "mean_reference_top20_overlap": statistics.fmean(overlaps),
        "mean_reference_top20_logprob_absolute_delta": statistics.fmean(deltas),
        "maximum_reference_top20_logprob_absolute_delta": max(deltas),
        "mean_selected_logprob_absolute_delta": statistics.fmean(selected_deltas),
        "maximum_selected_logprob_absolute_delta": max(selected_deltas),
    }


def run_prompt(
    run: Path,
    model: Path,
    prompt: dict[str, Any],
    cache_mode: str,
    stop_tokens: list[int],
    suppressed_tokens: list[int],
    vocabulary: int,
    logits_path: Path,
) -> dict[str, Any]:
    prompt_id = prompt.get("id")
    input_ids = unsigned_token_list(
        prompt.get("prompt_token_ids"), f"{prompt_id}.prompt_token_ids"
    )
    targets = unsigned_token_list(
        prompt.get("output_token_ids"), f"{prompt_id}.output_token_ids"
    )
    context = len(input_ids) + len(targets)
    if context > 1024:
        raise TeacherForcedError(
            f"{prompt_id} requires context {context}, beyond the current 1024-token path"
        )
    command = [
        str(run),
        "--model",
        str(model),
        "--input-token-ids",
        ",".join(map(str, input_ids)),
        "--teacher-forced-token-ids",
        ",".join(map(str, targets)),
        "--max-tokens",
        str(len(targets)),
        "--max-context",
        str(context),
        "--greedy",
        "--kv-cache",
        cache_mode,
        "--stop-token-ids",
        ",".join(map(str, stop_tokens)),
        "--dump-logits",
        str(logits_path),
    ]
    if suppressed_tokens:
        command.extend(
            ["--suppress-token-ids", ",".join(map(str, suppressed_tokens))]
        )
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise TeacherForcedError(
            f"{prompt_id} inference failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        inference = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise TeacherForcedError(
            f"{prompt_id} inference emitted invalid JSON: {error}"
        ) from error
    if not isinstance(inference, dict):
        raise TeacherForcedError(f"{prompt_id} inference did not emit an object")
    validate_inference(inference, targets, cache_mode)
    logit_steps = compare_logits.read_logits(logits_path, vocabulary)
    if len(logit_steps) != len(targets):
        raise TeacherForcedError(
            f"{prompt_id} emitted {len(logit_steps)} logit steps for {len(targets)} targets"
        )
    comparison = compare_logits.compare(logit_steps, prompt)
    predictions = inference["output_token_ids"]
    first_divergence = next(
        (
            index
            for index, (prediction, target) in enumerate(zip(predictions, targets))
            if prediction != target
        ),
        None,
    )
    return {
        "prompt_id": prompt_id,
        "prompt_tokens": len(input_ids),
        "positions": len(targets),
        "top1_agreements": inference["teacher_forced_matches"],
        "first_top1_divergence": first_divergence,
        "target_token_ids": targets,
        "engine_top1_token_ids": predictions,
        "inference": inference,
        "comparison": comparison,
    }


def main() -> int:
    args = parse_args()
    try:
        if args.vocabulary <= 0:
            raise TeacherForcedError("--vocabulary must be positive")
        run = args.run.resolve(strict=True)
        model = args.model.resolve(strict=True)
        golden_path = args.golden.resolve(strict=True)
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        if not isinstance(golden, dict) or golden.get("schema_version") != 1:
            raise TeacherForcedError("golden fixture must be a schema-version-1 object")
        validate_reference_cache(golden, args.kv_cache)
        prompts = select_prompts(golden, args.prompt_id)
        stop_tokens, suppressed_tokens = generation_controls(model)

        if args.keep_logits is not None:
            logits_directory = args.keep_logits.resolve()
            logits_directory.mkdir(parents=True, exist_ok=True)
            temporary: tempfile.TemporaryDirectory[str] | None = None
        else:
            temporary = tempfile.TemporaryDirectory(prefix="gem16-logits-")
            logits_directory = Path(temporary.name)

        try:
            results = []
            for index, prompt in enumerate(prompts, start=1):
                prompt_id = prompt["id"]
                print(
                    f"[{index}/{len(prompts)}] {prompt_id}",
                    file=sys.stderr,
                    flush=True,
                )
                results.append(
                    run_prompt(
                        run,
                        model,
                        prompt,
                        args.kv_cache,
                        stop_tokens,
                        suppressed_tokens,
                        args.vocabulary,
                        logits_directory / f"{prompt_id}.f32",
                    )
                )
        finally:
            if temporary is not None:
                temporary.cleanup()

        summary = aggregate_comparisons(results)
        document = {
            "schema_version": 1,
            "status": "diagnostic",
            "benchmark_qualified": False,
            "comparison_mode": "teacher_forced",
            "engine_kv_cache": args.kv_cache,
            "reference_golden": str(golden_path),
            "reference_execution": golden.get("execution"),
            "summary": summary,
            "prompts": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (
        TeacherForcedError,
        compare_logits.ComparisonError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
