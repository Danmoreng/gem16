#!/usr/bin/env python3
"""Validate the full 48-layer greedy characterization against a pinned fixture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


class ValidationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("tests/golden/vllm-gemma4-12b-nvfp4.json"),
    )
    parser.add_argument("--prompt-id", default="exact_blue_no_thinking")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def select_prompt(document: dict[str, Any], prompt_id: str) -> dict[str, Any]:
    prompts = document.get("prompts")
    if not isinstance(prompts, list):
        raise ValidationError("golden fixture has no prompt list")
    for prompt in prompts:
        if isinstance(prompt, dict) and prompt.get("id") == prompt_id:
            return prompt
    raise ValidationError(f"golden fixture has no prompt named {prompt_id!r}")


def validate_result(document: dict[str, Any], expected: list[int]) -> None:
    if document.get("status") != "characterization":
        raise ValidationError("inference did not report characterization status")
    if document.get("benchmark_qualified") is not False:
        raise ValidationError("inference must not claim benchmark qualification")
    if document.get("fallbacks") != 0:
        raise ValidationError("inference used a precision fallback")
    if document.get("packed_weight_source_layout_direct") is not False:
        raise ValidationError("inference did not report the tiled packed-weight layout")
    if document.get("weight_layout") != "sm120_row8_k64":
        raise ValidationError("inference did not use the required SM120 weight layout")
    if document.get("weight_scale_layout") != "sm120_row8_k64":
        raise ValidationError("inference did not use the required SM120 weight-scale layout")
    if document.get("load_time_weight_swizzle") is not True:
        raise ValidationError("inference did not report exact load-time weight swizzling")
    if document.get("load_time_scale_swizzle") is not True:
        raise ValidationError("inference did not report exact load-time scale swizzling")
    if document.get("persistent_repack_bytes") != 0:
        raise ValidationError("inference retained a persistent repack allocation")
    if document.get("fp8_prefill_tile") != "cutlass_m128n128k64":
        raise ValidationError("inference did not use the required CUTLASS FP8 prefill tile")
    if document.get("fp8_prefill_pipeline_stages") != 0:
        raise ValidationError("inference reported a manual FP8 prefill pipeline")
    if document.get("fp8_prefill_schedule") != "cutlass_auto":
        raise ValidationError("inference did not report the required CUTLASS FP8 schedule")
    if document.get("local_prefill_query_heads_per_cta") != 2:
        raise ValidationError("inference did not group local GQA heads per prefill CTA")
    if document.get("global_prefill_query_heads_per_cta") != 4:
        raise ValidationError("inference did not group global GQA heads per prefill CTA")
    if document.get("global_prefill_fp8_staging") != "fp8x4_bf16x2":
        raise ValidationError("inference did not use vectorized global FP8 staging")
    if document.get("prefill_chunk_tokens") != 2048:
        raise ValidationError("inference did not use the required checkpoint-FP8 prefill chunk")
    if document.get("grouped_qkv_prefill") is not False:
        raise ValidationError("inference did not report separate CUTLASS Q/K/V projections")
    if document.get("fused_rmsnorm_boundaries") is not True:
        raise ValidationError("inference did not fuse RMSNorm BF16 boundaries")
    if document.get("fused_prefill_rmsnorm_fp8_quantization") is not True:
        raise ValidationError("inference did not fuse prefill RMSNorm/FP8 quantization")
    if document.get("fused_prefill_rmsnorm_nvfp4_quantization") is not True:
        raise ValidationError("inference did not fuse prefill RMSNorm/NVFP4 quantization")
    if document.get("fused_prefill_gated_gelu_nvfp4_quantization") is not True:
        raise ValidationError("inference did not fuse prefill Gate/Up/GELU/NVFP4 quantization")
    if document.get("fused_prefill_qk_rmsnorm_rope") is not True:
        raise ValidationError("inference did not fuse prefill Q/K RMSNorm/RoPE")
    if document.get("prefill_rope_table") != "precomputed_exact_max_context":
        raise ValidationError("inference did not use the exact persistent prefill RoPE table")
    if document.get("token_loop_allocations") is not False:
        raise ValidationError("inference reported token-loop allocations")
    if document.get("kv_cache_mode") != "checkpoint_fp8":
        raise ValidationError("inference did not use checkpoint FP8 K/V semantics")
    if document.get("kv_cache_storage") != "uint8_e4m3fn":
        raise ValidationError("inference did not use physical FP8 K/V storage")
    if document.get("output_token_ids") != expected:
        raise ValidationError(
            f"greedy tokens differ: expected {expected}, got {document.get('output_token_ids')}"
        )
    for field in ("model_load_ms", "prompt_ms", "decode_ms", "decode_tokens_per_second"):
        value = document.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValidationError(f"inference has invalid {field}")
    if len(expected) > 1 and document["decode_tokens_per_second"] <= 0:
        raise ValidationError("inference did not report positive decode throughput")


def main() -> int:
    args = parse_args()
    try:
        run = args.run.resolve(strict=True)
        model = args.model.resolve(strict=True)
        golden_path = args.golden.resolve(strict=True)
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        prompt = select_prompt(golden, args.prompt_id)
        input_ids = prompt.get("prompt_token_ids")
        expected = prompt.get("output_token_ids")
        generation = json.loads(
            (model / "generation_config.json").read_text(encoding="utf-8")
        )
        stop_tokens = generation.get("eos_token_id")
        if isinstance(stop_tokens, int):
            stop_tokens = [stop_tokens]
        suppressed_tokens = generation.get("suppress_tokens", [])
        if (
            not isinstance(stop_tokens, list)
            or not stop_tokens
            or not all(isinstance(token, int) and token >= 0 for token in stop_tokens)
            or not isinstance(suppressed_tokens, list)
            or not all(isinstance(token, int) and token >= 0 for token in suppressed_tokens)
        ):
            raise ValidationError("checkpoint generation token controls are malformed")
        if not isinstance(input_ids, list) or not input_ids or not all(
            isinstance(token, int) and token >= 0 for token in input_ids
        ):
            raise ValidationError("golden prompt token IDs are malformed")
        if not isinstance(expected, list) or not expected or not all(
            isinstance(token, int) and token >= 0 for token in expected
        ):
            raise ValidationError("golden output token IDs are malformed")
        context = len(input_ids) + len(expected)
        command = [
            str(run),
            "--model",
            str(model),
            "--input-token-ids",
            ",".join(str(token) for token in input_ids),
            "--max-tokens",
            str(len(expected)),
            "--max-context",
            str(context),
            "--greedy",
            "--stop-token-ids",
            ",".join(str(token) for token in stop_tokens),
        ]
        if suppressed_tokens:
            command.extend(
                [
                    "--suppress-token-ids",
                    ",".join(str(token) for token in suppressed_tokens),
                ]
            )
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ValidationError(
                f"inference failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        try:
            inference = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValidationError(f"inference emitted invalid JSON: {error}") from error
        validate_result(inference, expected)
        result = {
            "schema_version": 1,
            "status": "ok",
            "benchmark_qualified": False,
            "prompt_id": args.prompt_id,
            "expected_output_token_ids": expected,
            "inference": inference,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (json.JSONDecodeError, OSError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
