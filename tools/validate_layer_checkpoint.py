#!/usr/bin/env python3
"""Run the real-checkpoint attention and decoder-layer characterization gates."""

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
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_probe(bench: Path, model: Path, projection: str) -> dict[str, Any]:
    command = [
        str(bench),
        "kernel",
        "--model",
        str(model),
        "--projection",
        projection,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ValidationError(
            f"{projection} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValidationError(f"{projection} did not emit exactly one JSON document")
    try:
        document = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValidationError(f"{projection} emitted invalid JSON: {error}") from error
    if document.get("status") != "characterization":
        raise ValidationError(f"{projection} has unexpected status")
    if document.get("benchmark_qualified") is not False:
        raise ValidationError(f"{projection} must not claim benchmark qualification")
    if document.get("fallbacks") != 0:
        raise ValidationError(f"{projection} used a fallback")
    if projection == "decoder-layer":
        if document.get("packed_weight_source_layout_direct") is not True:
            raise ValidationError(f"{projection} did not use direct packed weights")
        if document.get("weight_scale_layout") != "sm120_row8_k64":
            raise ValidationError(f"{projection} did not use tiled SM120 weight scales")
        if document.get("load_time_scale_swizzle") is not True:
            raise ValidationError(f"{projection} omitted load-time scale swizzling")
    elif document.get("source_layout_direct") is not True:
        raise ValidationError(f"{projection} did not use its direct source layout")
    if document.get("persistent_repack_bytes") != 0:
        raise ValidationError(f"{projection} retained a persistent repack")
    error_metrics = document.get("error")
    if not isinstance(error_metrics, dict):
        raise ValidationError(f"{projection} omitted error metrics")
    for name in (
        "reference_native_max_abs",
        "reference_native_rms",
        "reference_native_cosine",
    ):
        value = error_metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValidationError(f"{projection} has a non-finite {name}")
    return document


def main() -> int:
    args = parse_args()
    try:
        bench = args.bench.resolve(strict=True)
        model = args.model.resolve(strict=True)
        probes = {
            projection: run_probe(bench, model, projection)
            for projection in (
                "local-attention",
                "global-attention",
                "decoder-layer",
            )
        }
        decoder = probes["decoder-layer"]
        if decoder.get("no_host_roundtrip_between_sublayers") is not True:
            raise ValidationError("decoder-layer contains a host roundtrip")
        if decoder.get("layer_scalar_applied") is not True:
            raise ValidationError("decoder-layer omitted the layer scalar")
        if decoder.get("mlp_input_mismatched_bytes") != 0:
            raise ValidationError("decoder-layer MLP-input quantization paths differ")
        if decoder.get("down_input_mismatched_bytes") != 0:
            raise ValidationError("decoder-layer Down-input quantization paths differ")
        result = {
            "schema_version": 1,
            "status": "ok",
            "benchmark_qualified": False,
            "model_directory": str(model),
            "probes": probes,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
