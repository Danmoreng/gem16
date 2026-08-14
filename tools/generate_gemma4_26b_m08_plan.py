#!/usr/bin/env python3
"""Generate the deterministic complete M08 Gemma 4 26B hybrid plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gem16_compile.common import canonical_json_bytes
from tools.gem16_compile.profiles import (
    M05_VISION_EXCLUSION_REASON,
    M08_PROFILE,
    M08_SOURCE_CONTRACT,
    classify_m05_source,
    m06_expected_source_specs,
)

INVENTORY = ROOT / "benchmarks/goldens/gemma4_26b/source-inventories/qat-bf16.json"
QAT_LOCK = ROOT / "models/gemma4-26b-qat-bf16.lock.json"
FP8_PLAN = ROOT / "benchmarks/goldens/gemma4_26b/fp8/qat-compiler-plan.json"
NVFP4_PLAN = ROOT / "benchmarks/goldens/gemma4_26b/nvfp4/qat-compiler-plan.json"
HEAD_PLAN = ROOT / "benchmarks/goldens/gemma4_26b/nvfp4/qat-head-compiler-plan.json"
OUTPUT_PLAN = ROOT / "benchmarks/goldens/gemma4_26b/hybrid/qat-compiler-plan.json"
OUTPUT_SUMMARY = ROOT / "artifacts/m08/compiler-plan-summary.json"

REFERENCE_ENVIRONMENT = {
    "byteorder": "little",
    "locale": "C.UTF-8",
    "machine": "x86_64",
    "python_implementation": "CPython",
    "python_major_minor": "3.14",
    "python_version": "3.14.6",
    "system": "Linux",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_record(name: str, role: str, shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "aliased": False,
        "axis_transformation": "identity",
        "dequantization_equation": "output = source",
        "disk_layout": "source_bf16",
        "encoder": "copy-v1",
        "logical_dtype": "BF16",
        "logical_shape": list(shape),
        "operation_id": f"copy:{name}",
        "output_dtype": "BF16",
        "output_name": name,
        "physical_shape": list(shape),
        "quantizer_parameters": {},
        "residency_class": "immutable_device_text",
        "role": role,
        "runtime_layout": "source_bf16",
        "source_names": [name],
        "transformation": "identity-copy",
        "transformation_version": 1,
    }


def _cache_scale_record(layer: int, component: str) -> dict[str, Any]:
    name = f"model.language_model.layers.{layer}.self_attn.{component}_scale"
    return {
        "aliased": False,
        "axis_transformation": "scalar",
        "dequantization_equation": "output = BF16(1.0)",
        "disk_layout": "scalar_bf16",
        "encoder": "constant-bf16-one-v1",
        "logical_dtype": "BF16",
        "logical_shape": [1],
        "operation_id": f"constant-bf16-one:{name}",
        "output_dtype": "BF16",
        "output_name": name,
        "physical_shape": [1],
        "quantizer_parameters": {"encoding": "BF16-RNE", "value": 1.0},
        "residency_class": "immutable_device_text",
        "role": f"attention_{component}_cache_scale",
        "runtime_layout": "scalar_bf16",
        "source_names": [],
        "transformation": "constant-bf16-one",
        "transformation_version": 1,
    }


def make_plan() -> dict[str, Any]:
    inventory = _load(INVENTORY)
    specs = m06_expected_source_specs()
    source = {item["name"]: item for item in inventory["tensors"]}
    if set(source) != set(specs) or len(source) != 1013:
        raise ValueError("M08 requires the frozen 1,013-name QAT source inventory")

    fp8 = _load(FP8_PLAN)["tensors"]
    nvfp4 = _load(NVFP4_PLAN)["tensors"]
    head = _load(HEAD_PLAN)["tensors"]
    transformed_sources = {
        name
        for tensor in (*fp8, *nvfp4, *head)
        for name in tensor["source_names"]
    }
    tensors: list[dict[str, Any]] = [*fp8, *nvfp4, *head]
    exclusions: list[dict[str, Any]] = []
    copied = 0
    for name, (role, shape) in specs.items():
        descriptor = source[name]
        if descriptor.get("dtype") != "BF16" or tuple(descriptor.get("shape", ())) != shape:
            raise ValueError(f"M08 frozen source descriptor mismatch: {name}")
        if role.startswith("vision_"):
            exclusions.append({
                "family": "vision",
                "reason": M05_VISION_EXCLUSION_REASON,
                "residency_class": "compile_excluded_vision",
                "role": role,
                "source_name": name,
            })
        elif name not in transformed_sources:
            tensors.append(_copy_record(name, role, shape))
            copied += 1
    for layer in range(30):
        for component in ("k", "v"):
            tensors.append(_cache_scale_record(layer, component))

    tensors.sort(key=lambda item: item["output_name"])
    exclusions.sort(key=lambda item: item["source_name"])
    if len(transformed_sources) != 266 or copied != 391:
        raise ValueError(
            f"M08 source partition mismatch: transformed={len(transformed_sources)} copied={copied}"
        )
    if len(tensors) != 1285 or len(exclusions) != 356:
        raise ValueError(
            f"M08 inventory mismatch: outputs={len(tensors)} exclusions={len(exclusions)}"
        )
    return {
        "approved_metadata_files": [
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        "artifact_profile": M08_PROFILE.name,
        "excluded_tensors": exclusions,
        "head_format": M08_PROFILE.head_format,
        "omitted_families": ["audio", "mtp", "video", "vision"],
        "reference_environment": REFERENCE_ENVIRONMENT,
        "schema_version": 1,
        "source_contract": M08_SOURCE_CONTRACT,
        "source_lock_sha256": _sha256(QAT_LOCK),
        "target_shard_bytes": 1 << 30,
        "tensors": tensors,
    }


def make_summary(plan: dict[str, Any]) -> dict[str, Any]:
    by_encoder: dict[str, int] = {}
    output_bytes = 0
    element_size = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
    for tensor in plan["tensors"]:
        encoder = tensor["encoder"]
        by_encoder[encoder] = by_encoder.get(encoder, 0) + 1
        size = element_size[tensor["output_dtype"]]
        for extent in tensor["physical_shape"]:
            size *= extent
        output_bytes += size
    return {
        "schema_version": 1,
        "milestone": "M08",
        "artifact_profile": M08_PROFILE.name,
        "source_lock_sha256": plan["source_lock_sha256"],
        "compiler_plan_sha256": hashlib.sha256(canonical_json_bytes(plan)).hexdigest(),
        "source_tensor_count": 1013,
        "output_tensor_count": len(plan["tensors"]),
        "output_tensor_bytes": output_bytes,
        "excluded_vision_tensor_count": len(plan["excluded_tensors"]),
        "outputs_by_encoder": dict(sorted(by_encoder.items())),
        "one_physical_tied_head": True,
        "runtime_quantization_required": False,
    }


def generate() -> dict[Path, bytes]:
    plan = make_plan()
    return {
        OUTPUT_PLAN: canonical_json_bytes(plan),
        OUTPUT_SUMMARY: canonical_json_bytes(make_summary(plan)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    mismatches: list[str] = []
    for path, payload in generate().items():
        if args.check:
            try:
                if path.read_bytes() != payload:
                    mismatches.append(str(path))
            except OSError:
                mismatches.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    if mismatches:
        raise SystemExit("generated outputs differ: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
