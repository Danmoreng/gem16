#!/usr/bin/env python3
"""Verify a C++ schema-3 26B manifest against the frozen M03 fixture."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


MAX_MANIFEST_BYTES = 256 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("source_bf16", "external_unsloth_nvfp4"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, maximum_bytes: int) -> dict[str, Any]:
    size = path.stat().st_size
    if size > maximum_bytes:
        raise ValueError(f"JSON exceeds {maximum_bytes} bytes: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def summarize(
    manifest: dict[str, Any], fixture: dict[str, Any], profile: str
) -> dict[str, Any]:
    fixture_key = (
        "source_bf16" if profile == "source_bf16" else "external_unsloth_nvfp4"
    )
    expected = fixture[fixture_key]
    errors: list[str] = []
    scalar_expectations = {
        "schema_version": 3,
        "model_variant": "gemma4_moe_26b_a4b",
        "checkpoint_profile": profile,
        "validation_contract": "gemma4_26b_m03_exact_inventory_v1",
        "runtime_supported": False,
        "tensor_contract_validated": True,
        "total_tensor_bytes": expected["payload_bytes"],
        "skipped_tensor_bytes": expected["compile_excluded_vision_bytes"],
        "text_only_tensor_bytes": expected["text_bytes"],
    }
    for key, value in scalar_expectations.items():
        if manifest.get(key) != value:
            errors.append(f"{key}: expected {value!r}, got {manifest.get(key)!r}")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError("manifest tensors must be an array")
    if len(tensors) != expected["tensor_count"]:
        errors.append(
            f"tensor_count: expected {expected['tensor_count']}, got {len(tensors)}"
        )

    by_role: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_residency: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    names: set[str] = set()
    layer_experts: dict[int, set[int]] = defaultdict(set)
    unknown_metadata = 0
    for tensor in tensors:
        if not isinstance(tensor, dict):
            errors.append("tensor entry must be an object")
            continue
        name = tensor.get("name")
        if not isinstance(name, str) or not name or name in names:
            errors.append(f"missing or duplicate tensor name: {name!r}")
            continue
        names.add(name)
        role = tensor.get("tensor_role")
        residency = tensor.get("residency_class")
        byte_length = tensor.get("byte_length")
        if not isinstance(role, str) or not role or role == "unknown":
            unknown_metadata += 1
            continue
        if not isinstance(residency, str) or not residency or residency == "unknown":
            unknown_metadata += 1
            continue
        if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length < 0:
            errors.append(f"invalid byte length: {name}")
            continue
        required_strings = (
            "logical_dtype",
            "source_family",
            "quantization_component",
            "quantization_producer",
            "local_scale_dtype",
            "global_scale_role",
            "activation_scale_role",
            "final_gpu_layout",
        )
        if any(not isinstance(tensor.get(key), str) or not tensor[key] for key in required_strings):
            unknown_metadata += 1
        by_role[role][0] += 1
        by_role[role][1] += byte_length
        by_residency[residency][0] += 1
        by_residency[residency][1] += byte_length
        layer = tensor.get("layer_index")
        expert = tensor.get("expert_index")
        if profile == "external_unsloth_nvfp4" and isinstance(layer, int) and isinstance(expert, int) and expert >= 0:
            layer_experts[layer].add(expert)
    if unknown_metadata:
        errors.append(f"{unknown_metadata} tensors have unknown/incomplete semantic metadata")

    role_total_bytes = sum(values[1] for values in by_role.values())
    residency_total_bytes = sum(values[1] for values in by_residency.values())
    if role_total_bytes != expected["payload_bytes"]:
        errors.append("role bytes do not reconcile with payload")
    if residency_total_bytes != expected["payload_bytes"]:
        errors.append("residency bytes do not reconcile with payload")
    vision = by_residency.get("compile_excluded_vision", [0, 0])
    if vision != [
        expected["compile_excluded_vision_tensor_count"],
        expected["compile_excluded_vision_bytes"],
    ]:
        errors.append(f"vision residency mismatch: {vision}")
    if any("mtp" in name.lower() for name in names):
        errors.append("MTP tensor found in the first 26B profile")
    if "lm_head.weight" in names or "model.language_model.lm_head.weight" in names:
        errors.append("duplicate tied LM head found")

    embedding = next(
        (tensor for tensor in tensors if tensor.get("name") == "model.language_model.embed_tokens.weight"),
        None,
    )
    if embedding is None or embedding.get("aliased") is not True:
        errors.append("tied embedding alias marker is absent")
    if profile == "source_bf16":
        gate_up = next(
            (
                tensor
                for tensor in tensors
                if tensor.get("name")
                == "model.language_model.layers.0.experts.gate_up_proj"
            ),
            None,
        )
        if gate_up is None or gate_up.get("expert_axis") != 0 or gate_up.get(
            "logical_axis_order"
        ) != "expert,gate_then_up,input":
            errors.append("source fused Gate/Up axis/order contract mismatch")
    else:
        for layer in range(30):
            if layer_experts[layer] != set(range(128)):
                errors.append(f"external expert coverage mismatch at layer {layer}")
        if manifest.get("checkpoint_profile") == "gem16_compiled_hybrid":
            errors.append("external Unsloth manifest was accepted as project compiled")

    manifest_roles = {
        item["role"]: [item["tensor_count"], item["bytes"]]
        for item in manifest.get("totals_by_role", [])
    }
    manifest_residency = {
        item["residency_class"]: [item["tensor_count"], item["bytes"]]
        for item in manifest.get("totals_by_residency", [])
    }
    if manifest_roles != dict(sorted(by_role.items())):
        errors.append("manifest totals_by_role disagrees with tensor records")
    if manifest_residency != dict(sorted(by_residency.items())):
        errors.append("manifest totals_by_residency disagrees with tensor records")

    return {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "profile": profile,
        "tensor_count": len(tensors),
        "tensor_payload_bytes": role_total_bytes,
        "unknown_semantic_tensor_count": unknown_metadata,
        "totals_by_role": {
            role: {"tensor_count": values[0], "bytes": values[1]}
            for role, values in sorted(by_role.items())
        },
        "totals_by_residency": {
            residency: {"tensor_count": values[0], "bytes": values[1]}
            for residency, values in sorted(by_residency.items())
        },
        "external_experts_complete": (
            all(layer_experts[layer] == set(range(128)) for layer in range(30))
            if profile == "external_unsloth_nvfp4"
            else None
        ),
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    try:
        manifest = load_json(args.manifest, MAX_MANIFEST_BYTES)
        fixture = load_json(args.fixture, 1024 * 1024)
        report = summarize(manifest, fixture, args.profile)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as error:
        print(f"error: {error}")
        return 2
    print(f"{args.profile}: {report['status']} ({report['tensor_count']} tensors)")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
