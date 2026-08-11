#!/usr/bin/env python3
"""Generate and check the deterministic Gemma 4 26B M05 FP8 plans."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

# Direct execution places ``tools/`` rather than the repository root on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gem16_compile.common import canonical_json_bytes
from tools.gem16_compile.profiles import (
    M05_ATTENTION_TABLE,
    M05_DEFERRED_REASON,
    M05_DEQUANTIZATION_EQUATION,
    M05_PROFILE,
    M05_QUANTIZER_PARAMETERS,
    M05_SOURCE_CONTRACT,
    M05_SOURCE_LOCK_SHA256,
    M05_VISION_EXCLUSION_REASON,
    classify_m05_source,
)


DEFAULT_ORDINARY_INVENTORY = (
    ROOT / "benchmarks/goldens/gemma4_26b/source-inventories/ordinary-bf16.json"
)
DEFAULT_QAT_INVENTORY = (
    ROOT / "benchmarks/goldens/gemma4_26b/source-inventories/qat-bf16.json"
)
DEFAULT_ORDINARY_LOCK = ROOT / "models/gemma4-26b-base-bf16.lock.json"
DEFAULT_QAT_LOCK = ROOT / "models/gemma4-26b-qat-bf16.lock.json"
DEFAULT_OUTPUT = ROOT / "benchmarks/goldens/gemma4_26b/fp8"
DEFAULT_CONFIG = ROOT / "artifacts/m05/fp8-compiler-config.json"

REFERENCE_ENVIRONMENT = {
    "byteorder": "little",
    "locale": "C.UTF-8",
    "machine": "x86_64",
    "python_implementation": "CPython",
    "python_major_minor": "3.14",
    "python_version": "3.14.6",
    "system": "Linux",
}
OMITTED_FAMILIES = ["audio", "mtp", "video", "vision"]
TARGET_SHARD_BYTES = 1 << 30


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in generated-plan input: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError(f"JSON input exceeds 64 MiB: {path}")
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
        if not isinstance(value, dict):
            raise ValueError(f"JSON root must be an object: {path}")
        return value
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON {path}: {error}") from error


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"cannot hash {path}: {error}") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_source_identity(
    inventory: dict[str, Any], lock_path: Path, expected_family: str
) -> tuple[str, str]:
    _require(inventory.get("schema_version") == 1, "unsupported inventory schema")
    _require(inventory.get("status") == "raw_source_inventory", "inventory is not raw")
    _require(inventory.get("source_family") == expected_family, "unexpected inventory family")
    source = inventory.get("source")
    checkpoint = inventory.get("checkpoint")
    _require(isinstance(source, dict), "inventory source record is malformed")
    _require(isinstance(checkpoint, dict), "inventory checkpoint record is malformed")
    lock_hash = sha256_file(lock_path)
    _require(source.get("lock_sha256") == lock_hash, f"inventory lock hash mismatch: {lock_path}")
    _require(
        lock_hash == M05_SOURCE_LOCK_SHA256[expected_family],
        f"unapproved M05 source lock for {expected_family}: {lock_path}",
    )
    try:
        relative_lock = lock_path.resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"lock path must be inside repository: {lock_path}") from error
    _require(source.get("lock_path") == relative_lock, f"inventory lock path mismatch: {lock_path}")
    lock = load_json(lock_path)
    _require(source.get("revision") == lock.get("revision"), f"inventory revision mismatch: {lock_path}")
    _require(source.get("repository") == lock.get("repository"), f"inventory repository mismatch: {lock_path}")
    tensors = inventory.get("tensors")
    _require(isinstance(tensors, list), "inventory tensors are malformed")
    _require(checkpoint.get("tensor_count") == len(tensors), "inventory tensor count mismatch")
    _require(checkpoint.get("tensor_payload_bytes") == sum(int(t["bytes"]) for t in tensors), "inventory payload total mismatch")
    names = [t.get("name") for t in tensors]
    _require(all(isinstance(name, str) and name for name in names), "inventory tensor name is malformed")
    _require(len(set(names)) == len(names), "inventory tensor names are not unique")
    return str(source["revision"]), lock_hash


def _tensor_map(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(tensor["name"]): tensor for tensor in inventory["tensors"]}


def _plan_tensor(
    source: dict[str, Any], encoder: str, output_name: str, transformation: str,
    output_dtype: str, output_shape: list[int], layout: str, role: str,
) -> dict[str, Any]:
    source_name = str(source["name"])
    return {
        "aliased": False,
        "axis_transformation": "identity",
        "dequantization_equation": M05_DEQUANTIZATION_EQUATION,
        "disk_layout": layout,
        "encoder": encoder,
        "logical_dtype": "BF16",
        "logical_shape": list(output_shape),
        "operation_id": f"fp8-attention:{source_name.removesuffix('.weight')}",
        "output_dtype": output_dtype,
        "output_name": output_name,
        "physical_shape": list(output_shape),
        "quantizer_parameters": dict(M05_QUANTIZER_PARAMETERS),
        "residency_class": "immutable_device_text",
        "role": role,
        "runtime_layout": layout,
        "source_names": [source_name],
        "transformation": transformation,
        "transformation_version": 1,
    }


def _attention_records(tensors: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, spec in M05_ATTENTION_TABLE.items():
        source = tensors.get(name)
        _require(source is not None, f"missing approved attention source: {name}")
        _require(source.get("dtype") == "BF16", f"attention source must be BF16: {name}")
        shape = source.get("shape")
        _require(tuple(shape or ()) == spec.shape, f"attention source shape mismatch: {name}")
        _require(classify_m05_source(name) == spec.role, f"attention role mismatch: {name}")
        rows, columns = spec.shape
        stem = name.removesuffix(".weight")
        records.append(_plan_tensor(
            source, "fp8-rowwise-scale-v1", f"{stem}.weight_scale",
            "bf16-to-bf16-rowwise-scale", "BF16", [rows, 1],
            "row_bf16", spec.role,
        ))
        records.append(_plan_tensor(
            source, "fp8-rowwise-weight-v1", name,
            "bf16-to-fp8-e4m3fn-rowwise-weight", "F8_E4M3",
            [rows, columns], "source_nk_fp8", spec.role,
        ))
    _require(len(records) == 230, f"expected 230 FP8 outputs, got {len(records)}")
    return sorted(records, key=lambda value: str(value["output_name"]))


def _exclusions(tensors: dict[str, dict[str, Any]], attention_sources: set[str]) -> list[dict[str, Any]]:
    exclusions: list[dict[str, Any]] = []
    for name in sorted(tensors):
        if name in attention_sources:
            continue
        try:
            role = classify_m05_source(name)
        except ValueError as error:
            raise ValueError(f"unknown M05 source tensor: {name}") from error
        if role.startswith("vision_"):
            exclusions.append({
                "family": "vision",
                "reason": M05_VISION_EXCLUSION_REASON,
                "residency_class": "compile_excluded_vision",
                "role": role,
                "source_name": name,
            })
            continue
        _require(role not in {
            "attention_q_projection", "attention_k_projection",
            "attention_v_projection", "attention_o_projection",
        }, f"attention source was not selected: {name}")
        exclusions.append({
            "family": "deferred_non_attention",
            "reason": M05_DEFERRED_REASON,
            "residency_class": "m05_deferred_non_attention",
            "role": role,
            "source_name": name,
        })
    _require(len(exclusions) == 898, f"expected 898 exclusions, got {len(exclusions)}")
    return exclusions


def make_plan(inventory_path: Path, lock_path: Path, expected_family: str) -> dict[str, Any]:
    inventory = load_json(inventory_path)
    _revision, lock_hash = _validate_source_identity(inventory, lock_path, expected_family)
    tensors = _tensor_map(inventory)
    records = _attention_records(tensors)
    attention_sources = {str(item["source_names"][0]) for item in records}
    exclusions = _exclusions(tensors, attention_sources)
    _require(len(attention_sources) == 115, f"expected 115 attention matrices, got {len(attention_sources)}")
    _require(len(tensors) == 1013, f"expected 1013 source tensors, got {len(tensors)}")
    return {
        "approved_metadata_files": [],
        "artifact_profile": M05_PROFILE.name,
        "excluded_tensors": exclusions,
        "head_format": M05_PROFILE.head_format,
        "omitted_families": OMITTED_FAMILIES,
        "reference_environment": REFERENCE_ENVIRONMENT,
        "schema_version": 1,
        "source_contract": M05_SOURCE_CONTRACT,
        "source_lock_sha256": lock_hash,
        "target_shard_bytes": TARGET_SHARD_BYTES,
        "tensors": records,
    }


def plan_bytes(plan: dict[str, Any]) -> bytes:
    return canonical_json_bytes(plan)


def _plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    tensors = plan["tensors"]
    source_names = {name for item in tensors for name in item["source_names"]}
    roles = Counter(item["role"] for item in tensors if item["encoder"] == "fp8-rowwise-weight-v1")
    source_bytes = 0
    output_bytes = 0
    inventory_by_name = _tensor_map(load_json(DEFAULT_ORDINARY_INVENTORY))
    for name in source_names:
        source_bytes += int(inventory_by_name[name]["bytes"])
    for item in tensors:
        shape = item["physical_shape"]
        element_bytes = 1 if item["output_dtype"] == "F8_E4M3" else 2
        output_bytes += element_bytes * shape[0] * shape[1]
    return {
        "attention_matrix_count": len(source_names),
        "attention_matrices_by_role": dict(sorted(roles.items())),
        "output_tensor_count": len(tensors),
        "output_tensor_bytes": output_bytes,
        "source_attention_tensor_bytes": source_bytes,
        "exclusion_count": len(plan["excluded_tensors"]),
        "vision_exclusion_count": sum(item["family"] == "vision" for item in plan["excluded_tensors"]),
        "deferred_exclusion_count": sum(item["family"] == "deferred_non_attention" for item in plan["excluded_tensors"]),
    }


def make_config(
    ordinary: dict[str, Any],
    qat: dict[str, Any],
    ordinary_bytes: bytes,
    qat_bytes: bytes,
    ordinary_lock: Path,
    qat_lock: Path,
) -> dict[str, Any]:
    ordinary_hash = hashlib.sha256(ordinary_bytes).hexdigest()
    qat_hash = hashlib.sha256(qat_bytes).hexdigest()
    ordinary_lock_document = load_json(ordinary_lock)
    qat_lock_document = load_json(qat_lock)
    quantizer_path = "tools/gem16_compile/specs/fp8-attention-rowwise-v1.json"
    quantizer_hash = sha256_file(ROOT / quantizer_path)
    return {
        "schema_version": 1,
        "milestone": "M05",
        "artifact_profile": M05_PROFILE.name,
        "head_format": M05_PROFILE.head_format,
        "source_contract": M05_SOURCE_CONTRACT,
        "quantizer": {
            "path": quantizer_path,
            "sha256": quantizer_hash,
            "contract_id": M05_QUANTIZER_PARAMETERS["contract_id"],
            "contract_version": M05_QUANTIZER_PARAMETERS["contract_version"],
        },
        "plans": {
            "ordinary_bf16": {
                "path": "benchmarks/goldens/gemma4_26b/fp8/ordinary-compiler-plan.json",
                "inventory": "benchmarks/goldens/gemma4_26b/source-inventories/ordinary-bf16.json",
                "sha256": ordinary_hash,
                "source_lock": "models/gemma4-26b-base-bf16.lock.json",
                "source_lock_sha256": ordinary["source_lock_sha256"],
                "source_revision": ordinary_lock_document["revision"],
            },
            "qat_bf16": {
                "path": "benchmarks/goldens/gemma4_26b/fp8/qat-compiler-plan.json",
                "inventory": "benchmarks/goldens/gemma4_26b/source-inventories/qat-bf16.json",
                "sha256": qat_hash,
                "source_lock": "models/gemma4-26b-qat-bf16.lock.json",
                "source_lock_sha256": qat["source_lock_sha256"],
                "source_revision": qat_lock_document["revision"],
            },
        },
        "attention": {
            "matrix_count": 115,
            "output_tensor_count": 230,
            "output_tensor_bytes": 1_110_850_560,
            "source_attention_bf16_bytes": 2_220_359_680,
            "fp8_weight_payload_bytes": 1_110_179_840,
            "bf16_scale_payload_bytes": 670_720,
            "matrices_by_role": {"attention_k_projection": 30, "attention_o_projection": 30, "attention_q_projection": 30, "attention_v_projection": 25},
            "global_layers_without_v": [5, 11, 17, 23, 29],
        },
        "exclusions": {
            "total_count": 898,
            "vision_count": 356,
            "deferred_non_attention_count": 542,
            "vision_reason": M05_VISION_EXCLUSION_REASON,
            "deferred_reason": M05_DEFERRED_REASON,
        },
        "deterministic_environment": REFERENCE_ENVIRONMENT,
    }


def generate(output_dir: Path, config_path: Path, ordinary_inventory: Path = DEFAULT_ORDINARY_INVENTORY, qat_inventory: Path = DEFAULT_QAT_INVENTORY, ordinary_lock: Path = DEFAULT_ORDINARY_LOCK, qat_lock: Path = DEFAULT_QAT_LOCK) -> dict[Path, bytes]:
    ordinary = make_plan(ordinary_inventory, ordinary_lock, "ordinary_bf16")
    qat = make_plan(qat_inventory, qat_lock, "qat_bf16")
    ordinary_bytes = plan_bytes(ordinary)
    qat_bytes = plan_bytes(qat)
    config = make_config(
        ordinary, qat, ordinary_bytes, qat_bytes, ordinary_lock, qat_lock
    )
    return {
        output_dir / "ordinary-compiler-plan.json": ordinary_bytes,
        output_dir / "qat-compiler-plan.json": qat_bytes,
        config_path: canonical_json_bytes(config),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary-inventory", type=Path, default=DEFAULT_ORDINARY_INVENTORY)
    parser.add_argument("--qat-inventory", type=Path, default=DEFAULT_QAT_INVENTORY)
    parser.add_argument("--ordinary-lock", type=Path, default=DEFAULT_ORDINARY_LOCK)
    parser.add_argument("--qat-lock", type=Path, default=DEFAULT_QAT_LOCK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    generated = generate(args.output_dir, args.config, args.ordinary_inventory, args.qat_inventory, args.ordinary_lock, args.qat_lock)
    mismatches: list[str] = []
    for path, payload in generated.items():
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
