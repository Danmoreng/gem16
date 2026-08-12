#!/usr/bin/env python3
"""Generate the deterministic M06 QAT NVFP4 expert compiler plan."""

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

from tools.gem16_compile.common import canonical_json_bytes, tensor_bytes
from tools.gem16_compile.profiles import (
    M05_SOURCE_LOCK_SHA256,
    M06_COMPONENT_LAYOUTS,
    M06_DEFERRED_REASON,
    M06_DEQUANTIZATION_EQUATION,
    M06_PROFILE,
    M06_QUANTIZER_PARAMETERS,
    M06_SOURCE_CONTRACT,
    m06_component_parameters,
    m06_expected_source_specs,
    M06_VISION_EXCLUSION_REASON,
    classify_m05_source,
)

INVENTORY = ROOT / "benchmarks/goldens/gemma4_26b/source-inventories/qat-bf16.json"
QAT_LOCK = ROOT / "models/gemma4-26b-qat-bf16.lock.json"
ORDINARY_LOCK = ROOT / "models/gemma4-26b-base-bf16.lock.json"
UNSLOTH_LOCK = ROOT / "models/gemma4-26b-unsloth-nvfp4.lock.json"
OUTPUT_PLAN = ROOT / "benchmarks/goldens/gemma4_26b/nvfp4/qat-compiler-plan.json"
OUTPUT_CONFIG = ROOT / "artifacts/m06/nvfp4-compiler-config.json"
OUTPUT_SPEC = ROOT / "tools/gem16_compile/specs/nvfp4-experts-v1.json"
OMITTED_FAMILIES = ["audio", "mtp", "video", "vision"]
REFERENCE_ENVIRONMENT = {
    "byteorder": "little",
    "locale": "C.UTF-8",
    "machine": "x86_64",
    "python_implementation": "CPython",
    "python_major_minor": "3.14",
    "python_version": "3.14.6",
    "system": "Linux",
}
EXPERT_ROLES = frozenset({
    "shared_mlp_gate", "shared_mlp_up", "shared_mlp_down",
    "routed_expert_gate_up", "routed_expert_down",
})
EXPECTED_SHAPES = {
    "shared_mlp_gate": (2112, 2816),
    "shared_mlp_up": (2112, 2816),
    "shared_mlp_down": (2816, 2112),
    "routed_expert_gate_up": (128, 1408, 2816),
    "routed_expert_down": (128, 2816, 704),
}
SAMPLE_LAYERS = (0, 5, 24, 29)
SAMPLE_EXPERTS = (0, 63, 127)
SAMPLE_PROJECTIONS = ("gate", "up", "down")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_ref(path: Path) -> dict[str, str]:
    document = load_json(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "repository": str(document["repository"]),
        "revision": str(document["revision"]),
    }


def component_parameters(component: str) -> dict[str, Any]:
    return m06_component_parameters(component)


def source_role(name: str) -> str:
    return classify_m05_source(name)


def output_records(source: dict[str, Any], role: str) -> list[dict[str, Any]]:
    name = str(source["name"])
    shape = tuple(int(value) for value in source["shape"])
    if source.get("dtype") != "BF16" or shape != EXPECTED_SHAPES[role]:
        raise ValueError(f"unexpected M06 source contract: {name}")
    stem = name.removesuffix(".weight")
    routed = role.startswith("routed_")
    axis = (
        "expert,gate_then_up,input" if role == "routed_expert_gate_up" else
        "expert,output,input" if role == "routed_expert_down" else
        "output,input"
    )
    operation = f"nvfp4-experts:{stem}"
    common = {
        "aliased": False,
        "axis_transformation": axis,
        "dequantization_equation": M06_DEQUANTIZATION_EQUATION,
        "logical_dtype": "BF16",
        "logical_shape": list(shape),
        "operation_id": operation,
        "residency_class": "immutable_device_text",
        "role": role,
        "source_names": [name],
        "transformation_version": 1,
    }
    packed_shape = list(shape[:-1]) + [shape[-1] // 2]
    scale_shape = list(shape[:-1]) + [shape[-1] // 16]
    records = []
    for component, layout, physical_shape in (
        ("weight_packed", M06_COMPONENT_LAYOUTS["weight_packed"], packed_shape),
        ("weight_scale", M06_COMPONENT_LAYOUTS["weight_scale"], scale_shape),
        ("weight_global_scale", M06_COMPONENT_LAYOUTS["weight_global_scale"], [1]),
        ("input_global_scale", M06_COMPONENT_LAYOUTS["input_global_scale"], [1]),
    ):
        record = dict(common)
        record.update({
            "disk_layout": layout["disk_layout"],
            "encoder": layout["encoder"],
            "output_dtype": layout["output_dtype"],
            "output_name": f"{stem}.{component}",
            "physical_shape": physical_shape,
            "quantizer_parameters": component_parameters(component),
            "runtime_layout": layout["runtime_layout_routed" if routed else "runtime_layout_shared"],
            "transformation": layout["transformation"],
        })
        records.append(record)
    return records


def make_plan() -> dict[str, Any]:
    inventory = load_json(INVENTORY)
    if inventory.get("source_family") != "qat_bf16":
        raise ValueError("M06 requires the QAT BF16 inventory")
    tensors = inventory.get("tensors")
    expected_specs = m06_expected_source_specs()
    if not isinstance(tensors, list) or len(tensors) != len(expected_specs):
        raise ValueError("expected the complete 1,013-name QAT source inventory")
    actual_names = {item.get("name") for item in tensors if isinstance(item, dict)}
    if actual_names != set(expected_specs):
        raise ValueError("QAT inventory names do not match the frozen M06 inventory")
    outputs: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    expert_count = 0
    for source in sorted(tensors, key=lambda item: str(item["name"])):
        name = str(source["name"])
        role = source_role(name)
        expected_role, expected_shape = expected_specs[name]
        if (
            role != expected_role
            or source.get("dtype") != "BF16"
            or tuple(source.get("shape", ())) != expected_shape
            or source.get("bytes") != tensor_bytes("BF16", expected_shape, name)
        ):
            raise ValueError(f"unexpected frozen M06 source descriptor: {name}")
        if role in EXPERT_ROLES:
            outputs.extend(output_records(source, role))
            expert_count += 1
        else:
            vision = role.startswith("vision_")
            exclusions.append({
                "family": "vision" if vision else "deferred_non_expert",
                "reason": M06_VISION_EXCLUSION_REASON if vision else M06_DEFERRED_REASON,
                "residency_class": "compile_excluded_vision" if vision else "m06_deferred_non_expert",
                "role": role,
                "source_name": name,
            })
    if expert_count != 150 or len(outputs) != 600 or len(exclusions) != 863:
        raise ValueError(f"M06 counts mismatch: sources={expert_count}, outputs={len(outputs)}, exclusions={len(exclusions)}")
    return {
        "approved_metadata_files": [],
        "artifact_profile": M06_PROFILE.name,
        "excluded_tensors": exclusions,
        "head_format": M06_PROFILE.head_format,
        "omitted_families": OMITTED_FAMILIES,
        "reference_environment": REFERENCE_ENVIRONMENT,
        "schema_version": 1,
        "source_contract": M06_SOURCE_CONTRACT,
        "source_lock_sha256": sha256_file(QAT_LOCK),
        "target_shard_bytes": 1 << 30,
        "tensors": sorted(outputs, key=lambda item: item["output_name"]),
    }


def sampled_names() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for layer in SAMPLE_LAYERS:
        for projection in SAMPLE_PROJECTIONS:
            for kind, expert in [("shared", None), *[("routed", value) for value in SAMPLE_EXPERTS]]:
                if kind == "shared":
                    source = f"model.language_model.layers.{layer}.mlp.{projection}_proj.weight"
                    unsloth = f"model.language_model.layers.{layer}.mlp.{projection}_proj.weight_packed"
                    source_slice = {"axis": "none", "start": [0, 0], "stop": list(EXPECTED_SHAPES[f"shared_mlp_{projection}"])}
                elif projection == "down":
                    source = f"model.language_model.layers.{layer}.experts.down_proj"
                    unsloth = f"model.language_model.layers.{layer}.experts.{expert}.down_proj.weight_packed"
                    source_slice = {"axis": "expert", "start": [expert, 0, 0], "stop": [expert + 1, 2816, 704]}
                else:
                    source = f"model.language_model.layers.{layer}.experts.gate_up_proj"
                    unsloth = f"model.language_model.layers.{layer}.experts.{expert}.{projection}_proj.weight_packed"
                    row_start = 0 if projection == "gate" else 704
                    source_slice = {"axis": "expert,projection", "start": [expert, row_start, 0], "stop": [expert + 1, row_start + 704, 2816]}
                records.append({
                    "kind": kind,
                    "layer": layer,
                    **({"expert": expert} if expert is not None else {}),
                    "projection": projection,
                    "source_tensor": source,
                    "source_slice": source_slice,
                    "unsloth_component": unsloth,
                    "unsloth_component_range": {"name_template": unsloth, "range": "full_component"},
                })
    return records


def make_spec() -> dict[str, Any]:
    spec = dict(M06_QUANTIZER_PARAMETERS)
    spec.update({
        "schema_version": 1,
        "milestone": "M06",
        "disk_layout": "canonical_row_major_low_nibble_first",
        "local_scale_disk_layout": "canonical_row_major_group16_e4m3",
        "scalar_disk_layout": "scalar_f32",
        "runtime_layout_shared": "sm120_row8_k64",
        "runtime_layout_routed": "expert_major_sm120_row8_k64",
        "dequantization_equation": M06_DEQUANTIZATION_EQUATION,
        "encoders": sorted(M06_PROFILE.allowed_encoders),
    })
    return spec


def make_config(plan: dict[str, Any]) -> dict[str, Any]:
    outputs = plan["tensors"]
    output_bytes = sum(tensor_bytes(item["output_dtype"], tuple(item["physical_shape"]), item["output_name"]) for item in outputs)
    return {
        "schema_version": 1,
        "milestone": "M06",
        "artifact_profile": M06_PROFILE.name,
        "source_contract": M06_SOURCE_CONTRACT,
        "source": {"qat_bf16": lock_ref(QAT_LOCK)},
        "diagnostic_references": {
            "ordinary_bf16": lock_ref(ORDINARY_LOCK),
            "unsloth_nvfp4": lock_ref(UNSLOTH_LOCK),
            "status": "not_run",
            "byte_identity_required": False,
        },
        "quantizer": {
            "spec_path": "tools/gem16_compile/specs/nvfp4-experts-v1.json",
            "spec_sha256": hashlib.sha256(canonical_json_bytes(make_spec())).hexdigest(),
            "contract": dict(M06_QUANTIZER_PARAMETERS),
        },
        "diagnostic_sample": {
            "layers": list(SAMPLE_LAYERS),
            "experts": list(SAMPLE_EXPERTS),
            "projections": list(SAMPLE_PROJECTIONS),
            "logical_matrix_count": 48,
            "records": sampled_names(),
            "acceptance": {
                "status": "not_run",
                "matrix_count": {"exact": 48},
                "per_matrix": {
                    "elements": {"minimum": 1},
                    "all_finite": True,
                    "required_metrics": ["relative_l2", "cosine", "max_absolute_error", "sqnr", "code_histogram", "scale_histogram"],
                },
                "required_structure": [
                    "exact_shape_and_component_mapping",
                    "divisor_direction_and_relationships",
                    "fused_gate_up_split_order",
                    "matching_unsloth_component_and_range",
                    "full_parent_source_divisor_granularity",
                ],
                "thresholds": {
                    "ordinary_compiled": {"relative_l2_max": 0.25, "cosine_min": 0.95, "sqnr_min": 10.0},
                    "unsloth_reference": {"relative_l2_max": 0.25, "cosine_min": 0.95, "sqnr_min": 10.0},
                    "compiled_vs_unsloth": {"relative_l2_max": 0.25, "cosine_min": 0.95, "sqnr_min": 10.0},
                },
            },
        },
        "counts": {
            "source_tensor_count": 1013,
            "expert_shared_source_count": 150,
            "output_tensor_count": 600,
            "excluded_tensor_count": 863,
            "output_tensor_bytes": output_bytes,
            "expert_shared_output_tensor_bytes": 13_147_454_640,
        },
        "deterministic_environment": REFERENCE_ENVIRONMENT,
        "diagnostics": {"ordinary_vs_unsloth": "not_run", "status": "not_run", "quality_claim": False},
    }


def generate() -> dict[Path, bytes]:
    plan = make_plan()
    config = make_config(plan)
    spec = make_spec()
    return {
        OUTPUT_PLAN: canonical_json_bytes(plan),
        OUTPUT_CONFIG: canonical_json_bytes(config),
        OUTPUT_SPEC: canonical_json_bytes(spec),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-plan", type=Path, default=OUTPUT_PLAN)
    parser.add_argument("--output-config", type=Path, default=OUTPUT_CONFIG)
    parser.add_argument("--output-spec", type=Path, default=OUTPUT_SPEC)
    args = parser.parse_args(argv)
    generated = generate()
    if args.output_plan != OUTPUT_PLAN or args.output_config != OUTPUT_CONFIG or args.output_spec != OUTPUT_SPEC:
        generated = {
            args.output_plan: generated[OUTPUT_PLAN],
            args.output_config: generated[OUTPUT_CONFIG],
            args.output_spec: generated[OUTPUT_SPEC],
        }
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
