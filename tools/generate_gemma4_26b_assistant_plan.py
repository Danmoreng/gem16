#!/usr/bin/env python3
"""Generate the locked M25 QAT-Q4_0 Assistant hybrid compiler plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

try:
    from tools.gem16_compile.common import canonical_json_bytes
    from tools.gem16_compile.profiles import (
        M05_DEQUANTIZATION_EQUATION,
        M05_QUANTIZER_PARAMETERS,
        M07_COMPONENT_LAYOUTS,
        M07_DEQUANTIZATION_EQUATION,
        M25_PROFILE,
        M25_SOURCE_CONTRACT,
        M25_SOURCE_LOCK_SHA256,
        m07_component_parameters,
    )
except ModuleNotFoundError:
    from gem16_compile.common import canonical_json_bytes  # type: ignore[no-redef]
    from gem16_compile.profiles import (  # type: ignore[no-redef]
        M05_DEQUANTIZATION_EQUATION,
        M05_QUANTIZER_PARAMETERS,
        M07_COMPONENT_LAYOUTS,
        M07_DEQUANTIZATION_EQUATION,
        M25_PROFILE,
        M25_SOURCE_CONTRACT,
        M25_SOURCE_LOCK_SHA256,
        m07_component_parameters,
    )


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = (
    ROOT / "benchmarks/goldens/gemma4_26b/source-inventories/"
    "qat-q4_0-assistant-bf16.json"
)
OUTPUT = (
    ROOT / "benchmarks/goldens/gemma4_26b/mtp-assistant/"
    "qat-q4_0-hybrid-compiler-plan.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def role_for(name: str) -> str:
    exact = {
        "model.embed_tokens.weight": "tied_embedding_and_output",
        "model.norm.weight": "final_norm",
        "pre_projection.weight": "assistant_pre_projection",
        "post_projection.weight": "assistant_post_projection",
    }
    if name in exact:
        return exact[name]
    suffix_roles = {
        "input_layernorm.weight": "input_layer_norm",
        "layer_scalar": "layer_scalar",
        "mlp.down_proj.weight": "assistant_mlp_down",
        "mlp.gate_proj.weight": "assistant_mlp_gate",
        "mlp.up_proj.weight": "assistant_mlp_up",
        "post_attention_layernorm.weight": "post_attention_layer_norm",
        "post_feedforward_layernorm.weight": "post_feed_forward_layer_norm",
        "pre_feedforward_layernorm.weight": "pre_feed_forward_layer_norm",
        "self_attn.o_proj.weight": "assistant_attention_o_projection",
        "self_attn.q_norm.weight": "assistant_attention_q_norm",
        "self_attn.q_proj.weight": "assistant_attention_q_projection",
    }
    for suffix, role in suffix_roles.items():
        if name.endswith("." + suffix):
            return role
    raise ValueError(f"unknown Assistant source tensor: {name}")


def copy_tensor(name: str, shape: list[int], role: str) -> dict[str, Any]:
    return {
        "output_name": name,
        "operation_id": f"copy-assistant:{name}",
        "source_names": [name],
        "encoder": "copy-v1",
        "transformation": "identity-copy",
        "transformation_version": 1,
        "output_dtype": "BF16",
        "physical_shape": shape,
        "logical_dtype": "BF16",
        "logical_shape": shape,
        "axis_transformation": "identity",
        "quantizer_parameters": {},
        "dequantization_equation": "output = source",
        "role": role,
        "residency_class": "immutable_device_mtp_assistant",
        "disk_layout": "source_bf16",
        "runtime_layout": "source_bf16",
        "aliased": False,
    }


def fp8_tensors(name: str, shape: list[int], role: str) -> list[dict[str, Any]]:
    rows, columns = shape
    stem = name.removesuffix(".weight")
    common = {
        "operation_id": f"fp8-assistant:{stem}",
        "source_names": [name],
        "transformation_version": 1,
        "logical_dtype": "BF16",
        "axis_transformation": "identity",
        "quantizer_parameters": dict(M05_QUANTIZER_PARAMETERS),
        "dequantization_equation": M05_DEQUANTIZATION_EQUATION,
        "role": role,
        "residency_class": "immutable_device_mtp_assistant",
        "aliased": False,
    }
    return [
        {
            **common,
            "output_name": name,
            "encoder": "fp8-rowwise-weight-v1",
            "transformation": "bf16-to-fp8-e4m3fn-rowwise-weight",
            "output_dtype": "F8_E4M3",
            "physical_shape": [rows, columns],
            "logical_shape": [rows, columns],
            "disk_layout": "source_nk_fp8",
            "runtime_layout": "source_nk_fp8",
        },
        {
            **common,
            "output_name": f"{stem}.weight_scale",
            "encoder": "fp8-rowwise-scale-v1",
            "transformation": "bf16-to-bf16-rowwise-scale",
            "output_dtype": "BF16",
            "physical_shape": [rows, 1],
            "logical_shape": [rows, 1],
            "disk_layout": "row_bf16",
            "runtime_layout": "row_bf16",
        },
    ]


def nvfp4_tensors(name: str, shape: list[int], role: str) -> list[dict[str, Any]]:
    rows, columns = shape
    stem = name.removesuffix(".weight")
    tied = role == "tied_embedding_and_output"
    result: list[dict[str, Any]] = []
    components = {
        "weight_packed": ("U8", [rows, columns // 2]),
        "weight_scale": ("F8_E4M3", [rows, columns // 16]),
        "weight_global_scale": ("F32", [1]),
        "input_global_scale": ("F32", [1]),
    }
    for component, (dtype, physical_shape) in components.items():
        layout = M07_COMPONENT_LAYOUTS[component]
        result.append({
            "output_name": f"{stem}.{component}",
            "operation_id": f"nvfp4-assistant:{stem}",
            "source_names": [name],
            "encoder": layout["encoder"],
            "transformation": layout["transformation"],
            "transformation_version": 1,
            "output_dtype": dtype,
            "physical_shape": physical_shape,
            "logical_dtype": "BF16",
            "logical_shape": shape,
            "axis_transformation": "vocabulary,hidden" if tied else "output,input",
            "quantizer_parameters": m07_component_parameters(component),
            "dequantization_equation": M07_DEQUANTIZATION_EQUATION,
            "role": role,
            "residency_class": "immutable_device_mtp_assistant",
            "disk_layout": layout["disk_layout"],
            "runtime_layout": layout["runtime_layout_shared"],
            "aliased": tied,
        })
    return result


def generate() -> bytes:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    source_tensors = inventory.get("tensors")
    if not isinstance(source_tensors, list) or len(source_tensors) != 48:
        raise ValueError("locked Assistant inventory must contain exactly 48 tensors")
    tensors: list[dict[str, Any]] = []
    nvfp4_roles = {
        "tied_embedding_and_output", "assistant_mlp_down",
        "assistant_mlp_gate", "assistant_mlp_up",
    }
    fp8_roles = {
        "assistant_attention_q_projection", "assistant_attention_o_projection",
        "assistant_pre_projection", "assistant_post_projection",
    }
    for source in source_tensors:
        name = source["name"]
        shape = source["shape"]
        if source["dtype"] != "BF16":
            raise ValueError(f"Assistant source is not BF16: {name}")
        role = role_for(name)
        if role in nvfp4_roles:
            tensors.extend(nvfp4_tensors(name, shape, role))
        elif role in fp8_roles:
            tensors.extend(fp8_tensors(name, shape, role))
        else:
            tensors.append(copy_tensor(name, shape, role))
    document = {
        "schema_version": 1,
        "artifact_profile": M25_PROFILE.name,
        "head_format": M25_PROFILE.head_format,
        "source_contract": M25_SOURCE_CONTRACT,
        "source_lock_sha256": M25_SOURCE_LOCK_SHA256,
        "target_shard_bytes": 1024 * 1024 * 1024,
        "approved_metadata_files": [
            "chat_template.jinja", "config.json", "generation_config.json",
            "tokenizer.json", "tokenizer_config.json",
        ],
        "omitted_families": ["audio", "video", "vision"],
        "tensors": sorted(tensors, key=lambda item: item["output_name"]),
        "excluded_tensors": [],
        "reference_environment": {
            "byteorder": "little",
            "locale": "C.UTF-8",
            "machine": "x86_64",
            "python_implementation": "CPython",
            "python_major_minor": "3.14",
            "python_version": "3.14.6",
            "system": "Linux",
        },
    }
    return canonical_json_bytes(document)


def main() -> int:
    args = parse_args()
    payload = generate()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != payload:
            print(f"stale generated Assistant compiler plan: {OUTPUT}", file=sys.stderr)
            return 1
        print(f"Assistant compiler plan is current: {OUTPUT}")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(payload)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
