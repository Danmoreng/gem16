#!/usr/bin/env python3
"""Generate compact M03 Gemma 4 26B tensor-contract evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "benchmarks/goldens/gemma4_26b/source-inventories"
OUTPUT_ROOT = ROOT / "benchmarks/goldens/gemma4_26b/manifests"
FIXTURE = ROOT / "tests/fixtures/gemma4_26b_inventory.json"
REFERENCE_LOCK = ROOT / "models/gemma4-26b-reference-sources.lock.json"
LOCKS = {
    "qat_bf16": ROOT / "models/gemma4-26b-qat-bf16.lock.json",
    "ordinary_bf16": ROOT / "models/gemma4-26b-base-bf16.lock.json",
    "unsloth_nvfp4": ROOT / "models/gemma4-26b-unsloth-nvfp4.lock.json",
    "official_q4_0": ROOT / "models/gemma4-26b-qat-q4_0.lock.json",
}
RAW = {
    "qat_bf16": RAW_ROOT / "qat-bf16.json",
    "ordinary_bf16": RAW_ROOT / "ordinary-bf16.json",
    "unsloth_nvfp4": RAW_ROOT / "unsloth-nvfp4.json",
    "official_q4_0": RAW_ROOT / "google-q4_0.json",
}

LAYER_COUNT = 30
HIDDEN = 2816
SHARED = 2112
ROUTED = 704
EXPERTS = 128
VOCABULARY = 262144
VISION_HIDDEN = 1152
VISION_INTERMEDIATE = 4304
VISION_LAYERS = 27
VISION_BYTES = 1_145_588_832
SOURCE_PAYLOAD_BYTES = 51_611_872_412
UNSLOTH_PAYLOAD_BYTES = 16_903_408_612
Q4_FILE_BYTES = 14_439_363_584
MIB = 1024 * 1024
ARENA_ALIGNMENT = 256

SOURCE_LAYER_ROLES = {
    "input_layernorm.weight": "input_layer_norm",
    "layer_scalar": "layer_scalar",
    "mlp.down_proj.weight": "shared_mlp_down",
    "mlp.gate_proj.weight": "shared_mlp_gate",
    "mlp.up_proj.weight": "shared_mlp_up",
    "post_attention_layernorm.weight": "post_attention_layer_norm",
    "post_feedforward_layernorm.weight": "post_feed_forward_layer_norm",
    "post_feedforward_layernorm_1.weight": "post_feed_forward_layer_norm_1",
    "post_feedforward_layernorm_2.weight": "post_feed_forward_layer_norm_2",
    "pre_feedforward_layernorm.weight": "pre_feed_forward_layer_norm",
    "pre_feedforward_layernorm_2.weight": "pre_feed_forward_layer_norm_2",
    "router.per_expert_scale": "router_per_expert_scale",
    "router.proj.weight": "router_projection",
    "router.scale": "router_norm_scale",
    "self_attn.k_norm.weight": "attention_k_norm",
    "self_attn.k_proj.weight": "attention_k_projection",
    "self_attn.o_proj.weight": "attention_o_projection",
    "self_attn.q_norm.weight": "attention_q_norm",
    "self_attn.q_proj.weight": "attention_q_projection",
    "self_attn.v_proj.weight": "attention_v_projection",
    "experts.down_proj": "routed_expert_down",
    "experts.gate_up_proj": "routed_expert_gate_up",
}

Q4_LAYER_ROLES = {
    "attn_k.weight": "attention_k_projection",
    "attn_k_norm.weight": "attention_k_norm",
    "attn_norm.weight": "input_layer_norm",
    "attn_output.weight": "attention_o_projection",
    "attn_q.weight": "attention_q_projection",
    "attn_q_norm.weight": "attention_q_norm",
    "attn_v.weight": "attention_v_projection",
    "ffn_down.weight": "shared_mlp_down",
    "ffn_down_exps.scale": "router_per_expert_scale",
    "ffn_down_exps.weight": "routed_expert_down",
    "ffn_gate.weight": "shared_mlp_gate",
    "ffn_gate_inp.scale": "router_norm_scale",
    "ffn_gate_inp.weight": "router_projection",
    "ffn_gate_up_exps.weight": "routed_expert_gate_up",
    "ffn_norm.weight": "pre_feed_forward_layer_norm",
    "ffn_up.weight": "shared_mlp_up",
    "layer_output_scale.weight": "layer_scalar",
    "post_attention_norm.weight": "post_attention_layer_norm",
    "post_ffw_norm.weight": "post_feed_forward_layer_norm",
    "post_ffw_norm_1.weight": "post_feed_forward_layer_norm_1",
    "post_ffw_norm_2.weight": "post_feed_forward_layer_norm_2",
    "pre_ffw_norm_2.weight": "pre_feed_forward_layer_norm_2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify checked outputs without writing"
    )
    return parser.parse_args()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, maximum_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f"input exceeds {maximum_bytes} bytes: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def schema_tuple(dtype: str, shape: Iterable[int]) -> tuple[str, tuple[int, ...], int]:
    shape_tuple = tuple(shape)
    element_bytes = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}[dtype]
    elements = 1
    for dimension in shape_tuple:
        elements *= dimension
    return dtype, shape_tuple, elements * element_bytes


def add_schema(
    result: dict[str, tuple[str, tuple[int, ...], int]],
    name: str,
    dtype: str,
    shape: Iterable[int],
) -> None:
    if name in result:
        raise ValueError(f"duplicate generated schema name: {name}")
    result[name] = schema_tuple(dtype, shape)


def add_source_vision(
    result: dict[str, tuple[str, tuple[int, ...], int]],
) -> None:
    add_schema(
        result, "model.embed_vision.embedding_projection.weight", "BF16", [2816, 1152]
    )
    add_schema(
        result,
        "model.vision_tower.patch_embedder.input_proj.weight",
        "BF16",
        [1152, 768],
    )
    add_schema(
        result,
        "model.vision_tower.patch_embedder.position_embedding_table",
        "BF16",
        [2, 10240, 1152],
    )
    add_schema(result, "model.vision_tower.std_bias", "BF16", [1152])
    add_schema(result, "model.vision_tower.std_scale", "BF16", [1152])
    for layer in range(VISION_LAYERS):
        prefix = f"model.vision_tower.encoder.layers.{layer}."
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "pre_feedforward_layernorm.weight",
        ):
            add_schema(result, prefix + suffix, "BF16", [1152])
        for suffix in ("self_attn.k_norm.weight", "self_attn.q_norm.weight"):
            add_schema(result, prefix + suffix, "BF16", [72])
        for projection in ("k", "o", "q", "v"):
            add_schema(
                result,
                prefix + f"self_attn.{projection}_proj.linear.weight",
                "BF16",
                [1152, 1152],
            )
        add_schema(
            result,
            prefix + "mlp.down_proj.linear.weight",
            "BF16",
            [1152, 4304],
        )
        for projection in ("gate", "up"):
            add_schema(
                result,
                prefix + f"mlp.{projection}_proj.linear.weight",
                "BF16",
                [4304, 1152],
            )


def source_schema() -> dict[str, tuple[str, tuple[int, ...], int]]:
    result: dict[str, tuple[str, tuple[int, ...], int]] = {}
    add_schema(
        result,
        "model.language_model.embed_tokens.weight",
        "BF16",
        [VOCABULARY, HIDDEN],
    )
    add_schema(result, "model.language_model.norm.weight", "BF16", [HIDDEN])
    for layer in range(LAYER_COUNT):
        prefix = f"model.language_model.layers.{layer}."
        global_attention = layer % 6 == 5
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "post_feedforward_layernorm_1.weight",
            "post_feedforward_layernorm_2.weight",
            "pre_feedforward_layernorm.weight",
            "pre_feedforward_layernorm_2.weight",
            "router.scale",
        ):
            add_schema(result, prefix + suffix, "BF16", [HIDDEN])
        add_schema(result, prefix + "layer_scalar", "BF16", [1])
        add_schema(result, prefix + "router.proj.weight", "BF16", [128, HIDDEN])
        add_schema(result, prefix + "router.per_expert_scale", "BF16", [128])
        head = 512 if global_attention else 256
        add_schema(result, prefix + "self_attn.q_norm.weight", "BF16", [head])
        add_schema(result, prefix + "self_attn.k_norm.weight", "BF16", [head])
        q = 8192 if global_attention else 4096
        kv = 1024 if global_attention else 2048
        add_schema(result, prefix + "self_attn.q_proj.weight", "BF16", [q, HIDDEN])
        add_schema(result, prefix + "self_attn.k_proj.weight", "BF16", [kv, HIDDEN])
        add_schema(result, prefix + "self_attn.o_proj.weight", "BF16", [HIDDEN, q])
        if not global_attention:
            add_schema(
                result, prefix + "self_attn.v_proj.weight", "BF16", [kv, HIDDEN]
            )
        add_schema(
            result, prefix + "mlp.gate_proj.weight", "BF16", [SHARED, HIDDEN]
        )
        add_schema(
            result, prefix + "mlp.up_proj.weight", "BF16", [SHARED, HIDDEN]
        )
        add_schema(
            result, prefix + "mlp.down_proj.weight", "BF16", [HIDDEN, SHARED]
        )
        add_schema(
            result,
            prefix + "experts.gate_up_proj",
            "BF16",
            [EXPERTS, 2 * ROUTED, HIDDEN],
        )
        add_schema(
            result,
            prefix + "experts.down_proj",
            "BF16",
            [EXPERTS, HIDDEN, ROUTED],
        )
    add_source_vision(result)
    return result


def add_external_nvfp4(
    result: dict[str, tuple[str, tuple[int, ...], int]],
    module: str,
    logical_shape: tuple[int, int],
) -> None:
    output, contracting = logical_shape
    add_schema(result, module + ".weight_packed", "U8", [output, contracting // 2])
    add_schema(
        result, module + ".weight_scale", "F8_E4M3", [output, contracting // 16]
    )
    add_schema(result, module + ".weight_global_scale", "F32", [1])
    add_schema(result, module + ".input_global_scale", "F32", [1])


def external_schema() -> dict[str, tuple[str, tuple[int, ...], int]]:
    source = source_schema()
    result = {
        name: contract
        for name, contract in source.items()
        if (
            name.startswith("model.vision")
            or name.startswith("model.embed_vision")
            or name in (
                "model.language_model.embed_tokens.weight",
                "model.language_model.norm.weight",
            )
            or (
                name.startswith("model.language_model.layers.")
                and not any(
                    token in name
                    for token in (".self_attn.", ".mlp.", ".experts.")
                )
            )
            or name.endswith("self_attn.q_norm.weight")
            or name.endswith("self_attn.k_norm.weight")
        )
    }
    for layer in range(LAYER_COUNT):
        prefix = f"model.language_model.layers.{layer}."
        global_attention = layer % 6 == 5
        q = 8192 if global_attention else 4096
        kv = 1024 if global_attention else 2048
        for projection, shape in (
            ("q", (q, HIDDEN)),
            ("k", (kv, HIDDEN)),
            ("o", (HIDDEN, q)),
        ):
            module = prefix + f"self_attn.{projection}_proj"
            add_schema(result, module + ".weight", "F8_E4M3", shape)
            add_schema(result, module + ".weight_scale", "BF16", [shape[0], 1])
        if not global_attention:
            module = prefix + "self_attn.v_proj"
            add_schema(result, module + ".weight", "F8_E4M3", [kv, HIDDEN])
            add_schema(result, module + ".weight_scale", "BF16", [kv, 1])
        add_schema(result, prefix + "self_attn.k_scale", "BF16", [1])
        add_schema(result, prefix + "self_attn.v_scale", "BF16", [1])
        add_external_nvfp4(result, prefix + "mlp.gate_proj", (SHARED, HIDDEN))
        add_external_nvfp4(result, prefix + "mlp.up_proj", (SHARED, HIDDEN))
        add_external_nvfp4(result, prefix + "mlp.down_proj", (HIDDEN, SHARED))
        for expert in range(EXPERTS):
            expert_prefix = prefix + f"experts.{expert}."
            add_external_nvfp4(
                result, expert_prefix + "gate_proj", (ROUTED, HIDDEN)
            )
            add_external_nvfp4(
                result, expert_prefix + "up_proj", (ROUTED, HIDDEN)
            )
            add_external_nvfp4(
                result, expert_prefix + "down_proj", (HIDDEN, ROUTED)
            )
    return result


def inventory_contract(inventory: dict[str, Any]) -> dict[str, tuple[str, tuple[int, ...], int]]:
    result: dict[str, tuple[str, tuple[int, ...], int]] = {}
    for tensor in inventory["tensors"]:
        name = str(tensor["name"])
        if name in result:
            raise ValueError(f"duplicate tensor in raw inventory: {name}")
        result[name] = (
            str(tensor["dtype"]),
            tuple(int(value) for value in tensor["shape"]),
            int(tensor["bytes"]),
        )
    return result


def require_exact_schema(
    label: str,
    actual: dict[str, tuple[str, tuple[int, ...], int]],
    expected: dict[str, tuple[str, tuple[int, ...], int]],
) -> None:
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        name for name in set(actual) & set(expected) if actual[name] != expected[name]
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"{label} schema mismatch: missing={missing[:3]} "
            f"unexpected={unexpected[:3]} mismatched={mismatched[:3]}"
        )


def classify_vision(name: str) -> str | None:
    if name == "model.embed_vision.embedding_projection.weight":
        return "vision_projection"
    if name == "model.vision_tower.patch_embedder.position_embedding_table":
        return "vision_embedding"
    if name == "model.vision_tower.patch_embedder.input_proj.weight":
        return "vision_projection"
    if name in ("model.vision_tower.std_bias", "model.vision_tower.std_scale"):
        return "vision_norm"
    if name.startswith("model.vision_tower.encoder.layers."):
        if ".self_attn." in name and "_norm." not in name:
            return "vision_attention"
        if ".mlp." in name:
            return "vision_mlp"
        return "vision_norm"
    return None


def classify_source(name: str) -> tuple[str, int | None, int | None]:
    vision = classify_vision(name)
    if vision is not None:
        return vision, None, None
    if name == "model.language_model.embed_tokens.weight":
        return "tied_embedding_and_output", None, None
    if name == "model.language_model.norm.weight":
        return "final_norm", None, None
    match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", name)
    if match is None:
        raise ValueError(f"unknown source tensor: {name}")
    layer = int(match.group(1))
    suffix = match.group(2)
    role = SOURCE_LAYER_ROLES.get(suffix)
    if role is None:
        raise ValueError(f"unknown source layer tensor: {name}")
    return role, layer, None


def external_module(name: str) -> tuple[str, str]:
    for suffix, component in (
        (".input_global_scale", "activation_global_scale"),
        (".weight_global_scale", "weight_global_scale"),
        (".weight_packed", "weight_packed"),
        (".weight_scale", "weight_scale"),
        (".weight", "weight"),
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)], component
    raise ValueError(f"unknown quantized tensor component: {name}")


def classify_external(name: str) -> tuple[str, int | None, int | None]:
    vision = classify_vision(name)
    if vision is not None:
        return vision, None, None
    if name in (
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    ):
        return classify_source(name)
    layer_match = re.fullmatch(
        r"model\.language_model\.layers\.(\d+)\.(.+)", name
    )
    if layer_match is None:
        raise ValueError(f"unknown external tensor: {name}")
    layer = int(layer_match.group(1))
    suffix = layer_match.group(2)
    if suffix in SOURCE_LAYER_ROLES:
        return SOURCE_LAYER_ROLES[suffix], layer, None
    if suffix == "self_attn.k_scale":
        return "attention_k_cache_scale", layer, None
    if suffix == "self_attn.v_scale":
        return "attention_v_cache_scale", layer, None
    module, _ = external_module(suffix)
    attention = re.fullmatch(r"self_attn\.([qkvo])_proj", module)
    if attention is not None:
        return f"attention_{attention.group(1)}_projection", layer, None
    shared = re.fullmatch(r"mlp\.(gate|up|down)_proj", module)
    if shared is not None:
        return f"shared_mlp_{shared.group(1)}", layer, None
    expert = re.fullmatch(r"experts\.(\d+)\.(gate|up|down)_proj", module)
    if expert is not None:
        expert_index = int(expert.group(1))
        return f"routed_expert_{expert.group(2)}", layer, expert_index
    raise ValueError(f"unknown external layer tensor: {name}")


def role_summary(
    tensors: list[dict[str, Any]], classifier: Any
) -> tuple[dict[str, dict[str, int]], dict[int, dict[str, Any]]]:
    roles: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensor_count": 0, "bytes": 0}
    )
    layers: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"tensor_count": 0, "bytes": 0, "expert_indices": set()}
    )
    for tensor in tensors:
        role, layer, expert = classifier(str(tensor["name"]))
        roles[role]["tensor_count"] += 1
        roles[role]["bytes"] += int(tensor["bytes"])
        if layer is not None:
            layers[layer]["tensor_count"] += 1
            layers[layer]["bytes"] += int(tensor["bytes"])
            if expert is not None:
                layers[layer]["expert_indices"].add(expert)
    normalized_layers = {}
    for layer, summary in sorted(layers.items()):
        experts = sorted(summary.pop("expert_indices"))
        normalized_layers[layer] = {
            **summary,
            "serialized_expert_count": len(experts),
            "expert_indices_complete": experts == list(range(EXPERTS)) if experts else None,
        }
    return dict(sorted(roles.items())), normalized_layers


def classify_q4(name: str) -> tuple[str, int | None]:
    if name == "output_norm.weight":
        return "final_norm", None
    if name == "rope_freqs.weight":
        return "rope_control", None
    if name == "token_embd.weight":
        return "tied_embedding_and_output", None
    match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if match is None or match.group(2) not in Q4_LAYER_ROLES:
        raise ValueError(f"unknown official Q4_0 tensor: {name}")
    return Q4_LAYER_ROLES[match.group(2)], int(match.group(1))


def q4_summary(inventory: dict[str, Any]) -> tuple[dict[str, Any], dict[int, Any]]:
    roles: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensor_count": 0, "bytes": 0}
    )
    layers: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"tensor_count": 0, "bytes": 0, "owns_v": False}
    )
    for tensor in inventory["tensors"]:
        role, layer = classify_q4(str(tensor["name"]))
        roles[role]["tensor_count"] += 1
        roles[role]["bytes"] += int(tensor["byte_count"])
        if layer is not None:
            layers[layer]["tensor_count"] += 1
            layers[layer]["bytes"] += int(tensor["byte_count"])
            if role == "attention_v_projection":
                layers[layer]["owns_v"] = True
    return dict(sorted(roles.items())), dict(sorted(layers.items()))


def aligned(value: int, alignment: int = ARENA_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def compiled_summary(
    source_tensors: list[dict[str, Any]], head_format: str
) -> dict[str, Any]:
    records: list[tuple[str, str, int]] = []

    def add(role: str, component: str, byte_count: int) -> None:
        records.append((role, component, byte_count))

    for tensor in source_tensors:
        name = str(tensor["name"])
        role, _, _ = classify_source(name)
        if role.startswith("vision_"):
            continue
        shape = [int(value) for value in tensor["shape"]]
        elements = int(tensor["bytes"]) // 2
        if role == "tied_embedding_and_output":
            if head_format == "q4_0":
                add(role, "q4_0_blocks", elements * 9 // 16)
            else:
                add(role, "weight_packed", elements // 2)
                add(role, "weight_local_scale", elements // 16)
                add(role, "weight_global_scale", 4)
                add(role, "activation_global_scale", 4)
        elif role.startswith("attention_") and role.endswith("_projection"):
            add(role, "weight", elements)
            add(role, "weight_channel_scale", shape[0] * 2)
        elif role.startswith("shared_mlp_") or role.startswith("routed_expert_"):
            add(role, "weight_packed", elements // 2)
            add(role, "weight_local_scale", elements // 16)
            add(role, "weight_global_scale", 4)
            add(role, "activation_global_scale", 4)
        else:
            add(role, "source_bf16", int(tensor["bytes"]))
    for _layer in range(LAYER_COUNT):
        add("attention_k_cache_scale", "compiler_bf16", 2)
        add("attention_v_cache_scale", "compiler_bf16", 2)
    roles: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensor_count": 0, "payload_bytes": 0, "aligned_bytes": 0}
    )
    for role, _component, byte_count in records:
        roles[role]["tensor_count"] += 1
        roles[role]["payload_bytes"] += byte_count
        roles[role]["aligned_bytes"] += aligned(byte_count)
    for totals in roles.values():
        totals["alignment_padding_bytes"] = (
            totals["aligned_bytes"] - totals["payload_bytes"]
        )
    payload_bytes = sum(record[2] for record in records)
    aligned_weight_arena_bytes = sum(aligned(record[2]) for record in records)
    return {
        "head_format": head_format,
        "tensor_count": len(records),
        "payload_bytes": payload_bytes,
        "alignment_bytes": ARENA_ALIGNMENT,
        "alignment_padding_bytes": aligned_weight_arena_bytes - payload_bytes,
        "aligned_weight_arena_bytes": aligned_weight_arena_bytes,
        "roles": dict(sorted(roles.items())),
        "producer_contracts": {
            "attention": {
                "producer": "gem16",
                "format": "FP8 E4M3 per-output-channel weight / dynamic per-token activation",
                "local_scale_dtype": "BF16",
                "global_scale_role": "none",
                "activation_scale_role": "dynamic_per_token_dequant_multiplier",
                "final_gpu_layout": "source_nk_fp8",
            },
            "nvfp4": {
                "producer": "gem16",
                "format": "NVFP4 E2M1 packed / E4M3 group-16 scales",
                "local_scale_dtype": "F8_E4M3",
                "local_scale_vector_size": 16,
                "global_scale_role": "divisor",
                "activation_scale_role": "divisor",
                "shared_final_gpu_layout": "sm120_row8_k64",
                "expert_final_gpu_layout": "expert_major_sm120_row8_k64",
            },
            "head": {
                "producer": "gem16",
                "format": head_format,
                "tied_physical_allocation": True,
                "duplicate_lm_head_allowed": False,
            },
        },
    }


def source_manifest(
    label: str,
    inventory: dict[str, Any],
    roles: dict[str, Any],
    layers: dict[int, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "m03_exact_inventory",
        "source_family": label,
        "source_revision": inventory["source"]["revision"],
        "source_lock_sha256": sha256_file(LOCKS[label]),
        "raw_inventory": {
            "path": RAW[label].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(RAW[label]),
        },
        "tensor_count": len(inventory["tensors"]),
        "tensor_payload_bytes": sum(int(tensor["bytes"]) for tensor in inventory["tensors"]),
        "text_tensor_count": len(inventory["tensors"]) - 356,
        "text_tensor_bytes": sum(int(tensor["bytes"]) for tensor in inventory["tensors"])
        - VISION_BYTES,
        "compile_excluded_vision_tensor_count": 356,
        "compile_excluded_vision_bytes": VISION_BYTES,
        "mtp_tensor_count": 0,
        "unknown_tensor_count": 0,
        "roles": roles,
        "layer_count": len(layers),
        "fused_expert_contract": {
            "gate_up_name_template": "model.language_model.layers.{layer}.experts.gate_up_proj",
            "gate_up_shape": [128, 1408, 2816],
            "gate_up_axis_order": "expert,gate_then_up,input",
            "down_name_template": "model.language_model.layers.{layer}.experts.down_proj",
            "down_shape": [128, 2816, 704],
            "down_axis_order": "expert,output,input",
            "expert_axis": 0,
        },
    }


def build_outputs() -> dict[Path, bytes]:
    inventories = {name: load_json(path) for name, path in RAW.items()}
    reference_lock = load_json(REFERENCE_LOCK)
    transformers_references = [
        reference
        for reference in reference_lock["references"]
        if reference.get("name") == "transformers"
    ]
    if len(transformers_references) != 1:
        raise ValueError("reference lock must contain exactly one transformers source")
    transformers_reference = transformers_references[0]
    if (
        transformers_reference.get("revision")
        != "a08ace4bbd97e721c98751deec37d87b026acadc"
    ):
        raise ValueError("locked transformers semantic reference changed")
    expected_source = source_schema()
    expected_external = external_schema()
    require_exact_schema(
        "QAT BF16", inventory_contract(inventories["qat_bf16"]), expected_source
    )
    require_exact_schema(
        "ordinary BF16",
        inventory_contract(inventories["ordinary_bf16"]),
        expected_source,
    )
    require_exact_schema(
        "Unsloth NVFP4",
        inventory_contract(inventories["unsloth_nvfp4"]),
        expected_external,
    )
    if len(expected_source) != 1013 or len(expected_external) != 47478:
        raise ValueError("generated source tensor counts do not match M01 locks")

    qat_roles, qat_layers = role_summary(
        inventories["qat_bf16"]["tensors"], classify_source
    )
    ordinary_roles, ordinary_layers = role_summary(
        inventories["ordinary_bf16"]["tensors"], classify_source
    )
    unsloth_roles, unsloth_layers = role_summary(
        inventories["unsloth_nvfp4"]["tensors"], classify_external
    )
    if qat_roles != ordinary_roles or qat_layers != ordinary_layers:
        raise ValueError("QAT and ordinary compact role inventories differ")
    if sum(item["bytes"] for item in qat_roles.values()) != SOURCE_PAYLOAD_BYTES:
        raise ValueError("source role bytes do not reconcile")
    if sum(item["bytes"] for item in unsloth_roles.values()) != UNSLOTH_PAYLOAD_BYTES:
        raise ValueError("Unsloth role bytes do not reconcile")
    if any(
        summary["serialized_expert_count"] != 128
        or not summary["expert_indices_complete"]
        for summary in unsloth_layers.values()
    ):
        raise ValueError("Unsloth expert indices are incomplete")

    q4_roles, q4_layers = q4_summary(inventories["official_q4_0"])
    if len(inventories["official_q4_0"]["tensors"]) != 658:
        raise ValueError("official Q4_0 tensor count changed")
    if inventories["official_q4_0"]["gguf"]["size_bytes"] != Q4_FILE_BYTES:
        raise ValueError("official Q4_0 file bytes changed")

    compiled_q4 = compiled_summary(inventories["qat_bf16"]["tensors"], "q4_0")
    compiled_nvfp4 = compiled_summary(
        inventories["qat_bf16"]["tensors"], "nvfp4"
    )
    if compiled_q4["tensor_count"] != 1282:
        raise ValueError("compiled Q4_0-head tensor count changed")
    if compiled_nvfp4["tensor_count"] != 1285:
        raise ValueError("compiled NVFP4-head tensor count changed")
    if compiled_q4["aligned_weight_arena_bytes"] != 14_696_667_648:
        raise ValueError("compiled Q4_0-head aligned bytes changed")
    if compiled_nvfp4["aligned_weight_arena_bytes"] != 14_696_668_160:
        raise ValueError("compiled NVFP4-head aligned bytes changed")

    qat_lookup = {
        tensor["name"]: tensor for tensor in inventories["qat_bf16"]["tensors"]
    }
    gate_up = qat_lookup["model.language_model.layers.0.experts.gate_up_proj"]
    down = qat_lookup["model.language_model.layers.0.experts.down_proj"]
    gate_up_slice = int(gate_up["bytes"]) // EXPERTS
    down_slice = int(down["bytes"]) // EXPERTS
    expert_axis_proof = {
        "source_reference": {
            "repository": transformers_reference["repository"],
            "revision": transformers_reference["revision"],
            "path": "src/transformers/models/gemma4/modeling_gemma4.py",
            "class": "Gemma4TextExperts",
            "parameter_contract": "gate_up_proj[num_experts,2*intermediate,hidden]",
            "forward_contract": "linear(..., gate_up_proj[expert_idx]).chunk(2, dim=-1) -> gate, up",
            "reference_lock": REFERENCE_LOCK.relative_to(ROOT).as_posix(),
            "reference_lock_sha256": sha256_file(REFERENCE_LOCK),
        },
        "numerical_reference": "benchmarks/goldens/gemma4_26b/qat-bf16-selected/qat-bf16-selected.json",
        "gate_up": {
            "name": gate_up["name"],
            "shape": gate_up["shape"],
            "expert_axis": 0,
            "axis_order": "expert,gate_then_up,input",
            "expert_slice_bytes": gate_up_slice,
            "shard": gate_up["shard"],
            "expert_0_absolute_offset": gate_up["absolute_offset"],
            "expert_127_absolute_offset": gate_up["absolute_offset"]
            + 127 * gate_up_slice,
        },
        "down": {
            "name": down["name"],
            "shape": down["shape"],
            "expert_axis": 0,
            "axis_order": "expert,output,input",
            "expert_slice_bytes": down_slice,
            "shard": down["shard"],
            "expert_0_absolute_offset": down["absolute_offset"],
            "expert_127_absolute_offset": down["absolute_offset"]
            + 127 * down_slice,
        },
    }

    layer_table = []
    for layer in range(LAYER_COUNT):
        global_attention = layer % 6 == 5
        source_layer = qat_layers[layer]
        external_layer = unsloth_layers[layer]
        q4_layer = q4_layers[layer]
        expected_v = not global_attention
        if q4_layer["owns_v"] != expected_v:
            raise ValueError(f"official Q4_0 V ownership mismatch at layer {layer}")
        layer_table.append(
            {
                "layer": layer,
                "attention_type": "full_attention"
                if global_attention
                else "sliding_attention",
                "owns_v_projection": expected_v,
                "kv_producer": "k projection reused as raw V input"
                if global_attention
                else "separate k_proj/v_proj",
                "source_bf16": source_layer,
                "external_unsloth_nvfp4": external_layer,
                "official_q4_0": q4_layer,
            }
        )

    cross_map = {
        "schema_version": 1,
        "status": "m03_exact_cross_map",
        "raw_names_are_not_interchangeable": True,
        "roles": [
            {
                "role": role,
                "source_bf16": source_name,
                "external_unsloth": external_name,
                "official_q4_0": q4_name,
                "compiled_hybrid": compiled_name,
            }
            for role, source_name, external_name, q4_name, compiled_name in (
                (
                    "tied_embedding_and_output",
                    "model.language_model.embed_tokens.weight",
                    "model.language_model.embed_tokens.weight",
                    "token_embd.weight",
                    "model.language_model.embed_tokens.weight_q4_0 or explicit NVFP4 family",
                ),
                (
                    "attention_q_projection",
                    "...self_attn.q_proj.weight",
                    "...self_attn.q_proj.{weight,weight_scale}",
                    "blk.N.attn_q.weight",
                    "...self_attn.q_proj.{weight,weight_scale}",
                ),
                (
                    "shared_mlp_gate",
                    "...mlp.gate_proj.weight",
                    "...mlp.gate_proj.{weight_packed,weight_scale,weight_global_scale,input_global_scale}",
                    "blk.N.ffn_gate.weight",
                    "...mlp.gate_proj.{weight_packed,weight_scale,weight_global_scale,input_global_scale}",
                ),
                (
                    "router_projection",
                    "...router.proj.weight",
                    "...router.proj.weight",
                    "blk.N.ffn_gate_inp.weight",
                    "...router.proj.weight",
                ),
                (
                    "routed_expert_gate_up",
                    "...experts.gate_up_proj [128,1408,2816]",
                    "128 separate ...experts.E.{gate,up}_proj quantized families",
                    "blk.N.ffn_gate_up_exps.weight",
                    "...experts.gate_up_proj quantized fused family",
                ),
                (
                    "routed_expert_down",
                    "...experts.down_proj [128,2816,704]",
                    "128 separate ...experts.E.down_proj quantized families",
                    "blk.N.ffn_down_exps.weight",
                    "...experts.down_proj quantized fused family",
                ),
            )
        ],
        "quantization_semantics": {
            "external_unsloth_nvfp4": {
                "producer": "llm-compressor/compressed-tensors@0.17.2.a20260707",
                "local_scale_dtype": "F8_E4M3",
                "local_scale_vector_size": 16,
                "weight_global_scale_role": "divisor",
                "activation_global_scale_role": "divisor",
                "final_gpu_layout": "external source order; not a gem16 compiled artifact",
            },
            "compiled_hybrid": compiled_q4["producer_contracts"],
        },
    }

    compiled_document = {
        "schema_version": 1,
        "status": "m03_frozen_compiled_hybrid_role_contract",
        "source_family": "qat_bf16",
        "source_lock_sha256": sha256_file(LOCKS["qat_bf16"]),
        "profiles": {"q4_0_head": compiled_q4, "nvfp4_head": compiled_nvfp4},
        "selected_conservative_weight_arena_bytes": max(
            compiled_q4["aligned_weight_arena_bytes"],
            compiled_nvfp4["aligned_weight_arena_bytes"],
        ),
        "weight_target_bytes": 14_100 * MIB,
        "hard_stop_bytes": 14_300 * MIB,
        "fp8_kv_32k_bytes": 420 * MIB,
        "synthetic_fixed_regions": {
            "moe_prefill_workspace": 256 * MIB,
            "activation_output_workspace": 128 * MIB,
            "graph_private_reserve": 32 * MIB,
            "allocator_metadata_guard": 32 * MIB,
        },
        "required_direct_free_margin_bytes": 700 * MIB,
        "omitted_families": {
            "vision_tensor_count": 356,
            "vision_source_bytes": VISION_BYTES,
            "mtp_tensor_count": 0,
            "audio_tensor_count": 0,
            "video_tensor_count": 0,
        },
        "strictness": {
            "unknown_tensor_allowed": False,
            "duplicate_lm_head_allowed": False,
            "source_and_compiled_validators_are_separate": True,
            "external_unsloth_is_project_compiled": False,
        },
    }

    fixture = {
        "schema_version": 1,
        "contract": "gemma4_26b_m03_exact_inventory_v1",
        "dimensions": {
            "layers": LAYER_COUNT,
            "hidden": HIDDEN,
            "shared_intermediate": SHARED,
            "routed_intermediate": ROUTED,
            "experts": EXPERTS,
            "top_k": 8,
            "vocabulary": VOCABULARY,
            "local_layers": 25,
            "global_layers": 5,
        },
        "inputs": {
            name: {
                "lock_sha256": sha256_file(LOCKS[name]),
                "raw_inventory_sha256": sha256_file(RAW[name]),
            }
            for name in RAW
        },
        "semantic_references": {
            "transformers": expert_axis_proof["source_reference"]
        },
        "source_bf16": {
            "tensor_count": 1013,
            "payload_bytes": SOURCE_PAYLOAD_BYTES,
            "text_tensor_count": 657,
            "text_bytes": SOURCE_PAYLOAD_BYTES - VISION_BYTES,
            "compile_excluded_vision_tensor_count": 356,
            "compile_excluded_vision_bytes": VISION_BYTES,
            "mtp_tensor_count": 0,
            "unknown_tensor_count": 0,
        },
        "external_unsloth_nvfp4": {
            "tensor_count": 47478,
            "payload_bytes": UNSLOTH_PAYLOAD_BYTES,
            "text_tensor_count": 47122,
            "text_bytes": UNSLOTH_PAYLOAD_BYTES - VISION_BYTES,
            "compile_excluded_vision_tensor_count": 356,
            "compile_excluded_vision_bytes": VISION_BYTES,
            "serialized_experts_per_layer": 128,
            "is_project_compiled_artifact": False,
            "unknown_tensor_count": 0,
        },
        "official_q4_0": {
            "tensor_count": 658,
            "file_bytes": Q4_FILE_BYTES,
            "separate_mmproj_locked_but_text_inventory_excludes_it": True,
        },
        "expert_axis_proof": expert_axis_proof,
        "compiled_hybrid": compiled_document,
    }

    qat_document = source_manifest(
        "qat_bf16", inventories["qat_bf16"], qat_roles, qat_layers
    )
    qat_document["expert_axis_proof"] = expert_axis_proof
    ordinary_document = source_manifest(
        "ordinary_bf16",
        inventories["ordinary_bf16"],
        ordinary_roles,
        ordinary_layers,
    )
    unsloth_document = {
        "schema_version": 1,
        "status": "m03_exact_inventory",
        "source_family": "unsloth_nvfp4",
        "source_revision": inventories["unsloth_nvfp4"]["source"]["revision"],
        "source_lock_sha256": sha256_file(LOCKS["unsloth_nvfp4"]),
        "raw_inventory": {
            "path": RAW["unsloth_nvfp4"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(RAW["unsloth_nvfp4"]),
        },
        "tensor_count": 47478,
        "tensor_payload_bytes": UNSLOTH_PAYLOAD_BYTES,
        "text_tensor_count": 47122,
        "text_tensor_bytes": UNSLOTH_PAYLOAD_BYTES - VISION_BYTES,
        "compile_excluded_vision_tensor_count": 356,
        "compile_excluded_vision_bytes": VISION_BYTES,
        "mtp_tensor_count": 0,
        "unknown_tensor_count": 0,
        "project_compiled_artifact": False,
        "roles": unsloth_roles,
        "layer_count": len(unsloth_layers),
        "expert_serialization": {
            "kind": "individual_expert_projection_families",
            "experts_per_layer": 128,
            "projections_per_expert": ["gate", "up", "down"],
            "components_per_projection": [
                "weight_packed",
                "weight_scale",
                "weight_global_scale",
                "input_global_scale",
            ],
        },
        "quantization_semantics": cross_map["quantization_semantics"][
            "external_unsloth_nvfp4"
        ],
    }
    q4_document = {
        "schema_version": 1,
        "status": "m03_exact_inventory",
        "source_family": "official_q4_0",
        "source_revision": inventories["official_q4_0"]["source_revision"],
        "source_lock_sha256": sha256_file(LOCKS["official_q4_0"]),
        "raw_inventory": {
            "path": RAW["official_q4_0"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(RAW["official_q4_0"]),
        },
        "tensor_count": 658,
        "file_bytes": Q4_FILE_BYTES,
        "roles": q4_roles,
        "unknown_tensor_count": 0,
        "text_only": True,
        "separate_mmproj_in_source_lock": True,
    }
    layer_document = {
        "schema_version": 1,
        "status": "m03_exact_layer_table",
        "layer_count": LAYER_COUNT,
        "layers": layer_table,
    }

    outputs = {
        FIXTURE: json_bytes(fixture),
        OUTPUT_ROOT / "qat-bf16.json": json_bytes(qat_document),
        OUTPUT_ROOT / "ordinary-bf16.json": json_bytes(ordinary_document),
        OUTPUT_ROOT / "unsloth-nvfp4.json": json_bytes(unsloth_document),
        OUTPUT_ROOT / "google-q4_0.json": json_bytes(q4_document),
        OUTPUT_ROOT / "cross-map.json": json_bytes(cross_map),
        OUTPUT_ROOT / "layer-table.json": json_bytes(layer_document),
        OUTPUT_ROOT / "compiled-hybrid.json": json_bytes(compiled_document),
    }
    return outputs


def main() -> int:
    args = parse_args()
    try:
        outputs = build_outputs()
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    stale = [
        path
        for path, data in outputs.items()
        if not path.is_file() or path.read_bytes() != data
    ]
    if args.check:
        if stale:
            print(
                "stale Gemma 4 26B M03 manifests: "
                + ", ".join(path.relative_to(ROOT).as_posix() for path in stale),
                file=sys.stderr,
            )
            return 1
    else:
        for path, data in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    print(
        f"Gemma 4 26B M03 manifests {'verified' if args.check else 'generated'}: "
        f"{len(outputs)} files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
