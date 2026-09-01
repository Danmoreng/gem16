#!/usr/bin/env python3
"""Compare pinned Gemma 4 Vision padded and unpadded valid-row execution.

This diagnostic loads only the Vision tower and Vision-to-text projector from
the immutable local BF16 checkpoint. It never executes repository model code
and performs no network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
    Gemma4VisionRotaryEmbedding,
)


ORACLE_TRANSFORMERS_VERSION = "5.14.1"
ORACLE_TRANSFORMERS_COMMIT = "a08ace4bbd97e721c98751deec37d87b026acadc"
SOURCE_REVISION = "f1e06dc520982d9b9edd76859fdb7ab209449949"

# Declared before executing the comparison. These are deliberately much
# tighter than the FP8-vs-BF16 model-quality boundary: padded and unpadded runs
# use the same BF16 weights and operators and differ only by masked rows.
MAX_ABSOLUTE_ERROR = 0.0625
MAX_RELATIVE_L2 = 5.0e-4
MIN_COSINE = 0.999999


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_modules(source: Path, device: torch.device):
    config = Gemma4Config.from_pretrained(source, local_files_only=True)
    with torch.device("meta"):
        vision = Gemma4VisionModel(config.vision_config)
        projector = Gemma4MultimodalEmbedder(
            config.vision_config, config.text_config
        )

    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    vision_names = sorted(
        name for name in weight_map if name.startswith("model.vision_tower.")
    )
    projector_name = "model.embed_vision.embedding_projection.weight"
    if not vision_names or projector_name not in weight_map:
        raise RuntimeError("pinned checkpoint does not contain the Vision modules")

    by_shard: dict[str, list[str]] = {}
    for name in [*vision_names, projector_name]:
        by_shard.setdefault(weight_map[name], []).append(name)
    tensors: dict[str, torch.Tensor] = {}
    for shard_name, names in by_shard.items():
        with safe_open(source / shard_name, framework="pt", device="cpu") as shard:
            for name in names:
                tensors[name] = shard.get_tensor(name)

    vision_state = {
        name.removeprefix("model.vision_tower."): tensor
        for name, tensor in tensors.items()
        if name.startswith("model.vision_tower.")
    }
    missing, unexpected = vision.load_state_dict(
        vision_state, strict=False, assign=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Vision state mismatch: missing={missing}, unexpected={unexpected}"
        )
    missing, unexpected = projector.load_state_dict(
        {"embedding_projection.weight": tensors[projector_name]},
        strict=False,
        assign=True,
    )
    if missing or unexpected:
        raise RuntimeError(
            f"projector state mismatch: missing={missing}, unexpected={unexpected}"
        )
    # The RoPE frequency buffers are intentionally non-persistent, so a meta
    # construction does not receive them from the checkpoint. Recreate only
    # that immutable oracle-derived buffer before moving the loaded modules.
    vision.encoder.rotary_emb = Gemma4VisionRotaryEmbedding(
        config.vision_config, device=device
    )
    return vision.eval().to(device), projector.eval().to(device), config


def deterministic_patches(raw_tokens: int) -> torch.Tensor:
    elements = raw_tokens * 768
    values = torch.arange(elements, dtype=torch.int64)
    values = ((values * 37 + 11) % 251).to(torch.float32) / 250.0
    return values.reshape(1, raw_tokens, 768)


def positions(width: int, height: int) -> torch.Tensor:
    patch = torch.arange(width * height, dtype=torch.int64)
    return torch.stack((patch % width, patch // width), dim=-1).unsqueeze(0)


def tensor_hash(value: torch.Tensor) -> str:
    contiguous = value.detach().to(torch.bfloat16).cpu().contiguous()
    return hashlib.sha256(contiguous.view(torch.uint16).numpy().tobytes()).hexdigest()


def metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left = left.detach().float().cpu().reshape(-1)
    right = right.detach().float().cpu().reshape(-1)
    difference = left - right
    left_energy = torch.dot(left, left).item()
    right_energy = torch.dot(right, right).item()
    difference_energy = torch.dot(difference, difference).item()
    denominator = math.sqrt(max(left_energy * right_energy, 0.0))
    result = {
        "elements": left.numel(),
        "max_absolute_error": difference.abs().max().item(),
        "relative_l2": math.sqrt(difference_energy / max(left_energy, 1.0e-30)),
        "cosine": torch.dot(left, right).item() / max(denominator, 1.0e-30),
        "unpadded_bf16_sha256": tensor_hash(left),
        "padded_bf16_sha256": tensor_hash(right),
    }
    result["passed"] = (
        result["max_absolute_error"] <= MAX_ABSOLUTE_ERROR
        and result["relative_l2"] <= MAX_RELATIVE_L2
        and result["cosine"] >= MIN_COSINE
    )
    return result


def run_once(
    vision: Gemma4VisionModel,
    projector: Gemma4MultimodalEmbedder,
    pixel_values: torch.Tensor,
    pixel_positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    captures: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captures[name] = value.detach().clone()
        return hook

    handles = [vision.patch_embedder.register_forward_hook(capture("patch"))]
    for layer in (0, 13, 26):
        handles.append(vision.encoder.layers[layer].register_forward_hook(
            capture(f"layer_{layer}")
        ))
    handles.append(vision.pooler.register_forward_hook(capture("pool")))
    try:
        with torch.inference_mode():
            output = vision(
                pixel_values=pixel_values,
                pixel_position_ids=pixel_positions,
            ).last_hidden_state
            captures["final_1152"] = output.detach().clone()
            captures["final_2816"] = projector(output).detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    return captures


def compare_case(
    vision: Gemma4VisionModel,
    projector: Gemma4MultimodalEmbedder,
    device: torch.device,
    budget: int,
    label: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    raw_tokens = width * height
    capacity = budget * 9
    if width % 3 or height % 3 or raw_tokens > capacity:
        raise ValueError(f"invalid case geometry {label}: {width}x{height}/{budget}")
    real_values = deterministic_patches(raw_tokens)
    real_positions = positions(width, height)
    padded_values = torch.zeros((1, capacity, 768), dtype=torch.float32)
    padded_values[:, :raw_tokens] = real_values
    padded_positions = torch.full((1, capacity, 2), -1, dtype=torch.int64)
    padded_positions[:, :raw_tokens] = real_positions

    unpadded = run_once(
        vision, projector, real_values.to(device), real_positions.to(device)
    )
    padded = run_once(
        vision, projector, padded_values.to(device), padded_positions.to(device)
    )
    soft_tokens = raw_tokens // 9
    comparisons: dict[str, Any] = {}
    for boundary in ("patch", "layer_0", "layer_13", "layer_26"):
        comparisons[boundary] = metrics(
            unpadded[boundary][:, :raw_tokens],
            padded[boundary][:, :raw_tokens],
        )
    comparisons["pool"] = metrics(
        unpadded["pool"][:, :soft_tokens],
        padded["pool"][:, :soft_tokens],
    )
    comparisons["final_1152"] = metrics(
        unpadded["final_1152"], padded["final_1152"]
    )
    comparisons["final_2816"] = metrics(
        unpadded["final_2816"], padded["final_2816"]
    )
    return {
        "budget": budget,
        "label": label,
        "grid": [width, height],
        "raw_tokens": raw_tokens,
        "soft_tokens": soft_tokens,
        "padded_raw_tokens": capacity,
        "boundaries": comparisons,
        "passed": all(item["passed"] for item in comparisons.values()),
    }


def main() -> int:
    args = arguments()
    source = args.source.expanduser().resolve()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("the padding oracle requires a CUDA device")
    if not source.is_dir():
        raise SystemExit(f"source checkpoint does not exist: {source}")

    import transformers
    if transformers.__version__ != ORACLE_TRANSFORMERS_VERSION:
        raise SystemExit(
            f"Transformers {ORACLE_TRANSFORMERS_VERSION} required, got "
            f"{transformers.__version__}"
        )
    vision, projector, _config = load_modules(source, device)
    cases = (
        (70, "square", 24, 24),
        (70, "wide", 30, 18),
        (70, "tall", 18, 30),
        (70, "minimum_narrow", 3, 210),
        (140, "square", 33, 33),
        (140, "wide", 60, 21),
        (140, "tall", 21, 60),
        (140, "minimum_narrow", 3, 420),
        (280, "square", 48, 48),
        (280, "wide", 105, 24),
        (280, "tall", 24, 105),
        (280, "minimum_narrow", 3, 840),
    )
    results = [
        compare_case(vision, projector, device, *case) for case in cases
    ]
    equivalent = all(item["passed"] for item in results)
    report = {
        "schema_version": 1,
        "purpose": "Gemma 4 26B Vision padded-versus-unpadded equivalence",
        "source_revision": SOURCE_REVISION,
        "transformers_version": ORACLE_TRANSFORMERS_VERSION,
        "transformers_commit": ORACLE_TRANSFORMERS_COMMIT,
        "boundaries": {
            "max_absolute_error": MAX_ABSOLUTE_ERROR,
            "max_relative_l2": MAX_RELATIVE_L2,
            "min_cosine": MIN_COSINE,
        },
        "cases": results,
        "equivalent": equivalent,
        "decision": "keep_unpadded" if equivalent else "padding_required",
        "diagnostic_completed": True,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
