#!/usr/bin/env python3
"""Compare two gem16 layer-state dumps.

This is correctness tooling only. The runtime and chat CLI do not depend on
Python.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import sys


MAGIC = b"G16ST001"
HEADER = struct.Struct("<IIQQQII")
LAYER_HEADER = struct.Struct("<IIQ")
STAGE_NAMES = (
    "attention_output",
    "post_attention_norm",
    "post_attention_residual",
    "pre_feedforward_norm",
    "gate",
    "up",
    "gelu_product",
    "mlp_output",
    "post_feedforward_norm",
    "hidden",
)
LEGACY_STAGE_NAMES = tuple(
    name for name in STAGE_NAMES if name not in ("gate", "up", "gelu_product")
)
INTERMEDIATE_ELEMENTS = 15360


@dataclass(frozen=True)
class LayerState:
    index: int
    global_attention: bool
    attention_context: array | None
    stages: dict[str, array]
    key: array
    value: array

    @property
    def hidden(self) -> array:
        return self.stages["hidden"]


@dataclass(frozen=True)
class StateDump:
    version: int
    position: int
    projection_path: str
    kv_cache_mode: str
    layers: tuple[LayerState, ...]


def read_floats(data: bytes, offset: int, elements: int) -> tuple[array, int]:
    end = offset + elements * 4
    if end > len(data):
        raise ValueError("truncated float payload")
    values = array("f")
    values.frombytes(data[offset:end])
    if sys.byteorder != "little":
        values.byteswap()
    return values, end


def load_state(path: Path) -> StateDump:
    data = path.read_bytes()
    if len(data) < len(MAGIC) + HEADER.size or data[: len(MAGIC)] != MAGIC:
        raise ValueError(f"{path}: not a gem16 layer-state dump")
    offset = len(MAGIC)
    (
        version,
        layer_count,
        position,
        hidden_elements,
        total_elements,
        path_id,
        mode_or_reserved,
    ) = HEADER.unpack_from(data, offset)
    offset += HEADER.size
    if version not in (1, 2, 3, 4, 5):
        raise ValueError(f"{path}: unsupported state-dump header")
    if version < 4 and mode_or_reserved != 0:
        raise ValueError(f"{path}: unsupported state-dump header")
    if version >= 4 and mode_or_reserved not in (0, 1, 2):
        raise ValueError(f"{path}: unknown KV-cache mode {mode_or_reserved}")
    if hidden_elements == 0 or layer_count == 0:
        raise ValueError(f"{path}: invalid state geometry")
    if path_id not in (0, 1, 2):
        raise ValueError(f"{path}: unknown projection path {path_id}")

    layers: list[LayerState] = []
    counted_elements = 0
    for expected_index in range(layer_count):
        if offset + LAYER_HEADER.size > len(data):
            raise ValueError(f"{path}: truncated layer header")
        index, flags, kv_elements = LAYER_HEADER.unpack_from(data, offset)
        offset += LAYER_HEADER.size
        if index != expected_index or flags & ~1:
            raise ValueError(f"{path}: invalid layer record {expected_index}")
        attention_context = None
        if version >= 3:
            attention_elements = 16 * (512 if flags & 1 else 256)
            attention_context, offset = read_floats(
                data, offset, attention_elements
            )
            counted_elements += attention_elements
        stages: dict[str, array] = {}
        stage_names = (
            STAGE_NAMES
            if version >= 5
            else LEGACY_STAGE_NAMES
            if version >= 2
            else ("hidden",)
        )
        for stage_name in stage_names:
            stage_elements = (
                INTERMEDIATE_ELEMENTS
                if stage_name in ("gate", "up", "gelu_product")
                else hidden_elements
            )
            stages[stage_name], offset = read_floats(
                data, offset, stage_elements
            )
        key, offset = read_floats(data, offset, kv_elements)
        value, offset = read_floats(data, offset, kv_elements)
        counted_elements += (
            sum(
                INTERMEDIATE_ELEMENTS
                if name in ("gate", "up", "gelu_product")
                else hidden_elements
                for name in stage_names
            )
            + 2 * kv_elements
        )
        layers.append(
            LayerState(
                index,
                bool(flags & 1),
                attention_context,
                stages,
                key,
                value,
            )
        )
    if offset != len(data) or counted_elements != total_elements:
        raise ValueError(f"{path}: trailing bytes or inconsistent element count")
    return StateDump(
        version,
        position,
        ("native_sm120", "cuda_reference", "vllm")[path_id],
        (
            ("checkpoint_fp8", "bf16", "unspecified")[mode_or_reserved]
            if version >= 4
            else "unspecified"
        ),
        tuple(layers),
    )


@dataclass(frozen=True)
class Metrics:
    rms: float
    maximum: float
    cosine: float
    exact: bool


def metrics(left: array, right: array) -> Metrics:
    if len(left) != len(right):
        raise ValueError("tensor sizes differ")
    squared_error = 0.0
    maximum = 0.0
    left_norm = 0.0
    right_norm = 0.0
    dot = 0.0
    exact = True
    for left_value, right_value in zip(left, right, strict=True):
        difference = float(left_value) - float(right_value)
        squared_error += difference * difference
        maximum = max(maximum, abs(difference))
        left_norm += float(left_value) * float(left_value)
        right_norm += float(right_value) * float(right_value)
        dot += float(left_value) * float(right_value)
        exact = exact and left_value == right_value
    denominator = math.sqrt(left_norm * right_norm)
    cosine = dot / denominator if denominator != 0.0 else float("nan")
    return Metrics(
        math.sqrt(squared_error / len(left)) if left else 0.0,
        maximum,
        cosine,
        exact,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--details-layer", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        left = load_state(args.left)
        right = load_state(args.right)
        if left.position != right.position:
            raise ValueError(
                f"capture positions differ: {left.position} vs {right.position}"
            )
        if len(left.layers) != len(right.layers):
            raise ValueError("layer counts differ")

        print(
            f"position={left.position} "
            f"left={left.projection_path}/{left.kv_cache_mode}/v{left.version} "
            f"right={right.projection_path}/{right.kv_cache_mode}/v{right.version}"
        )
        print(
            "layer  hidden_rms  hidden_max  hidden_cos       "
            "k_rms       v_rms  exact"
        )
        first_difference: int | None = None
        for left_layer, right_layer in zip(
            left.layers, right.layers, strict=True
        ):
            if (
                left_layer.index != right_layer.index
                or left_layer.global_attention != right_layer.global_attention
            ):
                raise ValueError("layer metadata differs")
            hidden = metrics(left_layer.hidden, right_layer.hidden)
            key = metrics(left_layer.key, right_layer.key)
            value = metrics(left_layer.value, right_layer.value)
            exact = hidden.exact and key.exact and value.exact
            if not exact and first_difference is None:
                first_difference = left_layer.index
            print(
                f"{left_layer.index:5d}  {hidden.rms:10.3e}  "
                f"{hidden.maximum:10.3e}  {hidden.cosine:10.7f}  "
                f"{key.rms:10.3e}  {value.rms:10.3e}  "
                f"{'yes' if exact else 'no'}"
            )
        print(
            "first_bitwise_difference="
            + ("none" if first_difference is None else str(first_difference))
        )
        if args.details_layer is not None:
            layer = args.details_layer
            if layer < 0 or layer >= len(left.layers):
                raise ValueError("--details-layer is outside the dump")
            common_stages = [
                name
                for name in STAGE_NAMES
                if name in left.layers[layer].stages
                and name in right.layers[layer].stages
            ]
            print(f"details_layer={layer}")
            print("stage                       rms         max      cosine  exact")
            if (
                left.layers[layer].attention_context is not None
                and right.layers[layer].attention_context is not None
            ):
                context_metrics = metrics(
                    left.layers[layer].attention_context,
                    right.layers[layer].attention_context,
                )
                print(
                    f"{'attention_context':25s} {context_metrics.rms:10.3e} "
                    f"{context_metrics.maximum:10.3e} "
                    f"{context_metrics.cosine:10.7f} "
                    f"{'yes' if context_metrics.exact else 'no'}"
                )
            for name in common_stages:
                stage_metrics = metrics(
                    left.layers[layer].stages[name],
                    right.layers[layer].stages[name],
                )
                print(
                    f"{name:25s} {stage_metrics.rms:10.3e} "
                    f"{stage_metrics.maximum:10.3e} "
                    f"{stage_metrics.cosine:10.7f} "
                    f"{'yes' if stage_metrics.exact else 'no'}"
                )
        return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
