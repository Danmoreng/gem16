#!/usr/bin/env python3
"""Compare vLLM NVFP4 activation quantization with the gem16 host contract.

This is offline correctness tooling. The C++ runtime does not depend on Python,
PyTorch, or vLLM.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

from compare_states import INTERMEDIATE_ELEMENTS, load_state


E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX_RECIPROCAL = np.float32(1.0 / 6.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--stage", default="gelu_product")
    parser.add_argument("--global-divisor", type=float, required=True)
    return parser.parse_args()


def decode_e4m3fn(bits: int) -> float:
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    if exponent == 0:
        return math.ldexp(float(mantissa), -9)
    return math.ldexp(1.0 + float(mantissa) / 8.0, exponent - 7)


E4M3FN_LEVELS = tuple(decode_e4m3fn(bits) for bits in range(0x7F))


def nearest(values: tuple[float, ...], value: float) -> int:
    best = 0
    best_error = math.inf
    for candidate, decoded in enumerate(values):
        error = abs(value - decoded)
        if error < best_error or (
            error == best_error and candidate & 1 == 0 and best & 1 != 0
        ):
            best = candidate
            best_error = error
    return best


def encode_e4m3fn(value: np.float32) -> int:
    if value >= np.float32(448.0):
        return 0x7E
    return nearest(E4M3FN_LEVELS, float(value))


def encode_e2m1(value: np.float32) -> int:
    magnitude = nearest(E2M1_LEVELS, abs(float(value)))
    return magnitude | (0x8 if np.signbit(value) else 0)


def oracle(values: list[float], divisor: np.float32) -> tuple[bytes, bytes]:
    if not values or len(values) % 16:
        raise ValueError("activation extent must be a nonzero multiple of 16")
    packed = bytearray(len(values) // 2)
    scales = bytearray(len(values) // 16)
    source = np.asarray(values, dtype=np.float32)
    for block in range(len(scales)):
        begin = block * 16
        amax = np.max(np.abs(source[begin : begin + 16]))
        scale_value = np.float32(
            np.float32(amax * E2M1_MAX_RECIPROCAL) * divisor
        )
        scale_bits = encode_e4m3fn(scale_value)
        scales[block] = scale_bits
        decoded_scale = np.float32(E4M3FN_LEVELS[scale_bits])
        for local in range(0, 16, 2):
            nibbles = []
            for offset in (local, local + 1):
                scaled = np.float32(source[begin + offset] * divisor)
                normalized = (
                    np.float32(0.0)
                    if decoded_scale == 0
                    else np.float32(scaled / decoded_scale)
                )
                nibbles.append(encode_e2m1(normalized))
            packed[(begin + local) // 2] = nibbles[0] | (nibbles[1] << 4)
    return bytes(packed), bytes(scales)


def mismatch_summary(left: bytes, right: bytes) -> tuple[int, list[int]]:
    mismatches = [index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]]
    return len(mismatches), mismatches[:16]


def main() -> int:
    args = parse_args()
    if (
        args.layer < 0
        or not math.isfinite(args.global_divisor)
        or args.global_divisor <= 0
    ):
        print("error: invalid layer or global divisor", file=sys.stderr)
        return 2
    try:
        import torch
        from vllm._custom_ops import scaled_fp4_quant

        state = load_state(args.state)
        values = state.layers[args.layer].stages[args.stage]
        if args.stage == "gelu_product" and len(values) != INTERMEDIATE_ELEMENTS:
            raise ValueError("unexpected GELU-product extent")
        tensor = torch.tensor(
            values, dtype=torch.bfloat16, device="cuda"
        ).reshape(1, -1)
        divisor = torch.tensor(
            [args.global_divisor], dtype=torch.float32, device="cuda"
        )
        packed, scales = scaled_fp4_quant(
            tensor,
            divisor,
            is_sf_swizzled_layout=False,
        )
        reference_packed, reference_scales = oracle(
            list(values), np.float32(args.global_divisor)
        )
        actual_packed = packed.contiguous().cpu().numpy().tobytes()
        actual_scales = scales.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        packed_count, packed_head = mismatch_summary(
            actual_packed, reference_packed
        )
        scale_count, scale_head = mismatch_summary(
            actual_scales, reference_scales
        )
        print(
            f"elements={len(values)} packed_bytes={len(actual_packed)} "
            f"scale_bytes={len(actual_scales)}"
        )
        print(
            f"packed_mismatches={packed_count} first_packed={packed_head}"
        )
        print(f"scale_mismatches={scale_count} first_scales={scale_head}")
        source = np.asarray(values, dtype=np.float32)
        divisor_f32 = np.float32(args.global_divisor)
        for byte_index in packed_head:
            element = byte_index * 2
            block = element // 16
            scale = np.float32(E4M3FN_LEVELS[actual_scales[block] & 0x7F])
            block_begin = block * 16
            amax = np.max(np.abs(source[block_begin : block_begin + 16]))
            raw_scale = np.float32(
                np.float32(amax * E2M1_MAX_RECIPROCAL) * divisor_f32
            )
            normalized = [
                np.float32(np.float32(source[element + offset] * divisor_f32) / scale)
                if scale != 0
                else np.float32(0.0)
                for offset in (0, 1)
            ]
            normalized_raw = [
                np.float32(
                    np.float32(source[element + offset] * divisor_f32)
                    / raw_scale
                )
                if raw_scale != 0
                else np.float32(0.0)
                for offset in (0, 1)
            ]
            print(
                f"packed[{byte_index}] actual=0x{actual_packed[byte_index]:02x} "
                f"reference=0x{reference_packed[byte_index]:02x} "
                f"scale={float(scale)} raw_scale={float(raw_scale)} "
                f"input={[float(source[element]), float(source[element + 1])]} "
                f"normalized={[float(value) for value in normalized]} "
                f"normalized_raw={[float(value) for value in normalized_raw]}"
            )
        return 0 if packed_count == 0 and scale_count == 0 else 1
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
