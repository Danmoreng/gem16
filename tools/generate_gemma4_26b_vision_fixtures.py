#!/usr/bin/env python3
"""Generate deterministic, repository-owned BMP fixtures for Vision V10."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "tests/fixtures/gemma4_26b_vision"

CASES = (
    (70, "square", 96, 96, 384, 384, 576, 64),
    (70, "wide", 100, 70, 480, 336, 630, 70),
    (70, "tall", 70, 100, 336, 480, 630, 70),
    (140, "square", 96, 96, 528, 528, 1089, 121),
    (140, "wide", 70, 50, 672, 480, 1260, 140),
    (140, "tall", 50, 70, 480, 672, 1260, 140),
    (280, "square", 96, 96, 768, 768, 2304, 256),
    (280, "wide", 100, 70, 960, 672, 2520, 280),
    (280, "tall", 70, 100, 672, 960, 2520, 280),
)


def pixel(x: int, y: int, width: int, height: int, seed: int) -> tuple[int, int, int]:
    grid = x % max(3, width // 7) == 0 or y % max(3, height // 7) == 0
    diagonal = abs(x * height - y * width) < max(width, height)
    box = width // 5 <= x < 4 * width // 5 and height // 4 <= y < 3 * height // 4
    red = (17 * x + 29 * y + 31 * seed) & 255
    green = (43 * x + 11 * y + 47 * seed) & 255
    blue = (7 * x + 53 * y + 61 * seed) & 255
    if box:
        red, green, blue = 224, 72 + seed % 96, 48
    if diagonal:
        red, green, blue = 32, 220, 96
    if grid:
        red, green, blue = 238, 238, 238
    return red, green, blue


def bmp(width: int, height: int, seed: int) -> bytes:
    row_bytes = ((width * 3 + 3) // 4) * 4
    pixel_bytes = bytearray(row_bytes * height)
    for y in range(height):
        destination_y = height - 1 - y
        for x in range(width):
            red, green, blue = pixel(x, y, width, height, seed)
            offset = destination_y * row_bytes + x * 3
            pixel_bytes[offset : offset + 3] = bytes((blue, green, red))
    offset = 14 + 40
    size = offset + len(pixel_bytes)
    header = struct.pack("<2sIHHI", b"BM", size, 0, 0, offset)
    dib = struct.pack(
        "<IIIHHIIIIII",
        40,
        width,
        height,
        1,
        24,
        0,
        len(pixel_bytes),
        2835,
        2835,
        0,
        0,
    )
    return header + dib + pixel_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "license": "repository-owned generated test fixture",
        "generator": "tools/generate_gemma4_26b_vision_fixtures.py",
        "cases": [],
    }
    for index, case in enumerate(CASES):
        budget, shape, width, height, processed_width, processed_height, raw, soft = case
        name = f"budget-{budget}-{shape}.bmp"
        data = bmp(width, height, budget + index)
        (output / name).write_bytes(data)
        manifest["cases"].append(
            {
                "budget": budget,
                "shape": shape,
                "path": name,
                "source_width": width,
                "source_height": height,
                "source_sha256": hashlib.sha256(data).hexdigest(),
                "processed_width": processed_width,
                "processed_height": processed_height,
                "raw_patch_count": raw,
                "soft_token_count": soft,
            }
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
