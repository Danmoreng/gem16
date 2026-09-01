#!/usr/bin/env python3
"""Compare a GEM16 Vision dump with the pinned padded BF16 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from compare_gemma4_26b_vision_padding import load_modules, run_once


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input-prefix", type=Path, required=True)
    parser.add_argument("--gem16-output", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=(70, 140, 280), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.detach().float().cpu().reshape(-1)
    candidate = candidate.detach().float().cpu().reshape(-1)
    difference = candidate - reference
    reference_energy = torch.dot(reference, reference).item()
    candidate_energy = torch.dot(candidate, candidate).item()
    difference_energy = torch.dot(difference, difference).item()
    denominator = math.sqrt(max(reference_energy * candidate_energy, 1.0e-30))
    return {
        "elements": reference.numel(),
        "max_absolute_error": difference.abs().max().item(),
        "relative_l2": math.sqrt(difference_energy / max(reference_energy, 1.0e-30)),
        "cosine": torch.dot(reference, candidate).item() / denominator,
    }


def main() -> int:
    args = arguments()
    device = torch.device(args.device)
    patches_path = Path(str(args.input_prefix) + ".patches.f32")
    positions_path = Path(str(args.input_prefix) + ".positions.i32")
    patches = np.fromfile(patches_path, dtype=np.float32)
    positions = np.fromfile(positions_path, dtype=np.int32)
    if patches.size % 768 or positions.size % 2:
        raise SystemExit("invalid GEM16 Vision diagnostic input shape")
    raw_tokens = positions.size // 2
    if patches.size != raw_tokens * 768 or raw_tokens % 9:
        raise SystemExit("inconsistent GEM16 Vision diagnostic input")
    capacity = args.budget * 9
    if raw_tokens > capacity:
        raise SystemExit("diagnostic input exceeds selected budget")

    padded_values = torch.zeros((1, capacity, 768), dtype=torch.float32)
    padded_values[:, :raw_tokens] = torch.from_numpy(
        patches.reshape(1, raw_tokens, 768)
    )
    padded_positions = torch.full((1, capacity, 2), -1, dtype=torch.int64)
    padded_positions[:, :raw_tokens] = torch.from_numpy(
        positions.reshape(1, raw_tokens, 2).astype(np.int64)
    )
    vision, projector, _config = load_modules(args.source.resolve(), device)
    oracle = run_once(
        vision, projector, padded_values.to(device), padded_positions.to(device)
    )["final_2816"]
    candidate_values = np.fromfile(args.gem16_output, dtype=np.float32)
    expected_elements = (raw_tokens // 9) * 2816
    if candidate_values.size != expected_elements:
        raise SystemExit("GEM16 Vision output has an unexpected shape")
    candidate = torch.from_numpy(candidate_values.reshape(raw_tokens // 9, 2816))
    report = {
        "schema_version": 1,
        "purpose": "padded GEM16 FP8 Vision versus pinned padded BF16 oracle",
        "diagnostic_only": True,
        "image_sha256": sha256(args.image),
        "budget": args.budget,
        "raw_tokens": raw_tokens,
        "soft_tokens": raw_tokens // 9,
        "padded_raw_tokens": capacity,
        "gem16_output_sha256": sha256(args.gem16_output),
        "metrics": comparison(oracle, candidate),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
