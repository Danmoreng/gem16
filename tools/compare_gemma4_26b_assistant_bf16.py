#!/usr/bin/env python3
"""Compare one native M25 hybrid proposal with Google's locked BF16 Assistant."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Gemma4AssistantForCausalLM


VOCABULARY = 262_144
BACKBONE_HIDDEN = 2_816
SUPPRESSED = (258_883, 258_882)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_array(path: Path, dtype: np.dtype, elements: int) -> np.ndarray:
    expected = elements * np.dtype(dtype).itemsize
    if not path.is_file() or path.stat().st_size != expected:
        raise ValueError(f"{path} must contain exactly {expected} bytes")
    return np.fromfile(path, dtype=dtype, count=elements)


def bf16_bits(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.copy()).view(torch.bfloat16).float()


def fp8_cache(path: Path, elements: int, scale: torch.Tensor,
              heads: int, dimension: int) -> torch.Tensor:
    raw = exact_array(path, np.uint8, elements)
    values = torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn).float()
    tokens = elements // (heads * dimension)
    return (values * scale).reshape(tokens, heads, dimension).permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model.resolve(strict=True)
    fixture = args.fixture.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if output.exists():
        raise ValueError("refusing to overwrite BF16 oracle report")
    metadata = json.loads((fixture / "fixture.json").read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version", "status", "input_token", "position", "tokens",
        "target_token", "hybrid_draft_token", "hybrid_target_match",
    }
    if (
        set(metadata) != expected_keys
        or metadata["schema_version"] != 1
        or metadata["status"] != "diagnostic_oracle_input"
        or metadata["tokens"] != 1
        or metadata["position"] < 0
        or not 0 <= metadata["input_token"] < VOCABULARY
        or not 0 <= metadata["target_token"] < VOCABULARY
        or not 0 <= metadata["hybrid_draft_token"] < VOCABULARY
        or metadata["hybrid_target_match"]
        != (metadata["hybrid_draft_token"] == metadata["target_token"])
    ):
        raise ValueError("unexpected M25 oracle fixture metadata")

    concatenated_np = exact_array(
        fixture / "concatenated.f32", np.float32, 2 * BACKBONE_HIDDEN
    )
    hybrid_np = exact_array(
        fixture / "hybrid-logits.f32", np.float32, VOCABULARY
    )
    if not np.isfinite(concatenated_np).all() or not np.isfinite(hybrid_np).all():
        raise ValueError("native M25 fixture contains non-finite values")
    scales = bf16_bits(exact_array(
        fixture / "kv-scales.bf16bits", np.uint16, 4
    ))
    if not torch.isfinite(scales).all() or not torch.all(scales > 0):
        raise ValueError("native M25 fixture has invalid KV scales")

    shared_kv = {
        "sliding_attention": (
            fp8_cache(fixture / "sliding-key.e4m3fn", 8 * 256,
                      scales[0], 8, 256),
            fp8_cache(fixture / "sliding-value.e4m3fn", 8 * 256,
                      scales[1], 8, 256),
        ),
        "full_attention": (
            fp8_cache(fixture / "full-key.e4m3fn", 2 * 512,
                      scales[2], 2, 512),
            fp8_cache(fixture / "full-value.e4m3fn", 2 * 512,
                      scales[3], 2, 512),
        ),
    }

    loaded_at = time.perf_counter()
    model = Gemma4AssistantForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    load_seconds = time.perf_counter() - loaded_at
    inputs = torch.from_numpy(concatenated_np.copy()).reshape(1, 1, -1).to(torch.bfloat16)
    position_ids = torch.tensor([[metadata["position"]]], dtype=torch.long)
    started = time.perf_counter()
    with torch.no_grad():
        result = model(
            inputs_embeds=inputs,
            position_ids=position_ids,
            attention_mask=None,
            shared_kv_states=shared_kv,
            use_cache=False,
        )
    oracle_seconds = time.perf_counter() - started
    oracle = result.logits[0, 0].float()
    hybrid = torch.from_numpy(hybrid_np.copy())
    if not torch.isfinite(oracle).all():
        raise ValueError("official BF16 Assistant produced non-finite logits")
    oracle_for_selection = oracle.clone()
    hybrid_for_selection = hybrid.clone()
    for token in SUPPRESSED:
        oracle_for_selection[token] = -math.inf
        hybrid_for_selection[token] = -math.inf
    oracle_token = int(torch.argmax(oracle_for_selection).item())
    hybrid_token = int(torch.argmax(hybrid_for_selection).item())
    target_token = metadata.get("target_token")
    bf16_target_match = target_token is not None and oracle_token == target_token
    hybrid_target_match = target_token is not None and hybrid_token == target_token
    difference = hybrid - oracle
    oracle_norm = torch.linalg.vector_norm(oracle.double())
    difference_norm = torch.linalg.vector_norm(difference.double())
    relative_l2 = float((difference_norm / oracle_norm).item())
    cosine = float(torch.nn.functional.cosine_similarity(
        hybrid.double(), oracle.double(), dim=0
    ).item())
    top_k = 32
    oracle_top = set(torch.topk(oracle_for_selection, top_k).indices.tolist())
    hybrid_top = set(torch.topk(hybrid_for_selection, top_k).indices.tolist())

    files = sorted(path for path in fixture.iterdir() if path.is_file())
    report = {
        "schema_version": 1,
        "status": "diagnostic_target_verified_comparison",
        "oracle": {
            "repository": "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant",
            "revision": "9537141506fe8875b3ed45b264af13580cb29166",
            "model_sha256": sha256(model_dir / "model.safetensors"),
            "dtype": "BF16",
            "device": "cpu",
        },
        "fixture": {
            "path": str(fixture),
            "files": {path.name: sha256(path) for path in files},
            "position": metadata["position"],
            "input_token": metadata["input_token"],
        },
        "comparison": {
            "bf16_token": oracle_token,
            "hybrid_token": hybrid_token,
            "target_token": target_token,
            "top1_equal": oracle_token == hybrid_token,
            "bf16_target_match": bf16_target_match,
            "hybrid_target_match": hybrid_target_match,
            "relative_l2": relative_l2,
            "cosine_similarity": cosine,
            "maximum_absolute_error": float(torch.max(torch.abs(difference)).item()),
            "top32_overlap": len(oracle_top & hybrid_top),
        },
        "timing": {
            "model_load_seconds": load_seconds,
            "oracle_forward_seconds": oracle_seconds,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["comparison"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
