#!/usr/bin/env python3
"""Reduce ignored M12 real-GPU and sanitizer evidence to a compact record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/m12/diagnostic-summary.json"
ACCEPTANCE_OUTPUT = ROOT / "artifacts/m12/acceptance.json"
THRESHOLDS = {"boundary_relative_l2_max": 0.07,
              "boundary_cosine_min": 0.998}
EXPECTED_KV = {8192: 188_743_680, 32768: 440_401_920,
               65536: 775_946_240}


def load(path: Path, maximum: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"unsafe or oversized M12 input: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("M12 JSON root must be an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def revision() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                         cwd=ROOT, text=True).strip())
    return commit, dirty


def summarize(fixture_path: Path, gpu_path: Path, sanitizer_paths: list[Path],
              acceptance: bool) -> dict[str, Any]:
    fixture = load(fixture_path, 4 * 1024 * 1024)
    gpu = load(gpu_path, 1024 * 1024)
    if (fixture.get("schema_version") != 1 or fixture.get("milestone") != "M12" or
            gpu.get("schema_version") != 1 or gpu.get("milestone") != "M12" or
            gpu.get("artifact_arena_bytes") != 14_696_668_160):
        raise ValueError("M12 fixture/GPU identity mismatch")
    cases = gpu.get("cases")
    if not isinstance(cases, list) or [case.get("layer") for case in cases] != [0, 5]:
        raise ValueError("M12 real cases must be layer 0 and layer 5")
    expected_boundaries = {
        0: {"q_raw", "k_raw", "v_raw", "q_normalized", "k_normalized",
            "v_normalized", "attention", "output_projection", "post_attention"},
        5: {"q_raw", "k_raw", "q_normalized", "k_normalized",
            "v_normalized", "attention", "output_projection", "post_attention"},
    }
    worst_relative_l2 = 0.0
    worst_cosine = 1.0
    for case in cases:
        layer = int(case["layer"])
        metrics = case.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != expected_boundaries[layer]:
            raise ValueError("M12 boundary metric set mismatch")
        for record in metrics.values():
            values = [float(record[name]) for name in ("max_abs", "relative_l2", "cosine")]
            if any(not math.isfinite(value) for value in values):
                raise ValueError("M12 metrics contain non-finite values")
            worst_relative_l2 = max(worst_relative_l2, values[1])
            worst_cosine = min(worst_cosine, values[2])
    sanitizers = []
    for path in sanitizer_paths:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError(f"unsafe M12 sanitizer log: {path}")
        text = path.read_text(encoding="utf-8", errors="strict")
        passed = ("ERROR SUMMARY: 0 errors" in text or
                  "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in text)
        sanitizers.append({"tool": path.stem, "status": "pass" if passed else "fail",
                           "sha256": sha256(path), "bytes": path.stat().st_size})
    lifecycle = all(
        case.get("repeated_bitwise_identical") is True and
        case.get("separate_cache_addresses") is True and
        case.get("free_before_repeats_bytes") == case.get("free_after_repeats_bytes")
        for case in cases
    )
    semantics = (
        cases[0].get("attention_type") == "sliding" and
        cases[0].get("stores_v_projection") is True and
        cases[0].get("reuses_raw_k_for_v") is False and
        cases[1].get("attention_type") == "full" and
        cases[1].get("stores_v_projection") is False and
        cases[1].get("reuses_raw_k_for_v") is True
    )
    passed = (worst_relative_l2 <= THRESHOLDS["boundary_relative_l2_max"] and
              worst_cosine >= THRESHOLDS["boundary_cosine_min"] and
              lifecycle and semantics and len(sanitizers) == 3 and
              all(item["status"] == "pass" for item in sanitizers))
    if not passed:
        raise ValueError("M12 attention/KV evidence exceeds its fixed gates")
    commit, dirty = revision()
    if acceptance and dirty:
        raise ValueError("M12 acceptance requires a clean implementation commit")
    report: dict[str, Any] = {
        "schema_version": 1, "milestone": "M12",
        "status": "acceptance_pass" if acceptance else
                  "diagnostic_pass_acceptance_pending_clean_commit",
        "acceptance": acceptance,
        "code_revision": commit if acceptance else commit + ("-dirty" if dirty else ""),
        "path": "cuda_attention_correctness_only_not_performance_qualified",
        "artifact_arena_bytes": gpu["artifact_arena_bytes"],
        "traits": {"layers": 30, "sliding_layers": 25, "full_layers": 5,
                   "query_heads": 16, "local_kv_heads": 8,
                   "global_kv_heads": 2, "local_head_dimension": 256,
                   "global_head_dimension": 512, "local_window": 1024,
                   "max_positions": 262144, "cross_layer_kv_sharing": 0},
        "fp8_kv_bytes": {str(key): value for key, value in EXPECTED_KV.items()},
        "real_cases": cases,
        "thresholds": THRESHOLDS,
        "observed_worst": {"relative_l2": worst_relative_l2,
                           "cosine": worst_cosine},
        "lifecycle": {"repeated_bitwise_identical": True,
                      "forward_allocation_free": True,
                      "host_routing": False, "cache_commit": "after_attention"},
        "sanitizers": sanitizers,
        "raw_evidence": {
            "fixture": {"sha256": sha256(fixture_path),
                        "bytes": fixture_path.stat().st_size},
            "gpu": {"sha256": sha256(gpu_path), "bytes": gpu_path.stat().st_size}},
        "limitations": [
            "correctness-only scalar FP8 projection path; no native performance claim",
            "real comparison is position 0 for one local and one global layer; ring wrap and append boundaries use focused synthetic CUDA fixtures",
            "quantized-versus-BF16 thresholds measure the locked M08 QAT artifact rather than byte identity",
        ],
    }
    if acceptance:
        report["implementation_commit"] = commit
        report["owner_decision"] = {"date": "2026-08-22", "decision": "M12 accepted"}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--sanitizer", type=Path, action="append", required=True)
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.sanitizer) != 3:
        raise SystemExit("exactly three sanitizer logs are required")
    output = args.output or (ACCEPTANCE_OUTPUT if args.acceptance else DEFAULT_OUTPUT)
    report = summarize(args.fixture, args.gpu, args.sanitizer, args.acceptance)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
