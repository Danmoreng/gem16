#!/usr/bin/env python3
"""Reduce ignored M11 CUDA output and sanitizer logs to compact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/m11/diagnostic-summary.json"
ACCEPTANCE_OUTPUT = ROOT / "artifacts/m11/acceptance.json"
THRESHOLDS = {
    "router_probability_max_abs": 0.003,
    "router_weight_max_abs": 0.005,
    "shared_relative_l2_max": 0.20,
    "shared_cosine_min": 0.985,
    "expert_relative_l2_max": 0.30,
    "expert_cosine_min": 0.96,
    "routed_relative_l2_max": 0.20,
    "routed_cosine_min": 0.985,
}


def load(path: Path, maximum: int = 16 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"unsafe or oversized M11 input: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("M11 JSON root must be an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(actual: list[float], expected: list[float]) -> dict[str, float]:
    if len(actual) != len(expected) or not actual:
        raise ValueError("M11 boundary shape mismatch")
    left = [float(value) for value in actual]
    right = [float(value) for value in expected]
    if any(not math.isfinite(value) for value in left + right):
        raise ValueError("M11 boundary contains non-finite values")
    difference = [a - b for a, b in zip(left, right)]
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    return {
        "max_abs": max(abs(value) for value in difference),
        "relative_l2": math.sqrt(math.fsum(value * value for value in difference)) /
                       max(right_norm, 1e-30),
        "cosine": math.fsum(a * b for a, b in zip(left, right)) /
                  max(left_norm * right_norm, 1e-30),
    }


def revision() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                         cwd=ROOT, text=True).strip())
    return commit, dirty


def summarize(fixture_path: Path, gpu_path: Path,
              sanitizer_paths: list[Path], acceptance: bool) -> dict[str, Any]:
    fixture = load(fixture_path)
    gpu = load(gpu_path)
    if (fixture.get("schema_version") != 1 or fixture.get("milestone") != "M11" or
            gpu.get("schema_version") != 1 or gpu.get("milestone") != "M11" or
            gpu.get("path") != "cuda_correctness_only" or
            gpu.get("artifact_arena_bytes") != 14_696_668_160):
        raise ValueError("M11 fixture/GPU identity mismatch")
    expected = fixture.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("M11 expected boundaries are missing")
    expected_ids = [int(value) for value in expected["top_ids"]]
    actual_ids = [int(value) for value in gpu["top_ids"]]
    if len(expected_ids) != 8 or len(actual_ids) != 8 or len(set(actual_ids)) != 8:
        raise ValueError("M11 top-8 IDs are invalid")
    router_probability = metrics(gpu["router_probabilities"],
                                 expected["router_probabilities"])
    expected_weight_by_id = dict(zip(expected_ids, expected["top_weights"]))
    actual_weight_by_id = dict(zip(actual_ids, gpu["top_weights"]))
    router_weight_max_abs = max(
        abs(float(actual_weight_by_id[expert]) - float(expected_weight_by_id[expert]))
        for expert in expected_ids
    ) if set(actual_ids) == set(expected_ids) else math.inf
    shared = metrics(gpu["shared_output"], expected["shared_output"])
    expected_contribution_by_id = dict(zip(expected_ids, expected["expert_contributions"]))
    actual_contribution_by_id = dict(zip(actual_ids, gpu["expert_contributions"]))
    expert_records = []
    if set(actual_ids) == set(expected_ids):
        for expert in expected_ids:
            expert_records.append({
                "expert_id": expert,
                "metrics": metrics(actual_contribution_by_id[expert],
                                   expected_contribution_by_id[expert]),
            })
    routed = metrics(gpu["routed_sum"], expected["routed_sum"])
    sanitizers = []
    for path in sanitizer_paths:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError(f"unsafe M11 sanitizer log: {path}")
        text = path.read_text(encoding="utf-8", errors="strict")
        passed = ("ERROR SUMMARY: 0 errors" in text or
                  "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in text)
        sanitizers.append({"tool": path.stem, "status": "pass" if passed else "fail",
                           "sha256": sha256(path), "bytes": path.stat().st_size})
    commit, dirty = revision()
    if acceptance and dirty:
        raise ValueError("M11 acceptance requires a clean implementation commit")
    passed = (
        set(actual_ids) == set(expected_ids) and
        router_probability["max_abs"] <= THRESHOLDS["router_probability_max_abs"] and
        router_weight_max_abs <= THRESHOLDS["router_weight_max_abs"] and
        shared["relative_l2"] <= THRESHOLDS["shared_relative_l2_max"] and
        shared["cosine"] >= THRESHOLDS["shared_cosine_min"] and
        len(expert_records) == 8 and
        max(record["metrics"]["relative_l2"] for record in expert_records) <=
            THRESHOLDS["expert_relative_l2_max"] and
        min(record["metrics"]["cosine"] for record in expert_records) >=
            THRESHOLDS["expert_cosine_min"] and
        routed["relative_l2"] <= THRESHOLDS["routed_relative_l2_max"] and
        routed["cosine"] >= THRESHOLDS["routed_cosine_min"] and
        gpu.get("repeated_bitwise_identical") is True and
        gpu.get("forward_allocation_free") is True and
        len(sanitizers) == 3 and all(item["status"] == "pass" for item in sanitizers)
    )
    if not passed:
        raise ValueError("M11 CUDA correctness evidence exceeds its fixed gates")
    report: dict[str, Any] = {
        "schema_version": 1,
        "milestone": "M11",
        "status": "acceptance_pass" if acceptance else "diagnostic_pass_acceptance_pending_clean_commit",
        "acceptance": acceptance,
        "code_revision": commit if acceptance else commit + ("-dirty" if dirty else ""),
        "path": "cuda_correctness_only_not_performance_qualified",
        "artifact_arena_bytes": gpu["artifact_arena_bytes"],
        "routing": {
            "tie_policy": "lower_expert_id",
            "top8_set_match": True,
            "gpu_ids": actual_ids,
            "trusted_ids": expected_ids,
            "probability_metrics": router_probability,
            "weight_max_abs_by_expert_id": router_weight_max_abs,
        },
        "shared_metrics": shared,
        "expert_metrics": expert_records,
        "routed_sum_metrics": routed,
        "thresholds": THRESHOLDS,
        "lifecycle": {
            "repeated_bitwise_identical": True,
            "forward_allocation_free": True,
            "free_before_repeats_bytes": gpu["free_before_repeats_bytes"],
            "free_after_repeats_bytes": gpu["free_after_repeats_bytes"],
            "host_routing": False,
        },
        "sanitizers": sanitizers,
        "raw_evidence": {
            "fixture": {"sha256": sha256(fixture_path), "bytes": fixture_path.stat().st_size},
            "gpu": {"sha256": sha256(gpu_path), "bytes": gpu_path.stat().st_size},
        },
        "limitations": [
            "correctness-only CUDA reference; no native performance claim",
            "the trusted BF16 tie may order expert 58/83 differently; the accepted lower-ID policy is applied to GPU probabilities",
            "quantized-versus-BF16 boundary thresholds measure the locked M08 QAT artifact rather than byte identity",
        ],
    }
    if acceptance:
        report["implementation_commit"] = commit
        report["owner_decision"] = {"date": "2026-08-22", "decision": "M11 accepted"}
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
