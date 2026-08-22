#!/usr/bin/env python3
"""Create compact M13 early-quality evidence from ignored raw reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 2816
VOCAB = 262144
LAYERS = (0, 5, 6, 29)
POSITIONS = (0, 17)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def floats(path: Path) -> tuple[float, ...]:
    payload = path.read_bytes()
    if len(payload) != VOCAB * 4:
        raise ValueError(f"unexpected logit payload size: {len(payload)}")
    return struct.unpack(f"<{VOCAB}f", payload)


def metrics(actual: list[float], expected: list[float]) -> dict[str, float]:
    if len(actual) != len(expected):
        raise ValueError("capture extents differ")
    dot = sum(a * e for a, e in zip(actual, expected))
    aa = sum(a * a for a in actual)
    ee = sum(e * e for e in expected)
    error = sum((a - e) ** 2 for a, e in zip(actual, expected))
    return {"relative_l2": math.sqrt(error / max(ee, 1.0e-30)),
            "cosine": dot / math.sqrt(max(aa * ee, 1.0e-30))}


def logsumexp(values: tuple[float, ...]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--repeat-report", type=Path, required=True)
    parser.add_argument("--repeat-logits", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=ROOT / "benchmarks/goldens/gemma4_26b/qat-bf16-selected/qat-bf16-selected.json")
    parser.add_argument("--golden-logits", type=Path, default=ROOT / "benchmarks/goldens/gemma4_26b/qat-bf16-selected/qat-bf16-final-logits.f32le")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/m13/diagnostic-summary.json")
    parser.add_argument("--acceptance", action="store_true")
    args = parser.parse_args()

    report, repeat_report, golden = load(args.report), load(args.repeat_report), load(args.golden)
    actual_logits, golden_logits = floats(args.logits), floats(args.golden_logits)
    captures = {(item["position"], item["layer"]): item
                for item in report["captures"]}

    def golden_row(name: str, position: int) -> dict[str, Any]:
        return next(row for row in golden["captures"][name]["rows"]
                    if row["position"] == position)

    comparisons = []
    worst_relative_l2 = 0.0
    worst_cosine = 1.0
    minimum_router_overlap = 8
    for position in POSITIONS:
        for layer in LAYERS:
            actual = captures[(position, layer)]
            expected_output = golden_row(f"layer_{layer}.output", position)["values_f32"]
            value_metrics = metrics(actual["output"], expected_output)
            expected_ids = golden_row(f"layer_{layer}.router_top_ids", position)["values_i64"]
            overlap = len(set(actual["router_top_ids"]) & set(expected_ids))
            worst_relative_l2 = max(worst_relative_l2, value_metrics["relative_l2"])
            worst_cosine = min(worst_cosine, value_metrics["cosine"])
            minimum_router_overlap = min(minimum_router_overlap, overlap)
            comparisons.append({"position": position, "layer": layer,
                                **value_metrics, "router_top8_overlap": overlap})

    actual_lse = logsumexp(actual_logits)
    golden_lse = logsumexp(golden_logits)
    kl = sum(math.exp(expected - golden_lse) *
             ((expected - golden_lse) - (actual - actual_lse))
             for actual, expected in zip(actual_logits, golden_logits))
    target = golden["generated_token_ids"][0]
    target_rank = 1 + sum(value > actual_logits[target] for value in actual_logits)
    target_nll = actual_lse - actual_logits[target]
    golden_target_nll = golden_lse - golden_logits[target]
    memory = report["memory"]
    thresholds = {"layer_relative_l2_max": 0.25, "layer_cosine_min": 0.98,
                  "router_top8_overlap_min": 5, "teacher_forced_kl_max": 0.02,
                  "teacher_target_rank_max": 5, "warm_free_margin_min_bytes": 700 * 1024 * 1024}
    gates = {
        "prompt_ids_exact": report["prompt_token_ids"] == golden["prompt"]["input_token_ids"],
        "same_process_deterministic": report["deterministic"] is True,
        "fresh_process_deterministic": report["first_generated"] == repeat_report["first_generated"] and args.logits.read_bytes() == args.repeat_logits.read_bytes(),
        "all_logits_finite": report["all_logits_finite"] is True,
        "prose_task_first_token_exact": report["first_generated"][:1] == [target],
        "selected_layer_envelope": worst_relative_l2 <= thresholds["layer_relative_l2_max"] and worst_cosine >= thresholds["layer_cosine_min"],
        "router_envelope": minimum_router_overlap >= thresholds["router_top8_overlap_min"],
        "teacher_forced_logit_envelope": kl <= thresholds["teacher_forced_kl_max"] and target_rank <= thresholds["teacher_target_rank_max"],
        "resident_continuation_stable": report["continuation"]["end_position"] > report["continuation"]["start_position"] and report["continuation"]["first_prediction"] == report["continuation"]["second_prediction"],
        "warm_forward_allocation_free": memory["free_after_first_run_bytes"] == memory["free_after_runs_bytes"],
        "memory_margin_32k": memory["free_after_runs_bytes"] >= thresholds["warm_free_margin_min_bytes"],
        "exact_fp8_kv_bytes": memory["kv_cache_bytes"] == 440_401_920,
    }
    passed = all(gates.values())
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = {
        "schema_version": 1, "milestone": "M13",
        "status": "acceptance_pass" if args.acceptance and passed else
                  "diagnostic_pass_acceptance_pending_clean_commit" if passed else "diagnose",
        "early_quality_decision": "proceed" if passed else "diagnose",
        "acceptance": bool(args.acceptance and passed),
        "implementation_commit": revision if args.acceptance else None,
        "code_revision": revision, "path": "experimental_reference_only_not_performance_qualified",
        "gates": gates, "thresholds": thresholds,
        "generation": {"first": report["first_generated"], "second": report["second_generated"],
                       "bf16_reference": golden["generated_token_ids"]},
        "teacher_forced": {"kl_bf16_to_m13": kl, "target_token": target,
                           "target_rank": target_rank, "target_nll": target_nll,
                           "bf16_target_nll": golden_target_nll},
        "selected_captures": {"worst_relative_l2": worst_relative_l2,
                              "worst_cosine": worst_cosine,
                              "minimum_router_top8_overlap": minimum_router_overlap,
                              "cases": comparisons},
        "continuation": report["continuation"], "memory": memory,
        "raw_evidence": {"report": sha(args.report), "logits": sha(args.logits),
                         "repeat_report": sha(args.repeat_report),
                         "repeat_logits": sha(args.repeat_logits),
                         "golden": sha(args.golden), "golden_logits": sha(args.golden_logits)},
        "limitations": [
            "single bounded calibration prose/task prompt; held-out quality remains M19",
            "correctness-only scalar FP8/NVFP4 operators; no performance claim",
            "generated second token differs from BF16/Q4_0 turn-end but matches the direct Unsloth NVFP4 diagnostic period token; first-token rank/KL and drift gates decide this early screen",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
