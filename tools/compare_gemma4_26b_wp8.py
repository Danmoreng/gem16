#!/usr/bin/env python3
"""Compare frozen WP8 Trellis35 captures with the qualified 26B control."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
VOCAB = 262144


class ComparisonError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ComparisonError(f"{path} is not a JSON object")
    return document


def sha256(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def require_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)["sha256"]
    if actual != expected:
        raise ComparisonError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )


def read_logits(path: Path, vocab: int = VOCAB) -> array[float]:
    values = array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, vocab)
        if stream.read(1):
            raise ComparisonError(f"{path} contains trailing logit bytes")
    if len(values) != vocab:
        raise ComparisonError(
            f"{path} has {len(values)} logits; expected exactly {vocab}"
        )
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise ComparisonError(f"{path} contains a non-finite logit")
    return values


def vector_metrics(actual: Sequence[float], reference: Sequence[float]) -> dict[str, float]:
    if len(actual) != len(reference) or not actual:
        raise ComparisonError("vector extents differ or are empty")
    dot = 0.0
    actual_square = 0.0
    reference_square = 0.0
    error_square = 0.0
    maximum_absolute_error = 0.0
    for actual_value, reference_value in zip(actual, reference):
        if not math.isfinite(actual_value) or not math.isfinite(reference_value):
            raise ComparisonError("capture contains a non-finite value")
        dot += actual_value * reference_value
        actual_square += actual_value * actual_value
        reference_square += reference_value * reference_value
        difference = actual_value - reference_value
        error_square += difference * difference
        maximum_absolute_error = max(maximum_absolute_error, abs(difference))
    extent = len(actual)
    return {
        "relative_l2": math.sqrt(error_square / max(reference_square, 1.0e-30)),
        "cosine": dot / math.sqrt(max(actual_square * reference_square, 1.0e-30)),
        "rms_error": math.sqrt(error_square / extent),
        "max_abs_error": maximum_absolute_error,
    }


def probability_metrics(
    actual: Sequence[float], reference: Sequence[float]
) -> dict[str, float]:
    if len(actual) != len(reference) or not actual:
        raise ComparisonError("probability extents differ or are empty")
    epsilon = 1.0e-30
    actual_sum = sum(actual)
    reference_sum = sum(reference)
    if actual_sum <= 0.0 or reference_sum <= 0.0:
        raise ComparisonError("router probabilities do not have positive mass")
    l1 = 0.0
    kl = 0.0
    for actual_value, reference_value in zip(actual, reference):
        if actual_value < 0.0 or reference_value < 0.0:
            raise ComparisonError("router probability is negative")
        a = actual_value / actual_sum
        r = reference_value / reference_sum
        l1 += abs(a - r)
        if r > 0.0:
            kl += r * math.log(r / max(a, epsilon))
    return {"l1": l1, "kl_control_to_candidate": kl}


def logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def logit_metrics(
    candidate: Sequence[float], control: Sequence[float]
) -> dict[str, Any]:
    if len(candidate) != len(control) or not candidate:
        raise ComparisonError("logit extents differ or are empty")
    candidate_lse = logsumexp(candidate)
    control_lse = logsumexp(control)
    control_top1 = max(range(len(control)), key=control.__getitem__)
    candidate_top1 = max(range(len(candidate)), key=candidate.__getitem__)
    kl = 0.0
    for candidate_value, control_value in zip(candidate, control):
        probability = math.exp(control_value - control_lse)
        if probability:
            kl += probability * (
                (control_value - control_lse) - (candidate_value - candidate_lse)
            )
    control_top1_candidate_rank = 1 + sum(
        value > candidate[control_top1] for value in candidate
    )
    candidate_top20 = set(
        sorted(range(len(candidate)), key=candidate.__getitem__, reverse=True)[:20]
    )
    control_top20 = set(
        sorted(range(len(control)), key=control.__getitem__, reverse=True)[:20]
    )
    candidate_nll = candidate_lse - candidate[control_top1]
    control_nll = control_lse - control[control_top1]
    return {
        "kl_control_to_candidate": max(0.0, kl),
        "control_top1_token": control_top1,
        "candidate_top1_token": candidate_top1,
        "top1_agreement": control_top1 == candidate_top1,
        "control_top1_candidate_rank": control_top1_candidate_rank,
        "control_top1_nll_control": control_nll,
        "control_top1_nll_candidate": candidate_nll,
        "control_top1_nll_delta": candidate_nll - control_nll,
        "top20_set_overlap": len(candidate_top20 & control_top20),
    }


def capture_map(report: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    captures = report.get("captures")
    if not isinstance(captures, list):
        raise ComparisonError("report captures are missing")
    mapped: dict[tuple[int, int], dict[str, Any]] = {}
    for capture in captures:
        if not isinstance(capture, dict):
            raise ComparisonError("capture is not an object")
        key = (int(capture["position"]), int(capture["layer"]))
        if key in mapped:
            raise ComparisonError(f"duplicate capture {key}")
        mapped[key] = capture
    return mapped


def require_exact_capture_contract(
    report: dict[str, Any], layers: Iterable[int], positions: Iterable[int], label: str
) -> dict[tuple[int, int], dict[str, Any]]:
    mapped = capture_map(report)
    expected = {(position, layer) for position in positions for layer in layers}
    if set(mapped) != expected:
        missing = sorted(expected - set(mapped))
        extra = sorted(set(mapped) - expected)
        raise ComparisonError(
            f"{label} capture contract mismatch; missing={missing}, extra={extra}"
        )
    return mapped


def selected_qat_comparison(
    candidate_report: dict[str, Any],
    candidate_logits: Sequence[float],
    golden: dict[str, Any],
    golden_logits: Sequence[float],
) -> dict[str, Any]:
    candidate_captures = capture_map(candidate_report)

    def golden_row(name: str, position: int) -> dict[str, Any]:
        rows = golden.get("captures", {}).get(name, {}).get("rows", [])
        for row in rows:
            if row.get("position") == position:
                return row
        raise ComparisonError(f"QAT-BF16 golden is missing {name} at {position}")

    cases: list[dict[str, Any]] = []
    for position in (0, 17):
        for layer in (0, 5, 6, 29):
            actual = candidate_captures[(position, layer)]
            residual = vector_metrics(
                actual["output"],
                golden_row(f"layer_{layer}.output", position)["values_f32"],
            )
            expected_ids = golden_row(
                f"layer_{layer}.router_top_ids", position
            )["values_i64"]
            actual_ids = actual["router_top_ids"]
            cases.append(
                {
                    "position": position,
                    "layer": layer,
                    "residual": residual,
                    "router_top8_set_overlap": len(set(actual_ids) & set(expected_ids)),
                    "router_top8_ordered_agreement": sum(
                        actual_id == expected_id
                        for actual_id, expected_id in zip(actual_ids, expected_ids)
                    ),
                }
            )
    logits = logit_metrics(candidate_logits, golden_logits)
    target = int(golden["generated_token_ids"][0])
    candidate_lse = logsumexp(candidate_logits)
    golden_lse = logsumexp(golden_logits)
    logits.update(
        {
            "qat_target_token": target,
            "qat_target_candidate_rank": 1
            + sum(value > candidate_logits[target] for value in candidate_logits),
            "qat_target_nll_candidate": candidate_lse - candidate_logits[target],
            "qat_target_nll_golden": golden_lse - golden_logits[target],
        }
    )
    return {"cases": cases, "full_logits": logits}


def compare(
    suite: dict[str, Any],
    candidate_report: dict[str, Any],
    candidate_logits: Sequence[float],
    control_report: dict[str, Any],
    control_logits: Sequence[float],
    qat_golden: dict[str, Any],
    qat_logits: Sequence[float],
) -> dict[str, Any]:
    contract = suite["numerical_differential"]
    layers = [int(value) for value in contract["capture_layers"]]
    positions = [int(value) for value in contract["capture_positions"]]
    candidate_captures = require_exact_capture_contract(
        candidate_report, layers, positions, "candidate"
    )
    control_captures = require_exact_capture_contract(
        control_report, layers, positions, "control"
    )
    if candidate_report.get("prompt_token_ids") != control_report.get(
        "prompt_token_ids"
    ):
        raise ComparisonError("candidate and control prompt token IDs differ")
    if candidate_report.get("continuation_token_ids") != control_report.get(
        "continuation_token_ids"
    ):
        raise ComparisonError("candidate and control continuation token IDs differ")

    cases: list[dict[str, Any]] = []
    for position in positions:
        for layer in layers:
            candidate = candidate_captures[(position, layer)]
            control = control_captures[(position, layer)]
            candidate_ids = [int(value) for value in candidate["router_top_ids"]]
            control_ids = [int(value) for value in control["router_top_ids"]]
            cases.append(
                {
                    "position": position,
                    "layer": layer,
                    "residual": vector_metrics(candidate["output"], control["output"]),
                    "router": {
                        "top8_set_overlap": len(set(candidate_ids) & set(control_ids)),
                        "top8_ordered_agreement": sum(
                            candidate_id == control_id
                            for candidate_id, control_id in zip(
                                candidate_ids, control_ids
                            )
                        ),
                        "candidate_top8": candidate_ids,
                        "control_top8": control_ids,
                        **probability_metrics(
                            candidate["router_probabilities"],
                            control["router_probabilities"],
                        ),
                    },
                }
            )

    full_logits = logit_metrics(candidate_logits, control_logits)
    candidate_qat = selected_qat_comparison(
        candidate_report, candidate_logits, qat_golden, qat_logits
    )
    control_qat = selected_qat_comparison(
        control_report, control_logits, qat_golden, qat_logits
    )
    thresholds = contract["thresholds"]
    worst_relative_l2 = max(case["residual"]["relative_l2"] for case in cases)
    worst_cosine = min(case["residual"]["cosine"] for case in cases)
    minimum_router_overlap = min(
        case["router"]["top8_set_overlap"] for case in cases
    )
    candidate_qat_worst_relative_l2 = max(
        case["residual"]["relative_l2"] for case in candidate_qat["cases"]
    )
    candidate_qat_worst_cosine = min(
        case["residual"]["cosine"] for case in candidate_qat["cases"]
    )
    candidate_qat_minimum_router_overlap = min(
        case["router_top8_set_overlap"] for case in candidate_qat["cases"]
    )
    gates = {
        "prompt_ids_exact": True,
        "candidate_capture_mode_all": candidate_report.get("capture_layers") == "all",
        "control_capture_mode_all": control_report.get("capture_layers") == "all",
        "candidate_same_process_deterministic": bool(
            candidate_report.get("deterministic")
            and candidate_report.get("full_logits_repeat_equal")
        ),
        "control_same_process_deterministic": bool(
            control_report.get("deterministic")
            and control_report.get("full_logits_repeat_equal")
        ),
        "all_logits_finite": bool(
            candidate_report.get("all_logits_finite")
            and control_report.get("all_logits_finite")
        ),
        "all_layer_residual_envelope": (
            worst_relative_l2 <= thresholds["layer_relative_l2_max"]
            and worst_cosine >= thresholds["layer_cosine_min"]
        ),
        "all_layer_router_envelope": (
            minimum_router_overlap
            >= thresholds["router_top8_set_overlap_min"]
        ),
        "candidate_selected_qat_bf16_envelope": (
            candidate_qat_worst_relative_l2
            <= thresholds["layer_relative_l2_max"]
            and candidate_qat_worst_cosine >= thresholds["layer_cosine_min"]
            and candidate_qat_minimum_router_overlap
            >= thresholds["router_top8_set_overlap_min"]
        ),
        "full_logit_envelope": (
            full_logits["kl_control_to_candidate"]
            <= thresholds["full_logit_kl_control_to_candidate_max"]
            and full_logits["control_top1_candidate_rank"]
            <= thresholds["control_top1_candidate_rank_max"]
        ),
        "warm_run_allocation_free": (
            candidate_report["memory"]["free_after_first_run_bytes"]
            == candidate_report["memory"]["free_after_runs_bytes"]
            and control_report["memory"]["free_after_first_run_bytes"]
            == control_report["memory"]["free_after_runs_bytes"]
        ),
    }
    passed = all(gates.values())
    quality_gate_names = (
        "prompt_ids_exact",
        "candidate_capture_mode_all",
        "candidate_same_process_deterministic",
        "all_logits_finite",
        "all_layer_router_envelope",
        "candidate_selected_qat_bf16_envelope",
        "full_logit_envelope",
        "warm_run_allocation_free",
    )
    quality_gates = {name: gates[name] for name in quality_gate_names}
    return {
        "schema_version": 1,
        "work_packet": "WP8",
        "status": "numerical_pass" if passed else "numerical_diagnose",
        "decision": "proceed" if passed else "diagnose",
        "quality_decision": (
            "proceed" if all(quality_gates.values()) else "diagnose"
        ),
        "control_distance_decision": (
            "within_frozen_envelope"
            if gates["all_layer_residual_envelope"]
            else "outside_frozen_envelope_diagnose"
        ),
        "gates": gates,
        "quality_gates": quality_gates,
        "thresholds": thresholds,
        "summary": {
            "cases": len(cases),
            "worst_relative_l2": worst_relative_l2,
            "worst_cosine": worst_cosine,
            "minimum_router_top8_set_overlap": minimum_router_overlap,
            "mean_router_top8_set_overlap": sum(
                case["router"]["top8_set_overlap"] for case in cases
            )
            / len(cases),
            "maximum_router_probability_l1": max(
                case["router"]["l1"] for case in cases
            ),
            "maximum_router_kl_control_to_candidate": max(
                case["router"]["kl_control_to_candidate"] for case in cases
            ),
            "candidate_qat_bf16_selected_worst_relative_l2": (
                candidate_qat_worst_relative_l2
            ),
            "candidate_qat_bf16_selected_worst_cosine": (
                candidate_qat_worst_cosine
            ),
            "candidate_qat_bf16_selected_minimum_router_top8_set_overlap": (
                candidate_qat_minimum_router_overlap
            ),
        },
        "full_logits": full_logits,
        "all_layer_cases": cases,
        "qat_bf16_selected_anchor": {
            "candidate": candidate_qat,
            "control": control_qat,
        },
        "generation": {
            "candidate": candidate_report.get("first_generated"),
            "control": control_report.get("first_generated"),
            "candidate_continuation": candidate_report.get("continuation"),
            "control_continuation": control_report.get("continuation"),
        },
        "memory": {
            "candidate": candidate_report.get("memory"),
            "control": control_report.get("memory"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-logits", type=Path, required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--control-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    suite = load_json(args.suite)
    if suite.get("status") != "frozen_before_wp8_candidate_results":
        raise ComparisonError("WP8 suite is not the frozen pre-result contract")
    candidate_manifest = ROOT / suite["candidate"]["directory"] / suite[
        "candidate"
    ]["checkpoint_manifest"]
    control_manifest = ROOT / suite["control"]["directory"] / suite["control"][
        "compilation_manifest"
    ]
    tokenizer = ROOT / suite["candidate"]["directory"] / "tokenizer.json"
    qat_report_path = ROOT / suite["qat_bf16_anchor"]["report"]
    qat_logits_path = ROOT / suite["qat_bf16_anchor"]["logits"]
    require_sha(
        candidate_manifest,
        suite["candidate"]["checkpoint_manifest_sha256"],
        "candidate manifest",
    )
    require_sha(
        control_manifest,
        suite["control"]["compilation_manifest_sha256"],
        "control manifest",
    )
    require_sha(tokenizer, suite["shared_source"]["tokenizer_sha256"], "tokenizer")
    require_sha(
        qat_report_path, suite["qat_bf16_anchor"]["report_sha256"], "QAT report"
    )
    require_sha(
        qat_logits_path,
        suite["qat_bf16_anchor"]["logits_sha256"],
        "QAT logits",
    )

    result = compare(
        suite,
        load_json(args.candidate_report),
        read_logits(args.candidate_logits),
        load_json(args.control_report),
        read_logits(args.control_logits),
        load_json(qat_report_path),
        read_logits(qat_logits_path),
    )
    result["suite"] = sha256(args.suite)
    result["raw_evidence"] = {
        "candidate_report": sha256(args.candidate_report),
        "candidate_logits": sha256(args.candidate_logits),
        "control_report": sha256(args.control_report),
        "control_logits": sha256(args.control_logits),
        "qat_bf16_report": sha256(qat_report_path),
        "qat_bf16_logits": sha256(qat_logits_path),
    }
    result["code_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result["limitations"] = [
        "Discovery comparison only; it is not M19 product qualification.",
        "All-layer localization uses the qualified NVFP4 control; the immutable QAT-BF16 anchor contains selected layers 0/5/6/29 only.",
        "Weight-format and W4A8 activation effects are further separated by the WP4/WP5/WP6 operator-reference evidence, not by a full-model W4A16 Trellis arm.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["decision"] == "proceed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
