#!/usr/bin/env python3
"""Reconcile accepted 26B evidence and freeze the technical M23 Target."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class QualificationFailure(RuntimeError):
    pass


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationFailure(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationFailure(f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise QualificationFailure(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationFailure(message)


def relative_display(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def raw_record(path: Path) -> dict[str, Any]:
    return {
        "ignored_path": relative_display(path),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    m20 = load_object(args.m20)
    m21 = load_object(args.m21)
    m22 = load_object(args.m22)
    product_26b = load_object(args.product_26b)
    product_12b = load_object(args.product_12b)
    first = load_object(args.first_driver)
    relaunch = load_object(args.relaunch_driver)

    require(m20.get("acceptance") is True, "M20 is not accepted")
    require(m21.get("acceptance") is True, "M21 is not accepted")
    require(m22.get("status") == "accepted", "M22 is not accepted")
    require(product_26b.get("passed") is True, "updated 26B product test failed")
    require(product_12b.get("passed") is True, "updated 12B product test failed")
    require(first.get("passed") is True, "first product driver failed")
    require(relaunch.get("passed") is True, "relaunch product driver failed")

    artifact = m20["model"]["artifact_content_sha256"]
    source_lock = m20["model"]["source_lock_sha256"]
    artifact_lock = m20["model"]["artifact_lock_sha256"]
    toolchain_lock = m20["toolchain_lock_sha256"]
    require(
        artifact
        == m21["candidate"]["artifact_content_sha256"]
        == m22["qualified_artifact_content_sha256"]
        == first["artifact_content_sha256"]
        == relaunch["artifact_content_sha256"],
        "M20/M21/M22 product artifact hashes do not match",
    )
    require(
        source_lock
        == m21["candidate"]["source_lock_sha256"]
        == m22["source_lock_sha256"]
        == first["source_lock_sha256"]
        == relaunch["source_lock_sha256"],
        "M20/M21/M22 product source locks do not match",
    )
    require(
        artifact_lock == m21["candidate"]["artifact_lock_sha256"],
        "M20 and M21 artifact locks do not match",
    )
    require(
        toolchain_lock == m21["candidate"]["toolchain_lock_sha256"],
        "M20 and M21 toolchain locks do not match",
    )
    require(
        m20["native_instruction_evidence"]["binary_sha256"]
        == m21["candidate"]["benchmark_binary_sha256"],
        "M20 and M21 benchmark binaries do not match",
    )
    require(
        all(run.get("output_token_sha256") == args.output_token_sha256
            for run in m20["runs"]),
        "M20 retained runs do not share the frozen output-token hash",
    )
    require(
        all(run.get("fallback_count") == 0 and
            run.get("recurring_allocation_count") == 0
            for run in m20["runs"]),
        "M20 contains fallback or recurring-allocation evidence",
    )
    warmups = m20.get("configuration", {}).get("warmups")
    retained_runs = m20.get("configuration", {}).get("retained_runs")
    summary = m20.get("summary", {}).get(
        m20.get("promotion", {}).get("scenario"), {}
    )
    require(
        warmups == 3 and retained_runs == 10 and len(m20["runs"]) == 13,
        "M20 does not contain the required 3-warm-up/10-retained protocol",
    )
    require(
        summary.get("prompt_tps", {}).get("count") == retained_runs
        and summary.get("decode_tps", {}).get("count") == retained_runs,
        "M20 retained summary counts do not match its protocol",
    )
    require(m21.get("base_64k_result") == "passed", "M21 did not pass 64K")
    require(m21.get("base_max_context") == 98304, "unexpected base maximum")
    require(
        m21.get("first_capacity_rejection") == 102400,
        "unexpected first capacity rejection",
    )
    require(first == relaunch, "fresh product driver processes are not identical")

    model_report = product_26b.get("model_report", {})
    server_health = product_26b.get("server_health", {})
    for name, report in (("model report", model_report), ("server health", server_health)):
        require(report.get("default_context") == 32768, f"{name} default changed")
        require(report.get("qualified_64k") is True, f"{name} does not report 64K")
        require(
            report.get("base_max_context") == 98304,
            f"{name} reports the wrong base maximum",
        )
        require(report.get("mtp_max_context") is None, f"{name} advertises MTP")
    require(
        product_12b.get("run_output", {}).get("output_token_ids")
        == product_12b.get("expected_output_token_ids"),
        "protected 12B output changed",
    )
    require(
        product_12b.get("run_output", {}).get("fallbacks") == 0,
        "protected 12B path used a fallback",
    )

    inputs = {
        "m20_acceptance": {"path": relative_display(args.m20), "sha256": sha256(args.m20)},
        "m21_acceptance": {"path": relative_display(args.m21), "sha256": sha256(args.m21)},
        "m22_acceptance": {"path": relative_display(args.m22), "sha256": sha256(args.m22)},
    }
    binaries = {
        "gem16_chat_sha256": sha256(args.chat),
        "gem16_server_sha256": sha256(args.server),
        "m22_product_driver_sha256": sha256(args.product_driver),
        "protected_12b_runner_sha256": sha256(args.run_binary),
    }
    require(
        product_26b.get("qualified_binaries")
        == {
            "gem16_chat_sha256": binaries["gem16_chat_sha256"],
            "gem16_server_sha256": binaries["gem16_server_sha256"],
            "m22_product_driver_sha256": binaries["m22_product_driver_sha256"],
        },
        "26B product evidence is not bound to the supplied binaries",
    )
    require(
        product_12b.get("qualified_binaries")
        == {
            "gem16_server_sha256": binaries["gem16_server_sha256"],
            "protected_12b_runner_sha256": binaries[
                "protected_12b_runner_sha256"
            ],
        },
        "12B product evidence is not bound to the supplied binaries",
    )
    raw = [
        raw_record(args.first_driver),
        raw_record(args.relaunch_driver),
        raw_record(args.product_26b),
        raw_record(args.product_12b),
    ]

    return {
        "schema_version": 1,
        "milestone": "M23",
        "status": "accepted_technical_target",
        "accepted_on": args.accepted_on,
        "implementation_revision": args.implementation_revision,
        "target": {
            "model_variant": "gemma4_moe_26b_a4b",
            "artifact_profile": "sm120-text-hybrid-v1",
            "artifact_content_sha256": artifact,
            "artifact_lock_sha256": artifact_lock,
            "source_lock_sha256": source_lock,
            "compiler_commit": first["compiler_commit"],
            "toolchain_lock_sha256": toolchain_lock,
            "native_path": first["native_path"],
            "head_format": first["head_format"],
            "kv_cache": "checkpoint_fp8",
            "output_token_sha256": args.output_token_sha256,
        },
        "capability_statement": {
            "experimental_text_only": True,
            "batch_size": 1,
            "maximum_execution_slots": 1,
            "default_context_tokens": 32768,
            "qualified_64k": True,
            "base_max_context_tokens": 98304,
            "first_capacity_rejection_tokens": 102400,
            "supports_audio": False,
            "supports_vision": False,
            "supports_mtp": False,
            "mtp_max_context_tokens": None,
            "m19_full_quality_suite": "owner_deferred_pending",
            "shipping_or_production_quality_claim": False,
        },
        "accepted_evidence": {
            "inputs": inputs,
            "same_artifact_source_and_toolchain": True,
            "m20": {
                "scenario": m20["promotion"]["scenario"],
                "prompt_tps_median": m20["promotion"]["prompt_tps_median"],
                "decode_tps_median": m20["promotion"]["decode_tps_median"],
                "benchmark_binary_sha256": m20["native_instruction_evidence"]["binary_sha256"],
                "retained_runs": retained_runs,
                "warmups": warmups,
            },
            "m21": {
                "qualified_contexts": [32768, 65536, 98304],
                "base_max_context_tokens": 98304,
                "first_capacity_rejection_tokens": 102400,
                "context_driver_sha256": m21["candidate"]["context_driver_sha256"],
            },
            "m22_revalidation": {
                "product_26b": "passed",
                "protected_12b": "passed",
                "fresh_process_bitwise_equal": True,
                "qualified_binaries": binaries,
            },
        },
        "rollback": {
            "m25_base_target": "this M23 ordinary-decode Target",
            "ordinary_decode_remains_available_when_mtp_is_disabled": True,
            "failed_or_nonbeneficial_mtp_does_not_replace_this_target": True,
            "router_rollback": "serial_exact selected explicitly before engine execution",
            "silent_precision_model_cpu_or_runtime_fallback_allowed": False,
            "protected_12b_is_a_separate_product_profile_not_a_silent_model_substitution": True,
        },
        "raw_evidence": raw,
        "limitations": [
            "M19 full held-out task/prose quality qualification remains owner-deferred and pending.",
            "This is an experimental engineering Target, not a shipping or production-quality release.",
            "Vision and audio are excluded; MTP remains disabled until M25 passes its independent gates.",
            "M21 boundary prompts are synthetic; M20 performance uses the real Wikipedia 16K manifest.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m20", type=Path, required=True)
    parser.add_argument("--m21", type=Path, required=True)
    parser.add_argument("--m22", type=Path, required=True)
    parser.add_argument("--product-26b", type=Path, required=True)
    parser.add_argument("--product-12b", type=Path, required=True)
    parser.add_argument("--first-driver", type=Path, required=True)
    parser.add_argument("--relaunch-driver", type=Path, required=True)
    parser.add_argument("--chat", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--product-driver", type=Path, required=True)
    parser.add_argument("--run-binary", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--accepted-on", required=True)
    parser.add_argument("--output-token-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = qualify(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except QualificationFailure as exc:
        print(f"M23 qualification failed: {exc}")
        return 1
    print(f"M23 technical Target accepted: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
