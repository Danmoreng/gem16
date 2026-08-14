#!/usr/bin/env python3
"""Reduce ignored M09 raw runs to compact diagnostic or acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gem16_compile.common import canonical_json_bytes, write_file_atomic

RAW = ROOT / "artifacts/raw/m09"
DEFAULT_OUTPUT = ROOT / "artifacts/m09/diagnostic-summary.json"
ACCEPTANCE_OUTPUT = ROOT / "artifacts/m09/acceptance.json"
M08_ACCEPTANCE = ROOT / "artifacts/m08/acceptance.json"

EXPECTED_IDENTITY = {
    "artifact_content_sha256": "471805f7dad8abb84300be78b2822a63dcb1d35bff5aa98426a162cc8532ee17",
    "artifact_lock_sha256": "d7d2d30743e7c42aa55537a83f563047c9cbd2d83e0d583a2d2c8bcacbfa51a4",
    "compilation_manifest_sha256": "7e5e78b9c6f61fbe8829866395634085261e1261a8c783f69affc5a16bd1847a",
    "compiler_manifest_sha256": "5d7975756b331900e02a7de915cd1e473a88a36becd5582f91b4c57dd30b5f06",
    "source_lock_sha256": "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(probe: dict[str, Any], tokens: int) -> dict[str, Any]:
    matches = [item for item in probe.get("profiles", []) if item.get("context_tokens") == tokens]
    if len(matches) != 1:
        raise ValueError(f"M09 evidence requires exactly one {tokens}-token profile")
    return matches[0]


def summarize(raw: Path = RAW, mode: str = "diagnostic") -> dict[str, Any]:
    if mode not in {"diagnostic", "acceptance"}:
        raise ValueError(f"unsupported M09 evidence mode: {mode}")
    acceptance = mode == "acceptance"
    probe_name = "real-residency-acceptance.json" if acceptance else "real-residency-diagnostic.json"
    probe = _load(raw / probe_name)
    inspect_12b = _load(raw / "12b-inspect-regression.json")
    memory_12b_standard = _load(raw / "12b-memory-standard.json")
    memory_12b_long = _load(raw / "12b-memory-long.json")
    m08 = _load(M08_ACCEPTANCE)

    model_directory = Path(str(probe.get("model_directory", "")))
    if not model_directory.is_absolute():
        model_directory = ROOT / model_directory
    artifact_lock_path = model_directory.parent / f"{model_directory.name}.lock.json"
    artifact_lock = _load(artifact_lock_path)

    identity = m08.get("identity")
    artifact = probe.get("artifact", {})
    upload = probe.get("upload", {})
    measurement = probe.get("measurement", {})
    gates = probe.get("gates", {})
    p32 = _profile(probe, 32_768)
    p64 = _profile(probe, 65_536)
    candidate_tokens = measurement.get("measured_max_context_candidate_tokens")
    if not isinstance(candidate_tokens, int) or candidate_tokens <= 32_768:
        raise ValueError("M09 evidence lacks a measured context candidate above 32K")
    candidate = _profile(probe, candidate_tokens)

    expected_gates = {
        "idle_baseline_at_most_512_mib",
        "immutable_weights_at_most_14100_mib",
        "immutable_weights_below_14300_mib_hard_stop",
        "real_32k_slot_at_least_700_mib",
        "base_profiles_measured",
        "64k_classified",
        "real_64k_slot_at_least_400_mib",
        "max_context_candidate_measured",
        "second_slot_rejected_before_allocation",
        "second_slot_check_has_zero_allocation_delta",
        "free_after_release_at_least_context_baseline",
    }
    if (
        identity != EXPECTED_IDENTITY
        or _sha256(artifact_lock_path) != identity["artifact_lock_sha256"]
        or artifact_lock.get("artifact_content_sha256")
        != identity["artifact_content_sha256"]
        or artifact_lock.get("source_lock_sha256")
        != identity["source_lock_sha256"]
        or probe.get("milestone") != "M09"
        or probe.get("status") != "pass"
        or probe.get("probe_kind") != "real_artifact_residency"
        or probe.get("not_model_execution") is not True
        or artifact.get("validation_contract") != "gemma4_26b_m08_compiled_hybrid_v1"
        or artifact.get("tensor_count") != 1_285
        or artifact.get("shard_count") != 16
        or artifact.get("payload_bytes") != 14_696_569_196
        or artifact.get("immutable_weight_arena_bytes") != 14_696_668_160
        or artifact.get("one_device_weight_arena") is not True
        or artifact.get("persistent_device_repack_bytes") != 0
        or artifact.get("mapped_shards_one_at_a_time") is not True
        or upload.get("status") != "pass"
        or upload.get("tensor_count") != 1_285
        or upload.get("direct_tensor_count") != 983
        or upload.get("nvfp4_weight_tensor_count") != 151
        or upload.get("nvfp4_scale_tensor_count") != 151
        or set(gates) != expected_gates
        or not all(gates.values())
        or p32.get("preflight_admitted") is not True
        or p32.get("allocations_complete") is not True
        or p32.get("margin_pass") is not True
        or p32.get("required_margin_bytes") != 700 * 1024 * 1024
        or p32.get("final_free_bytes", 0) < 700 * 1024 * 1024
        or p64.get("preflight_admitted") is not True
        or p64.get("allocations_complete") is not True
        or p64.get("margin_pass") is not True
        or p64.get("required_margin_bytes") != 400 * 1024 * 1024
        or p64.get("final_free_bytes", 0) < 400 * 1024 * 1024
        or candidate.get("preflight_admitted") is not True
        or candidate.get("allocations_complete") is not True
        or candidate.get("margin_pass") is not True
        or candidate.get("required_margin_bytes") != 400 * 1024 * 1024
        or candidate.get("final_free_bytes", 0) < 400 * 1024 * 1024
        or measurement.get("free_before_second_slot_bytes")
        != measurement.get("free_after_second_slot_check_bytes")
    ):
        raise ValueError("M09 real-residency evidence does not pass its compact contract")

    if (
        inspect_12b.get("model_variant") != "gemma4_unified_12b"
        or inspect_12b.get("runtime_supported") is not True
        or inspect_12b.get("tensor_contract_validated") is not True
        or len(inspect_12b.get("tensors", [])) != 1_389
        or inspect_12b.get("total_tensor_bytes") != 9_304_786_336
        or memory_12b_standard.get("status") != "ok"
        or memory_12b_standard.get("fallbacks") != 0
        or memory_12b_standard.get("context_tokens") != 32_768
        or memory_12b_standard.get("selected_kv_bytes") != 436_207_616
        or memory_12b_standard.get("total_arena_bytes") != 9_740_994_304
        or memory_12b_long.get("status") != "ok"
        or memory_12b_long.get("fallbacks") != 0
        or memory_12b_long.get("context_tokens") != 65_536
        or memory_12b_long.get("selected_kv_bytes") != 704_643_072
        or memory_12b_long.get("total_arena_bytes") != 10_009_429_760
    ):
        raise ValueError("M09 protected 12B evidence does not pass its compact contract")

    code_revision = probe.get("code_revision")
    if not isinstance(code_revision, str):
        raise ValueError("M09 evidence lacks a code revision")
    clean_revision = re.fullmatch(r"[0-9a-f]{40}", code_revision) is not None
    if acceptance and not clean_revision:
        raise ValueError("M09 acceptance requires a clean 40-character code revision")
    if not acceptance and not code_revision.endswith("-dirty"):
        raise ValueError("M09 diagnostic evidence must identify its dirty worktree")

    return {
        "schema_version": 1,
        "milestone": "M09",
        "status": "acceptance_pass" if acceptance else "diagnostic_pass_not_acceptance",
        "acceptance": acceptance,
        "code_revision": code_revision,
        "identity": identity,
        "device": probe["device"],
        "artifact_residency": {
            "tensor_count": artifact["tensor_count"],
            "shard_count": artifact["shard_count"],
            "payload_bytes": artifact["payload_bytes"],
            "artifact_content_sha256": identity["artifact_content_sha256"],
            "artifact_lock_sha256": identity["artifact_lock_sha256"],
            "immutable_weight_arena_bytes": artifact["immutable_weight_arena_bytes"],
            "one_device_weight_arena": True,
            "persistent_device_repack_bytes": 0,
            "mapped_shards_one_at_a_time": True,
            "upload_duration_seconds": upload["duration_seconds"],
            "host_staging_peak_bytes": upload["host_staging_peak_bytes"],
            "uploaded_tensors": {
                "direct": upload["direct_tensor_count"],
                "nvfp4_weight_transform": upload["nvfp4_weight_tensor_count"],
                "nvfp4_scale_transform": upload["nvfp4_scale_tensor_count"],
            },
        },
        "one_slot": {
            "context_tokens": 32_768,
            "fp8_kv_bytes": p32["fp8_kv_bytes"],
            "fixed_region_bytes": p32["free_before_bytes"] - p32["fp8_kv_bytes"] - p32["final_free_bytes"],
            "final_free_bytes": p32["final_free_bytes"],
            "required_margin_bytes": p32["required_margin_bytes"],
            "pass": True,
        },
        "context_feasibility": {
            "64k": {
                "status": "admitted",
                "fp8_kv_bytes": p64["fp8_kv_bytes"],
                "final_free_bytes": p64["final_free_bytes"],
                "projected_final_free_bytes": p64["projected_final_free_bytes"],
                "required_margin_bytes": p64["required_margin_bytes"],
                "margin_shortfall_bytes": p64["margin_shortfall_bytes"],
                "partial_allocation": False,
                "allocations_complete": True,
            },
            "measured_candidate": {
                "context_tokens": candidate_tokens,
                "fp8_kv_bytes": candidate["fp8_kv_bytes"],
                "final_free_bytes": candidate["final_free_bytes"],
                "required_margin_bytes": candidate["required_margin_bytes"],
                "pass": True,
                "advertised": False,
            },
        },
        "second_slot": {
            "status": "preflight_rejected",
            "partial_allocation": False,
            "cuda_allocation_delta_bytes": 0,
        },
        "protected_12b": {
            "inspect_tensor_count": len(inspect_12b["tensors"]),
            "inspect_total_tensor_bytes": inspect_12b["total_tensor_bytes"],
            "standard_32k_total_arena_bytes": memory_12b_standard["total_arena_bytes"],
            "long_64k_total_arena_bytes": memory_12b_long["total_arena_bytes"],
            "status": "pass",
        },
        "gates": gates,
        "limitations": [
            "this is real artifact upload and residency accounting, not model execution",
            "fixed graph/workspace regions are touched reserves, not captured execution graphs",
            "warm model execution and graph capture begin with M11/M12 and must revalidate the 32K margin",
            "the 65536-token result is residency-only and cannot be advertised before warm execution and correctness qualification",
        ] + ([] if acceptance else ["the implementation worktree was dirty; clean commit-bound acceptance remains required"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--mode", choices=("diagnostic", "acceptance"), default="diagnostic")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output or (ACCEPTANCE_OUTPUT if args.mode == "acceptance" else DEFAULT_OUTPUT)
    payload = canonical_json_bytes(summarize(args.raw, args.mode))
    if args.check:
        if not output.is_file() or output.read_bytes() != payload:
            raise SystemExit(f"generated output is stale: {output}")
    else:
        write_file_atomic(output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
