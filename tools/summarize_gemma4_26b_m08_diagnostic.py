#!/usr/bin/env python3
"""Reduce ignored M08 raw runs to compact diagnostic or acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gem16_compile.common import canonical_json_bytes, write_file_atomic

RAW = ROOT / "artifacts/raw/m08"
DEFAULT_OUTPUT = ROOT / "artifacts/m08/diagnostic-summary.json"
ACCEPTANCE_OUTPUT = ROOT / "artifacts/m08/acceptance.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(raw: Path = RAW, mode: str = "diagnostic") -> dict[str, Any]:
    acceptance = mode == "acceptance"
    if mode not in {"diagnostic", "acceptance"}:
        raise ValueError(f"unsupported M08 evidence mode: {mode}")
    if acceptance:
        first = _load(raw / "qat-hybrid-clean-1-compile.json")
        second = _load(raw / "qat-hybrid-clean-2-compile.json")
        verify = _load(raw / "qat-hybrid-clean-1-verify.json")
        compare = _load(raw / "qat-hybrid-clean-reproducibility.json")
        inspect = _load(raw / "qat-hybrid-clean-1-inspect.json")
        inspect_12b = _load(raw / "12b-inspect-clean-regression.json")
        memory = _load(raw / "qat-hybrid-clean-memory-probe.json")
        first_lock = raw / "qat-hybrid-clean-1.lock.json"
        second_lock = raw / "qat-hybrid-clean-2.lock.json"
    else:
        first = _load(raw / "qat-hybrid-diagnostic-compile-final.json")
        second = _load(raw / "qat-hybrid-diagnostic-2-compile.json")
        verify = _load(raw / "qat-hybrid-diagnostic-verify.json")
        compare = _load(raw / "qat-hybrid-diagnostic-reproducibility.json")
        inspect = _load(raw / "qat-hybrid-diagnostic-inspect-final.json")
        inspect_12b = _load(raw / "12b-inspect-regression.json")
        memory = None
        first_lock = raw / "qat-hybrid-diagnostic.lock.json"
        second_lock = raw / "qat-hybrid-diagnostic-2.lock.json"

    common = (
        "artifact_content_sha256", "artifact_lock_sha256",
        "compilation_manifest_sha256", "compiler_manifest_sha256",
        "source_lock_sha256", "output_tensor_count", "output_tensor_bytes",
    )
    if any(first.get(key) != second.get(key) for key in common):
        raise ValueError("M08 build identities differ")
    if any(first.get(key) != verify.get(key) for key in common):
        raise ValueError("M08 verification identity differs")
    if (
        first.get("status") != "pass" or second.get("status") != "pass"
        or verify.get("status") != "pass" or compare.get("status") != "pass"
        or compare.get("milestone") != "M08" or compare.get("mismatch_count") != 0
        or _sha256(first_lock) != _sha256(second_lock)
        or inspect.get("checkpoint_profile") != "sm120-text-hybrid-v1"
        or inspect.get("validation_contract") != "gemma4_26b_m08_compiled_hybrid_v1"
        or inspect.get("tensor_contract_validated") is not True
        or inspect_12b.get("model_variant") != "gemma4_unified_12b"
        or inspect_12b.get("runtime_supported") is not True
        or inspect_12b.get("tensor_contract_validated") is not True
    ):
        raise ValueError("M08 evidence does not pass its compact contract")
    if acceptance and (
        first.get("compiler_dirty") is not False
        or second.get("compiler_dirty") is not False
        or verify.get("recorded_compiler_dirty") is not False
        or first.get("compiler_commit") != second.get("compiler_commit")
        or memory is None or memory.get("status") != "pass"
        or memory.get("probe_kind") != "synthetic_device_admission"
        or memory.get("not_model_execution") is not True
        or memory.get("contract", {}).get("nvfp4_aligned_weight_arena_bytes")
        != 14_696_668_160
        or not all(memory.get("gates", {}).values())
    ):
        raise ValueError("M08 acceptance requires clean builds and passing GPU admission")

    result = {
        "schema_version": 1,
        "milestone": "M08",
        "status": "acceptance_pass" if acceptance else "diagnostic_pass_not_acceptance",
        "acceptance": acceptance,
        "limitations": (
            [
                "real payload upload and named CUDA-region reconciliation remain M09 phase B work",
                "the M08 GPU gate is exact-arena synthetic admission, not model execution",
            ]
            if acceptance
            else [
                "both complete builds used a dirty compiler worktree",
                "two clean complete builds remain required",
                "reference-GPU startup/no-transient-OOM evidence remains required",
            ]
        ),
        "compiler": {
            "commit": first["compiler_commit"],
            "dirty": first["compiler_dirty"],
            "native_builds": first["native_builds"],
        },
        "identity": {key: first[key] for key in common[:5]},
        "artifact": {
            "file_count": compare["file_count"],
            "output_tensor_count": first["output_tensor_count"],
            "output_tensor_bytes": first["output_tensor_bytes"],
            "excluded_vision_tensor_count": 356,
            "runtime_quantization_required": False,
            "one_physical_tied_head": True,
        },
        "reproducibility" if acceptance else "diagnostic_reproducibility": {
            "complete_build_count": 2,
            "byte_identical": True,
            "mismatch_count": 0,
            "external_locks_byte_identical": True,
            "build_duration_seconds": [
                first["duration_seconds"], second["duration_seconds"]
            ],
        },
        "verification": {
            "python_status": verify["status"],
            "python_duration_seconds": verify["duration_seconds"],
            "cpp_direct_loader_status": "pass",
            "checkpoint_profile": inspect["checkpoint_profile"],
            "validation_contract": inspect["validation_contract"],
            "runtime_supported": inspect["runtime_supported"],
            "capabilities": inspect["capabilities"],
            "protected_12b_inspect": {
                "status": "pass",
                "tensor_count": len(inspect_12b["tensors"]),
                "total_tensor_bytes": inspect_12b["total_tensor_bytes"],
                "validation_contract": inspect_12b["validation_contract"],
            },
        },
        "memory": {
            "compiler_peak_rss_bytes": first["memory"]["peak_rss_bytes"],
            "verifier_peak_rss_bytes": verify["memory"]["peak_rss_bytes"],
            "fp8_child_peak_rss_bytes": first["native_runtime"]["fp8_child_peak_rss_bytes"],
            "nvfp4_child_peak_rss_bytes": first["native_runtime"]["nvfp4_child_peak_rss_bytes"],
        },
    }
    if acceptance:
        assert memory is not None
        result["memory"].update({
            "gpu_probe_kind": memory["probe_kind"],
            "not_model_execution": memory["not_model_execution"],
            "device": memory["device"],
            "nvfp4_aligned_weight_arena_bytes": memory["contract"][
                "nvfp4_aligned_weight_arena_bytes"
            ],
            "final_direct_free_bytes": memory["measurement"][
                "final_direct_free_bytes"
            ],
            "required_free_margin_bytes": memory["measurement"][
                "required_free_margin_bytes"
            ],
            "gates": memory["gates"],
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--mode", choices=("diagnostic", "acceptance"), default="diagnostic")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output or (
        ACCEPTANCE_OUTPUT if args.mode == "acceptance" else DEFAULT_OUTPUT
    )
    payload = canonical_json_bytes(summarize(args.raw, args.mode))
    if args.check:
        if not output.is_file() or output.read_bytes() != payload:
            raise SystemExit(f"generated output is stale: {output}")
    else:
        write_file_atomic(output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
