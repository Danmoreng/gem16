#!/usr/bin/env python3
"""Generate the deterministic WP1 Trellis35 format and byte-estimate records."""

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

from tools.gem16_compile.common import InvalidPlanError, canonical_json_bytes, load_json
from tools.gem16_compile.trellis35_layout import (
    DOWN,
    GATE_UP,
    GATE_UP_BOUNDARY,
    TRELLIS35_ALIGNMENT,
    TRELLIS35_DESCRIPTOR_BYTES,
    TRELLIS35_EXPERT_COUNT,
    TRELLIS35_HADAMARD_BLOCK,
    TRELLIS35_K3_COUNT,
    TRELLIS35_K4_COUNT,
    TRELLIS35_LAYER_COUNT,
    TRELLIS35_SIDECAR_ELEMENT_BYTES,
    TRELLIS35_TILE,
    estimate_trellis35_layout,
)


INVENTORY = ROOT / "benchmarks/goldens/gemma4_26b/source-inventories/qat-bf16.json"
QAT_LOCK = ROOT / "models/gemma4-26b-qat-bf16.lock.json"
BASELINE_MANIFEST = ROOT / "benchmarks/goldens/gemma4_26b/manifests/compiled-hybrid.json"
OUTPUT_SPEC = ROOT / "tools/gem16_compile/specs/trellis35-experts-v1.json"
OUTPUT_ESTIMATE = ROOT / "artifacts/trellis35/wp1-layout-estimate.json"

GATE_UP_NAME = "model.language_model.layers.{layer}.experts.gate_up_proj"
DOWN_NAME = "model.language_model.layers.{layer}.experts.down_proj"
SOURCE_REPOSITORY = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
SOURCE_REVISION = "f1e06dc520982d9b9edd76859fdb7ab209449949"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _required_uint(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidPlanError(f"{description} must be a non-negative integer")
    return value


def _validate_source() -> dict[str, str]:
    inventory = load_json(INVENTORY)
    lock = load_json(QAT_LOCK)
    lock_hash = _sha256(QAT_LOCK)
    if inventory.get("source_family") != "qat_bf16":
        raise InvalidPlanError("Trellis35 requires the frozen QAT-BF16 inventory")
    source = inventory.get("source")
    if not isinstance(source, dict):
        raise InvalidPlanError("QAT-BF16 inventory source must be an object")
    if source.get("lock_sha256") != lock_hash:
        raise InvalidPlanError("QAT-BF16 inventory lock hash mismatch")
    if source.get("revision") != lock.get("revision"):
        raise InvalidPlanError("QAT-BF16 inventory revision mismatch")
    if lock.get("repository") != SOURCE_REPOSITORY or lock.get("revision") != SOURCE_REVISION:
        raise InvalidPlanError("Trellis35 source repository or revision is not the owner-pinned source")
    tensors = inventory.get("tensors")
    if not isinstance(tensors, list):
        raise InvalidPlanError("QAT-BF16 inventory tensors must be an array")
    by_name = {
        item.get("name"): item for item in tensors
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if len(by_name) != len(tensors):
        raise InvalidPlanError("QAT-BF16 inventory contains invalid or duplicate tensor names")
    for layer in range(TRELLIS35_LAYER_COUNT):
        for template, shape in (
            (GATE_UP_NAME, [TRELLIS35_EXPERT_COUNT, GATE_UP.output, GATE_UP.logical_input]),
            (DOWN_NAME, [TRELLIS35_EXPERT_COUNT, DOWN.output, DOWN.logical_input]),
        ):
            name = template.format(layer=layer)
            item = by_name.get(name)
            expected_bytes = 2
            for dimension in shape:
                expected_bytes *= dimension
            if (
                item is None
                or item.get("dtype") != "BF16"
                or item.get("shape") != shape
                or item.get("bytes") != expected_bytes
            ):
                raise InvalidPlanError(f"unexpected Trellis35 source tensor: {name}")
    return {
        "lock_path": _relative(QAT_LOCK),
        "lock_sha256": lock_hash,
        "repository": str(lock.get("repository")),
        "revision": str(lock.get("revision")),
    }


def _baseline() -> tuple[int, int, int]:
    document = load_json(BASELINE_MANIFEST)
    profile = document.get("profiles", {}).get("nvfp4_head")
    if not isinstance(profile, dict):
        raise InvalidPlanError("compiled baseline lacks nvfp4_head")
    roles = profile.get("roles")
    if not isinstance(roles, dict):
        raise InvalidPlanError("compiled baseline lacks role accounting")
    try:
        arena = _required_uint(profile["aligned_weight_arena_bytes"], "baseline arena")
        gate_up = _required_uint(
            roles["routed_expert_gate_up"]["aligned_bytes"], "baseline Gate+Up"
        )
        down = _required_uint(roles["routed_expert_down"]["aligned_bytes"], "baseline Down")
    except (KeyError, TypeError) as error:
        raise InvalidPlanError("compiled baseline routed-expert accounting is invalid") from error
    return arena, gate_up, down


def make_spec(source: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format": "GEM16-Trellis35",
        "format_version": 1,
        "status": "experimental_not_runtime_loadable",
        "scope": "gemma4_26b_routed_experts_only",
        "source": source,
        "layer_count": TRELLIS35_LAYER_COUNT,
        "experts_per_layer": TRELLIS35_EXPERT_COUNT,
        "trellis_tile": [TRELLIS35_TILE, TRELLIS35_TILE],
        "hadamard_block": TRELLIS35_HADAMARD_BLOCK,
        "payload_rates": {
            "allowed_rate_bits": [3, 4],
            "k3_experts_per_family_layer": TRELLIS35_K3_COUNT,
            "k4_experts_per_family_layer": TRELLIS35_K4_COUNT,
            "payload_bpw_encoded": 3.5,
            "selection": "top_64_positive_proxy_error_k3_minus_k4",
            "tie_break": "ascending_expert_id",
            "final_rate_maps_deferred_to_wp2": True,
        },
        "descriptor": {
            "bytes": TRELLIS35_DESCRIPTOR_BYTES,
            "fields": [
                {"name": "pool_offset", "dtype": "U32"},
                {"name": "rate_bits", "dtype": "U16"},
                {"name": "codebook_id", "dtype": "U16"},
            ],
            "offset_validation": "checked_u64_before_narrowing",
        },
        "alignment_bytes": TRELLIS35_ALIGNMENT,
        "sidecar_dtype": "F16",
        "sidecar_element_bytes": TRELLIS35_SIDECAR_ELEMENT_BYTES,
        "codebook_storage": "kernel_builtin_selected_by_validated_id",
        "codebook_bytes": 0,
        "families": {
            "gate_up": {
                "source_name": GATE_UP_NAME,
                "logical_shape": [TRELLIS35_EXPERT_COUNT, GATE_UP.output, GATE_UP.logical_input],
                "physical_shape": [TRELLIS35_EXPERT_COUNT, GATE_UP.output, GATE_UP.physical_input],
                "logical_axis_order": "expert,gate_then_up,input",
                "padding": "none",
                "gate_up_boundary": GATE_UP_BOUNDARY,
                "gate_up_inverse_before_split": True,
                "suh_shape": [TRELLIS35_EXPERT_COUNT, GATE_UP.suh_elements_per_expert],
                "svh_shape": [TRELLIS35_EXPERT_COUNT, GATE_UP.svh_elements_per_expert],
            },
            "down": {
                "source_name": DOWN_NAME,
                "logical_shape": [TRELLIS35_EXPERT_COUNT, DOWN.output, DOWN.logical_input],
                "physical_shape": [TRELLIS35_EXPERT_COUNT, DOWN.output, DOWN.physical_input],
                "logical_axis_order": "expert,output,input",
                "padding": "input_zero_pad_704_to_768",
                "suh_shape": [TRELLIS35_EXPERT_COUNT, DOWN.suh_elements_per_expert],
                "svh_shape": [TRELLIS35_EXPERT_COUNT, DOWN.svh_elements_per_expert],
            },
        },
        "per_layer_region_order": [
            "gate_up_k3_payload_pool",
            "gate_up_k4_payload_pool",
            "gate_up_descriptor",
            "gate_up_suh",
            "gate_up_svh",
            "down_k3_payload_pool",
            "down_k4_payload_pool",
            "down_descriptor",
            "down_suh",
            "down_svh",
        ],
        "payload_bitstream_contract_deferred_to_wp2": True,
    }


def make_estimate(source: dict[str, str], spec_sha256: str) -> dict[str, Any]:
    arena, gate_up, down = _baseline()
    estimate = estimate_trellis35_layout(
        baseline_arena_bytes=arena,
        baseline_gate_up_bytes=gate_up,
        baseline_down_bytes=down,
    )
    return {
        "schema_version": 1,
        "work_package": "WP1",
        "status": "format_byte_estimate_not_runtime_evidence",
        "format": "GEM16-Trellis35",
        "format_version": 1,
        "source": source,
        "baseline_manifest": {
            "path": _relative(BASELINE_MANIFEST),
            "sha256": _sha256(BASELINE_MANIFEST),
        },
        "format_spec": {
            "path": _relative(OUTPUT_SPEC),
            "sha256": spec_sha256,
        },
        "rate_map_status": "deferred_until_wp2_proxy_errors",
        "estimate": estimate,
        "limitations": [
            "This is static byte accounting, not a compiled artifact or measured device arena.",
            "WP2 must finalize payload packing, codebook IDs, proxy errors, and all 60 rate maps.",
            "Context and workspace capacity are not requalified by this estimate.",
        ],
    }


def generated_documents() -> dict[Path, bytes]:
    source = _validate_source()
    spec_bytes = canonical_json_bytes(make_spec(source))
    # The report must hash the exact spec bytes, including when generated into a clean tree.
    spec_hash = hashlib.sha256(spec_bytes).hexdigest()
    estimate = make_estimate(source, spec_hash)
    return {
        OUTPUT_SPEC: spec_bytes,
        OUTPUT_ESTIMATE: canonical_json_bytes(estimate),
    }


def _write_or_check(check: bool) -> int:
    documents = generated_documents()
    stale: list[str] = []
    for path, payload in documents.items():
        if check:
            if not path.is_file() or path.read_bytes() != payload:
                stale.append(_relative(path))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f"wrote {_relative(path)}")
    if stale:
        print("stale generated Trellis35 files: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if checked-in outputs differ")
    arguments = parser.parse_args()
    try:
        return _write_or_check(arguments.check)
    except InvalidPlanError as error:
        print(str(error), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
