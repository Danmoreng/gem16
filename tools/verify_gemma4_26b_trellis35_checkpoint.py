#!/usr/bin/env python3
"""Independently verify a complete 30-layer GEM16-Trellis35 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import stat
import struct
import sys

import numpy as np

try:
    from tools.gem16_compile.common import (
        CompilerError,
        InvalidPlanError,
        canonical_json_bytes,
        load_json,
    )
    from tools.gem16_compile.trellis35_artifact import sha256_file
    from tools.gem16_compile.trellis35_layout import select_rate_map, validate_rate_map
    from tools.verify_gemma4_26b_trellis35_layer import SHAPES, region_layout
except ModuleNotFoundError:  # Direct execution with tools/ as sys.path[0].
    from gem16_compile.common import (
        CompilerError,
        InvalidPlanError,
        canonical_json_bytes,
        load_json,
    )
    from gem16_compile.trellis35_artifact import sha256_file
    from gem16_compile.trellis35_layout import select_rate_map, validate_rate_map
    from verify_gemma4_26b_trellis35_layer import SHAPES, region_layout


PROFILE = "gem16-trellis35-w4a8-v1"
SOURCE_LOCK = "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230"
LAYERS = 30
EXPERTS = 128
ALIGNMENT = 256
LAYER_BYTES = 345_147_392
NON_ROUTED_BYTES = 1_850_270_720
ROUTED_BYTES = LAYERS * LAYER_BYTES
TOTAL_BYTES = NON_ROUTED_BYTES + ROUTED_BYTES


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _hex_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular(path: Path, description: str) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise InvalidPlanError(f"{description} must be a regular non-symlink file")
    return metadata


def _safe_file(root: Path, relative: object, description: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise InvalidPlanError(f"{description} path is invalid")
    path = Path(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise InvalidPlanError(f"{description} path escapes the checkpoint")
    candidate = root
    for part in path.parts:
        candidate /= part
        if candidate.is_symlink():
            raise InvalidPlanError(f"{description} path contains a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise InvalidPlanError(f"{description} path escapes the checkpoint") from error
    _regular(resolved, description)
    return resolved


def _verify_content_hash(document: dict[str, object], description: str) -> str:
    recorded = document.get("checkpoint_content_sha256")
    if not _hex_sha256(recorded):
        raise InvalidPlanError(f"{description} content hash is invalid")
    unhashed = dict(document)
    del unhashed["checkpoint_content_sha256"]
    actual = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()
    if recorded != actual:
        raise InvalidPlanError(f"{description} content hash mismatch")
    return actual


def _verify_record_hash(path: Path, record: dict[str, object], field: str, description: str) -> str:
    expected = record.get(field)
    if not _hex_sha256(expected):
        raise InvalidPlanError(f"{description} SHA-256 is invalid")
    actual = sha256_file(path)
    if actual != expected:
        raise InvalidPlanError(f"{description} SHA-256 mismatch")
    return actual


def _verify_layer(root: Path, layer: int, record: object) -> dict[str, object]:
    if not isinstance(record, dict) or record.get("layer") != layer:
        raise InvalidPlanError(f"Trellis35 layer {layer} index record is invalid")
    artifact = _safe_file(root, record.get("artifact"), f"Trellis35 layer {layer} artifact")
    manifest_path = _safe_file(root, record.get("manifest"), f"Trellis35 layer {layer} manifest")
    verification_path = _safe_file(
        root, record.get("verification"), f"Trellis35 layer {layer} verification"
    )
    if record.get("artifact_bytes") != LAYER_BYTES or artifact.stat().st_size != LAYER_BYTES:
        raise InvalidPlanError(f"Trellis35 layer {layer} byte extent changed")
    artifact_sha256 = _verify_record_hash(
        artifact, record, "artifact_sha256", f"Trellis35 layer {layer} artifact"
    )
    _verify_record_hash(
        manifest_path, record, "manifest_sha256", f"Trellis35 layer {layer} manifest"
    )
    _verify_record_hash(
        verification_path,
        record,
        "verification_sha256",
        f"Trellis35 layer {layer} verification",
    )
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format") != "GEM16-Trellis35"
        or manifest.get("format_version") != 1
        or manifest.get("checkpoint_profile") != PROFILE
        or manifest.get("source_lock_sha256") != SOURCE_LOCK
        or manifest.get("layer") != layer
        or manifest.get("alignment_bytes") != ALIGNMENT
        or manifest.get("trellis_tile") != [16, 16]
        or manifest.get("hadamard_block") != 128
        or manifest.get("codebook_id") != 2
        or manifest.get("gate_up_boundary") != 704
        or manifest.get("gate_up_inverse_before_split") is not True
        or manifest.get("padding_contract")
        != {"gate_up": "none", "down": "input_zero_pad_704_to_768"}
    ):
        raise InvalidPlanError(f"Trellis35 layer {layer} manifest contract is invalid")
    artifact_meta = manifest.get("artifact")
    if (
        not isinstance(artifact_meta, dict)
        or artifact_meta.get("path") != artifact.name
        or artifact_meta.get("bytes") != LAYER_BYTES
        or artifact_meta.get("sha256") != artifact_sha256
    ):
        raise InvalidPlanError(f"Trellis35 layer {layer} artifact identity is invalid")
    regions = region_layout(manifest, LAYER_BYTES)
    rate_maps = manifest.get("rate_maps")
    proxies = manifest.get("candidate_proxy")
    if not isinstance(rate_maps, dict) or not isinstance(proxies, dict):
        raise InvalidPlanError(f"Trellis35 layer {layer} rate metadata is invalid")
    rates: dict[str, tuple[int, ...]] = {}
    for family in SHAPES:
        rates[family] = validate_rate_map(rate_maps.get(family))
        candidates = proxies.get(family)
        if not isinstance(candidates, list) or len(candidates) != EXPERTS:
            raise InvalidPlanError(f"Trellis35 layer {layer} {family} proxies are invalid")
        benefits: list[float] = []
        for expert, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or candidate.get("expert") != expert:
                raise InvalidPlanError(f"Trellis35 layer {layer} {family} proxy order is invalid")
            k3, k4, benefit = candidate.get("k3"), candidate.get("k4"), candidate.get("benefit")
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and float(value) >= 0.0
                for value in (k3, k4)
            ) or not isinstance(benefit, (int, float)) or isinstance(benefit, bool):
                raise InvalidPlanError(f"Trellis35 layer {layer} {family} proxy is non-finite")
            benefit = float(benefit)
            if not math.isfinite(benefit) or not math.isclose(
                benefit, float(k3) - float(k4), rel_tol=1e-12, abs_tol=1e-15
            ):
                raise InvalidPlanError(f"Trellis35 layer {layer} {family} benefit is invalid")
            if candidate.get("selected_rate") != rates[family][expert]:
                raise InvalidPlanError(f"Trellis35 layer {layer} {family} rate selection differs")
            benefits.append(benefit)
        if select_rate_map(benefits) != rates[family]:
            raise InvalidPlanError(f"Trellis35 layer {layer} {family} K4 ranking differs")

    with artifact.open("rb") as stream, mmap.mmap(
        stream.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        position = 0
        for region in regions.values():
            if any(data[position:region["offset"]]):
                raise InvalidPlanError(f"Trellis35 layer {layer} alignment gap is non-zero")
            position = region["offset"] + region["bytes"]
        if any(data[position:LAYER_BYTES]):
            raise InvalidPlanError(f"Trellis35 layer {layer} final alignment gap is non-zero")
        for family, (rows, columns, suh_elements, svh_elements) in SHAPES.items():
            descriptor = regions[f"{family}_descriptor"]
            next_offset = {3: 0, 4: 0}
            for expert, rate in enumerate(rates[family]):
                offset, actual_rate, codebook = struct.unpack_from(
                    "<IHH", data, descriptor["offset"] + expert * 8
                )
                if actual_rate != rate or codebook != 2 or offset != next_offset[rate]:
                    raise InvalidPlanError(
                        f"Trellis35 layer {layer} {family} descriptor {expert} is invalid"
                    )
                next_offset[rate] += rows * columns * rate // 8
            for rate in (3, 4):
                if next_offset[rate] != regions[f"{family}_k{rate}_payload_pool"]["bytes"]:
                    raise InvalidPlanError(
                        f"Trellis35 layer {layer} {family} K{rate} pool coverage differs"
                    )
            for sidecar, elements in (("suh", suh_elements), ("svh", svh_elements)):
                region = regions[f"{family}_{sidecar}"]
                values = np.frombuffer(
                    data, dtype="<f2", count=EXPERTS * elements, offset=region["offset"]
                )
                finite = bool(np.isfinite(values).all())
                del values
                if not finite:
                    raise InvalidPlanError(
                        f"Trellis35 layer {layer} {family} {sidecar} is non-finite"
                    )

    verification = load_json(verification_path)
    checks = verification.get("descriptor_checks")
    numerical = verification.get("numerical_checks")
    if (
        verification.get("schema_version") != 1
        or verification.get("work_package") != "WP2"
        or verification.get("status") != "complete_single_layer_artifact_pass"
        or verification.get("layer") != layer
        or verification.get("artifact") != artifact_meta
        or verification.get("manifest_sha256") != record.get("manifest_sha256")
        or verification.get("source_lock_sha256") != SOURCE_LOCK
        or verification.get("payload_bpw") != 3.5
        or verification.get("gate_up_padding") != "none"
        or verification.get("down_padding") != "704_to_768"
        or verification.get("gate_up_inverse_before_split") is not True
        or not _hex_sha256(verification.get("verifier_sha256"))
        or not isinstance(checks, dict)
        or not isinstance(numerical, list)
        or len(numerical) != 2
    ):
        raise InvalidPlanError(f"Trellis35 layer {layer} verification report is invalid")
    for family in SHAPES:
        if checks.get(family) != {
            "descriptors": EXPERTS,
            "k3": 64,
            "k4": 64,
            "pool_coverage": "exact",
        }:
            raise InvalidPlanError(f"Trellis35 layer {layer} descriptor evidence differs")
    if sorted(item.get("rate_bits") for item in numerical if isinstance(item, dict)) != [3, 4]:
        raise InvalidPlanError(f"Trellis35 layer {layer} numerical rate evidence differs")
    for item in numerical:
        if (
            item.get("codebook_id") != 2
            or not isinstance(item.get("quantized_output_nmse"), (int, float))
            or not math.isfinite(float(item["quantized_output_nmse"]))
            or float(item["quantized_output_nmse"]) < 0.0
            or not isinstance(item.get("full_output_semantic_max_abs"), (int, float))
            or not isinstance(item.get("gelu_product_semantic_max_abs"), (int, float))
            or float(item["full_output_semantic_max_abs"]) >= 2e-12
            or float(item["gelu_product_semantic_max_abs"]) >= 2e-12
        ):
            raise InvalidPlanError(f"Trellis35 layer {layer} numerical evidence is invalid")
    return {
        "layer": layer,
        "artifact_sha256": artifact_sha256,
        "gate_up_k3": rates["gate_up"].count(3),
        "gate_up_k4": rates["gate_up"].count(4),
        "down_k3": rates["down"].count(3),
        "down_k4": rates["down"].count(4),
    }


def _sha256_extent(stream: object, offset: int, length: int) -> str:
    stream.seek(offset)
    remaining = length
    digest = hashlib.sha256()
    while remaining:
        chunk = stream.read(min(4 * 1024 * 1024, remaining))
        if not chunk:
            raise InvalidPlanError("short read while verifying non-routed tensor")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _verify_zero_extent(stream: object, offset: int, length: int) -> None:
    stream.seek(offset)
    remaining = length
    while remaining:
        chunk = stream.read(min(4096, remaining))
        if not chunk or any(chunk):
            raise InvalidPlanError("non-routed alignment gap is missing or non-zero")
        remaining -= len(chunk)


def _verify_non_routed(root: Path, checkpoint: dict[str, object]) -> dict[str, object]:
    identity = checkpoint.get("non_routed")
    if not isinstance(identity, dict):
        raise InvalidPlanError("Trellis35 non-routed checkpoint identity is invalid")
    artifact = _safe_file(root, identity.get("artifact"), "Trellis35 non-routed artifact")
    manifest_path = _safe_file(root, identity.get("manifest"), "Trellis35 non-routed manifest")
    if artifact.stat().st_size != NON_ROUTED_BYTES or identity.get("artifact_bytes") != NON_ROUTED_BYTES:
        raise InvalidPlanError("Trellis35 non-routed byte extent changed")
    artifact_sha256 = _verify_record_hash(
        artifact, identity, "artifact_sha256", "Trellis35 non-routed artifact"
    )
    manifest_sha256 = sha256_file(manifest_path)
    if identity.get("manifest_sha256") != manifest_sha256:
        raise InvalidPlanError("Trellis35 non-routed manifest SHA-256 mismatch")
    manifest = load_json(manifest_path)
    tensors = manifest.get("tensors")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("checkpoint_profile") != PROFILE
        or manifest.get("status")
        != "wp2_non_routed_import_from_accepted_direct_bf16_derivative"
        or manifest.get("source_lock_sha256") != SOURCE_LOCK
        or manifest.get("excluded_source_roles")
        != ["routed_expert_down", "routed_expert_gate_up"]
        or manifest.get("tensor_count") != 1045
        or manifest.get("bytes") != NON_ROUTED_BYTES
        or manifest.get("sha256") != artifact_sha256
        or not isinstance(tensors, list)
        or len(tensors) != 1045
        or identity.get("tensor_count") != 1045
    ):
        raise InvalidPlanError("Trellis35 non-routed manifest contract is invalid")
    names: set[str] = set()
    cursor = 0
    with artifact.open("rb", buffering=0) as stream:
        for tensor in tensors:
            if not isinstance(tensor, dict):
                raise InvalidPlanError("Trellis35 non-routed tensor record is invalid")
            name, role = tensor.get("name"), tensor.get("role")
            offset, size = tensor.get("destination_offset"), tensor.get("bytes")
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(role, str)
                or role.startswith("routed_expert_")
                or isinstance(offset, bool)
                or not isinstance(offset, int)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or offset != (cursor + ALIGNMENT - 1) & -ALIGNMENT
                or not _hex_sha256(tensor.get("sha256"))
            ):
                raise InvalidPlanError("Trellis35 non-routed tensor identity or extent is invalid")
            _verify_zero_extent(stream, cursor, offset - cursor)
            if _sha256_extent(stream, offset, size) != tensor["sha256"]:
                raise InvalidPlanError(f"Trellis35 non-routed tensor {name} SHA-256 mismatch")
            names.add(name)
            cursor = offset + size
        final = (cursor + ALIGNMENT - 1) & -ALIGNMENT
        _verify_zero_extent(stream, cursor, final - cursor)
    if final != NON_ROUTED_BYTES:
        raise InvalidPlanError("Trellis35 non-routed final extent changed")
    return {
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "tensor_count": len(tensors),
        "bytes": NON_ROUTED_BYTES,
        "routed_expert_tensors": 0,
    }


def main() -> int:
    args = arguments()
    root = args.checkpoint.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise InvalidPlanError("Trellis35 checkpoint must be a real directory")
    checkpoint_path = _safe_file(root, "trellis35-checkpoint.json", "Trellis35 checkpoint manifest")
    experts_path = _safe_file(root, "trellis35-experts.json", "Trellis35 expert index")
    checkpoint = load_json(checkpoint_path)
    experts = load_json(experts_path)
    checkpoint_content_sha256 = _verify_content_hash(checkpoint, "Trellis35 checkpoint")
    experts_content_sha256 = _verify_content_hash(experts, "Trellis35 expert index")
    layers = experts.get("layers")
    routed = checkpoint.get("routed_experts")
    arena = checkpoint.get("arena")
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("format") != "GEM16-Trellis35"
        or checkpoint.get("format_version") != 1
        or checkpoint.get("checkpoint_profile") != PROFILE
        or checkpoint.get("status")
        != "wp7_complete_text_only_runtime_characterized"
        or checkpoint.get("runtime_supported") is not True
        or checkpoint.get("source_lock_sha256") != SOURCE_LOCK
        or experts.get("schema_version") != 1
        or experts.get("format") != "GEM16-Trellis35"
        or experts.get("format_version") != 1
        or experts.get("checkpoint_profile") != PROFILE
        or experts.get("status") != "wp2_complete_30_layer_routed_expert_artifact"
        or experts.get("source_lock_sha256") != SOURCE_LOCK
        or experts.get("layer_count") != LAYERS
        or experts.get("routed_expert_bytes") != ROUTED_BYTES
        or experts.get("payload_bpw_encoded") != 3.5
        or experts.get("gate_up_padding") != "none"
        or experts.get("down_padding") != "input_zero_pad_704_to_768"
        or not isinstance(layers, list)
        or len(layers) != LAYERS
        or not isinstance(routed, dict)
        or routed.get("index") != experts_path.name
        or routed.get("index_sha256") != sha256_file(experts_path)
        or routed.get("checkpoint_content_sha256") != experts_content_sha256
        or routed.get("layer_count") != LAYERS
        or routed.get("bytes") != ROUTED_BYTES
        or routed.get("layers") != layers
        or arena != {
            "alignment_bytes": ALIGNMENT,
            "one_immutable_device_representation": True,
            "nvfp4_routed_expert_bytes": 0,
            "non_routed_bytes": NON_ROUTED_BYTES,
            "trellis35_routed_expert_bytes": ROUTED_BYTES,
            "total_bytes": TOTAL_BYTES,
        }
    ):
        raise InvalidPlanError("Trellis35 checkpoint or expert-index contract is invalid")

    layer_results = []
    for layer, record in enumerate(layers):
        layer_results.append(_verify_layer(root, layer, record))
        print(f"trellis35_checkpoint_verify_layer {layer + 1}/{LAYERS}", flush=True)
    non_routed = _verify_non_routed(root, checkpoint)
    report = {
        "schema_version": 1,
        "work_package": "WP2",
        "status": "complete_30_layer_checkpoint_artifact_pass",
        "checkpoint_profile": PROFILE,
        "runtime_supported": False,
        "checkpoint_manifest_sha256": sha256_file(checkpoint_path),
        "checkpoint_content_sha256": checkpoint_content_sha256,
        "expert_index_sha256": sha256_file(experts_path),
        "expert_index_content_sha256": experts_content_sha256,
        "source_lock_sha256": SOURCE_LOCK,
        "layer_count": len(layer_results),
        "expert_count_per_layer": EXPERTS,
        "payload_bpw_encoded": 3.5,
        "gate_up_padding": "none",
        "down_padding": "input_zero_pad_704_to_768",
        "rate_totals": {
            "gate_up_k3": sum(item["gate_up_k3"] for item in layer_results),
            "gate_up_k4": sum(item["gate_up_k4"] for item in layer_results),
            "down_k3": sum(item["down_k3"] for item in layer_results),
            "down_k4": sum(item["down_k4"] for item in layer_results),
        },
        "non_routed": non_routed,
        "arena": arena,
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(canonical_json_bytes(report))
    print(
        f"trellis35_checkpoint_verify_ok layers={LAYERS} bytes={TOTAL_BYTES} "
        f"sha256={checkpoint_content_sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CompilerError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"trellis35_checkpoint_verify_error: {error}", file=sys.stderr)
        raise SystemExit(getattr(error, "exit_code", 1))
