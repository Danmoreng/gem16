#!/usr/bin/env python3
"""Strict structural and fused-GateUp verification for one Trellis35 layer."""

from __future__ import annotations

import argparse
import mmap
from pathlib import Path
import struct
import sys

import numpy as np

try:
    from tools.gem16_compile.common import (
        BoundedWorkspace,
        CompilerError,
        InvalidPlanError,
        canonical_json_bytes,
        load_json,
    )
    from tools.gem16_compile.reader import read_source_tensors, verify_source_lock
    from tools.gem16_compile.trellis35 import read_source_expert
    from tools.gem16_compile.trellis35_artifact import decode_payload_matrix, sha256_file
    from tools.gem16_compile.trellis35_layout import align_up, validate_rate_map
    from tools.gem16_compile.trellis35_quant import (
        blockwise_hadamard_right,
        gelu_tanh_product,
        inverse_gate_up_output,
        reconstruct_matrix,
    )
except ModuleNotFoundError:  # Direct execution with tools/ as sys.path[0].
    from gem16_compile.common import (
        BoundedWorkspace,
        CompilerError,
        InvalidPlanError,
        canonical_json_bytes,
        load_json,
    )
    from gem16_compile.reader import read_source_tensors, verify_source_lock
    from gem16_compile.trellis35 import read_source_expert
    from gem16_compile.trellis35_artifact import decode_payload_matrix, sha256_file
    from gem16_compile.trellis35_layout import align_up, validate_rate_map
    from gem16_compile.trellis35_quant import (
        blockwise_hadamard_right,
        gelu_tanh_product,
        inverse_gate_up_output,
        reconstruct_matrix,
    )


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "models/gemma4-26b-qat-bf16.lock.json"
ALIGNMENT = 256
REGION_ORDER = (
    "gate_up_k3_payload_pool", "gate_up_k4_payload_pool", "gate_up_descriptor",
    "gate_up_suh", "gate_up_svh", "down_k3_payload_pool", "down_k4_payload_pool",
    "down_descriptor", "down_suh", "down_svh",
)
SHAPES = {
    "gate_up": (2816, 1408, 2816, 1408),
    "down": (768, 2816, 768, 2816),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def expected_region_bytes(name: str) -> int:
    family = "gate_up" if name.startswith("gate_up") else "down"
    rows, columns, suh, svh = SHAPES[family]
    if "k3_payload" in name:
        return rows * columns * 3 // 8 * 64
    if "k4_payload" in name:
        return rows * columns * 4 // 8 * 64
    if name.endswith("descriptor"):
        return 128 * 8
    if name.endswith("suh"):
        return 128 * suh * 2
    if name.endswith("svh"):
        return 128 * svh * 2
    raise AssertionError(name)


def region_layout(manifest: dict[str, object], artifact_bytes: int) -> dict[str, dict[str, int]]:
    regions = manifest.get("regions")
    if not isinstance(regions, dict) or set(regions) != set(REGION_ORDER):
        raise InvalidPlanError("Trellis35 manifest regions are incomplete")
    position = 0
    result: dict[str, dict[str, int]] = {}
    for name in REGION_ORDER:
        raw = regions[name]
        if not isinstance(raw, dict) or set(raw) != {"offset", "bytes"}:
            raise InvalidPlanError(f"Trellis35 region {name} is malformed")
        offset, size = raw["offset"], raw["bytes"]
        if (
            isinstance(offset, bool) or not isinstance(offset, int)
            or isinstance(size, bool) or not isinstance(size, int)
        ):
            raise InvalidPlanError(f"Trellis35 region {name} extent is not integral")
        position = align_up(position, ALIGNMENT, f"Trellis35 {name}")
        if offset != position or size != expected_region_bytes(name):
            raise InvalidPlanError(f"Trellis35 region {name} violates the v1 layout")
        position += size
        result[name] = {"offset": offset, "bytes": size}
    if align_up(position, ALIGNMENT, "Trellis35 layer extent") != artifact_bytes:
        raise InvalidPlanError("Trellis35 artifact byte extent is inconsistent")
    return result


def main() -> int:
    args = arguments()
    manifest = load_json(args.manifest.resolve(strict=True))
    artifact = args.artifact.resolve(strict=True)
    layer = manifest.get("layer")
    if (
        manifest.get("format") != "GEM16-Trellis35"
        or manifest.get("format_version") != 1
        or isinstance(layer, bool)
        or not isinstance(layer, int)
        or not 0 <= layer < 30
        or manifest.get("gate_up_boundary") != 704
        or manifest.get("gate_up_inverse_before_split") is not True
    ):
        raise InvalidPlanError("Trellis35 layer manifest contract is invalid")
    artifact_meta = manifest.get("artifact")
    if not isinstance(artifact_meta, dict) or artifact_meta.get("path") != artifact.name:
        raise InvalidPlanError("Trellis35 artifact identity is invalid")
    artifact_bytes = artifact.stat().st_size
    if artifact_meta.get("bytes") != artifact_bytes or artifact_meta.get("sha256") != sha256_file(artifact):
        raise InvalidPlanError("Trellis35 artifact size or SHA-256 is invalid")
    regions = region_layout(manifest, artifact_bytes)
    rate_maps = manifest.get("rate_maps")
    if not isinstance(rate_maps, dict) or set(rate_maps) != set(SHAPES):
        raise InvalidPlanError("Trellis35 rate maps are invalid")
    rates = {family: validate_rate_map(rate_maps[family]) for family in SHAPES}

    descriptor_checks: dict[str, object] = {}
    numerical: list[dict[str, object]] = []
    workspace = BoundedWorkspace(8 * 1024**3, 1024 * 1024)
    source = verify_source_lock(LOCK, args.source_root, workspace)
    if manifest.get("source_lock_sha256") != source.lock_sha256:
        raise InvalidPlanError("Trellis35 source-lock identity is invalid")
    tensors = read_source_tensors(source, workspace)
    gate_tensor_name = f"model.language_model.layers.{layer}.experts.gate_up_proj"
    if gate_tensor_name not in tensors:
        raise InvalidPlanError("Trellis35 source lacks the manifest layer Gate+Up tensor")
    gate_tensor = tensors[gate_tensor_name]
    rng = np.random.default_rng(20260829 + layer)
    with artifact.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for family, (rows, columns, suh_elements, svh_elements) in SHAPES.items():
            descriptor_region = regions[f"{family}_descriptor"]
            next_offset = {3: 0, 4: 0}
            descriptors = []
            for expert in range(128):
                descriptor = struct.unpack_from(
                    "<IHH", data, descriptor_region["offset"] + expert * 8
                )
                offset, rate, codebook = descriptor
                if rate != rates[family][expert] or codebook != 2 or offset != next_offset[rate]:
                    raise InvalidPlanError(f"Trellis35 {family} descriptor {expert} is invalid")
                next_offset[rate] += rows * columns * rate // 8
                descriptors.append(descriptor)
            for rate in (3, 4):
                if next_offset[rate] != regions[f"{family}_k{rate}_payload_pool"]["bytes"]:
                    raise InvalidPlanError(f"Trellis35 {family} K{rate} pool coverage is invalid")
            for sidecar, elements in (("suh", suh_elements), ("svh", svh_elements)):
                region = regions[f"{family}_{sidecar}"]
                values = np.frombuffer(
                    data, dtype="<f2", count=128 * elements, offset=region["offset"]
                )
                if not np.isfinite(values).all():
                    raise InvalidPlanError(f"Trellis35 {family} {sidecar} is non-finite")
                del values
            descriptor_checks[family] = {
                "descriptors": len(descriptors),
                "k3": rates[family].count(3),
                "k4": rates[family].count(4),
                "pool_coverage": "exact",
            }

        for rate in (3, 4):
            expert = rates["gate_up"].index(rate)
            descriptor_region = regions["gate_up_descriptor"]
            offset, actual_rate, codebook = struct.unpack_from(
                "<IHH", data, descriptor_region["offset"] + expert * 8
            )
            payload_region = regions[f"gate_up_k{rate}_payload_pool"]
            payload_bytes = 2816 * 1408 * rate // 8
            payload = bytes(data[
                payload_region["offset"] + offset:
                payload_region["offset"] + offset + payload_bytes
            ])
            q = decode_payload_matrix(payload, actual_rate, 2816, 1408).astype(np.float64)
            sidecars = {}
            for name, elements in (("suh", 2816), ("svh", 1408)):
                region = regions[f"gate_up_{name}"]
                sidecars[name] = np.frombuffer(
                    data, dtype="<f2", count=elements,
                    offset=region["offset"] + expert * elements * 2,
                ).astype(np.float64)
            activation = rng.normal(0.0, 0.07, 2816)
            transformed_activation = blockwise_hadamard_right(
                (activation * sidecars["suh"]).reshape(1, -1)
            ).reshape(-1)
            transformed_output = transformed_activation @ q
            gate, up = inverse_gate_up_output(transformed_output, sidecars["svh"])
            fused = gelu_tanh_product(gate, up)
            explicit_output = activation @ reconstruct_matrix(
                q, sidecars["suh"], sidecars["svh"]
            )
            explicit_fused = gelu_tanh_product(explicit_output[:704], explicit_output[704:])
            original_output = activation @ read_source_expert(
                gate_tensor, expert, "gate_up"
            ).astype(np.float64)
            full_error = float(np.max(np.abs(np.concatenate((gate, up)) - explicit_output)))
            fused_error = float(np.max(np.abs(fused - explicit_fused)))
            if full_error >= 2e-12 or fused_error >= 2e-12:
                raise InvalidPlanError("Trellis35 fused Gate+Up inverse-before-split check failed")
            numerical.append({
                "rate_bits": rate,
                "expert": expert,
                "codebook_id": codebook,
                "full_output_semantic_max_abs": full_error,
                "gelu_product_semantic_max_abs": fused_error,
                "quantized_output_nmse": float(
                    np.sum(np.square(explicit_output - original_output))
                    / np.sum(np.square(original_output))
                ),
            })

    report = {
        "schema_version": 1,
        "work_package": "WP2",
        "status": "complete_single_layer_artifact_pass",
        "layer": layer,
        "artifact": artifact_meta,
        "manifest_sha256": sha256_file(args.manifest),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "descriptor_checks": descriptor_checks,
        "payload_bpw": 3.5,
        "gate_up_padding": "none",
        "down_padding": "704_to_768",
        "gate_up_inverse_before_split": True,
        "numerical_checks": numerical,
        "source_lock_sha256": source.lock_sha256,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(canonical_json_bytes(report))
    print(f"trellis35_layer_verify_ok artifact_sha256={artifact_meta['sha256']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CompilerError, OSError, ValueError) as error:
        print(f"trellis35_layer_verify_error: {error}", file=sys.stderr)
        raise SystemExit(getattr(error, "exit_code", 1))
