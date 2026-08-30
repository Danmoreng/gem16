#!/usr/bin/env python3
"""Package non-routed M08 weights and 30 Trellis35 layers as one profile.

The accepted M08 device image is itself a pinned direct derivative of the same
Google BF16 source lock. This packager copies only its non-routed tensors into
a compact image; no NVFP4 routed-expert byte is retained in the new artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

try:
    from tools.gem16_compile.common import InvalidPlanError, canonical_json_bytes
    from tools.gem16_compile.trellis35_artifact import sha256_file
except ModuleNotFoundError:  # Direct execution with tools/ as sys.path[0].
    from gem16_compile.common import InvalidPlanError, canonical_json_bytes
    from gem16_compile.trellis35_artifact import sha256_file


PROFILE = "gem16-trellis35-w4a8-v1"
SOURCE_LOCK = "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230"
M08_ARTIFACT = "471805f7dad8abb84300be78b2822a63dcb1d35bff5aa98426a162cc8532ee17"
M08_IMAGE_SHA256 = "1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72"
M08_IMAGE_BYTES = 14_696_668_160
NON_ROUTED_BYTES = 1_850_270_720
EXPERT_LAYER_BYTES = 345_147_392
LAYERS = 30
ALIGNMENT = 256


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def regular(path: Path, description: str) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise InvalidPlanError(f"{description} must be a regular non-symlink file")
    return metadata


def load_base(base: Path) -> tuple[dict[str, object], Path]:
    compilation_path = base / "gem16_compilation.json"
    lock_path = base / "gem16.lock.json"
    image = base / "model.gem16"
    for path, description in (
        (compilation_path, "M08 compilation"), (lock_path, "M08 lock"),
        (image, "M08 device image"),
    ):
        regular(path, description)
    compilation = json.loads(compilation_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        compilation.get("artifact_profile") != "sm120-text-hybrid-v1"
        or compilation.get("source", {}).get("lock_sha256") != SOURCE_LOCK
        or lock.get("artifact_content_sha256") != M08_ARTIFACT
        or lock.get("source_lock_sha256") != SOURCE_LOCK
        or image.stat().st_size != M08_IMAGE_BYTES
        or sha256_file(image) != M08_IMAGE_SHA256
    ):
        raise InvalidPlanError("base model is not the accepted source-bound M08 image")
    return compilation, image


def tensor_plans(
    compilation: dict[str, object],
    *,
    expected_tensor_count: int = 1285,
    expected_source_bytes: int = M08_IMAGE_BYTES,
    expected_selected_count: int = 1045,
    expected_non_routed_bytes: int = NON_ROUTED_BYTES,
) -> tuple[list[dict[str, object]], int]:
    records = compilation.get("tensors")
    if not isinstance(records, list) or len(records) != expected_tensor_count:
        raise InvalidPlanError("M08 tensor inventory is incomplete")
    source_cursor = 0
    destination_cursor = 0
    selected: list[dict[str, object]] = []
    names: set[str] = set()
    for raw in sorted(records, key=lambda item: item.get("output_name", "")):
        if not isinstance(raw, dict):
            raise InvalidPlanError("M08 tensor record is invalid")
        name, role, size = raw.get("output_name"), raw.get("role"), raw.get("byte_length")
        if (
            not isinstance(name, str) or not name or name in names
            or not isinstance(role, str) or not role
            or isinstance(size, bool) or not isinstance(size, int) or size <= 0
        ):
            raise InvalidPlanError("M08 tensor identity or extent is invalid")
        names.add(name)
        source_cursor = align(source_cursor)
        source_offset = source_cursor
        source_cursor += size
        if role.startswith("routed_expert_"):
            continue
        destination_cursor = align(destination_cursor)
        selected.append({
            "name": name,
            "role": role,
            "runtime_layout": raw.get("runtime_layout"),
            "storage_dtype": raw.get("output_dtype"),
            "physical_shape": raw.get("physical_shape"),
            "logical_shape": raw.get("logical_shape"),
            "source_image_offset": source_offset,
            "destination_offset": destination_cursor,
            "bytes": size,
        })
        destination_cursor += size
    if align(source_cursor) != expected_source_bytes or len(selected) != expected_selected_count:
        raise InvalidPlanError("M08 device-image ordering contract changed")
    if align(destination_cursor) != expected_non_routed_bytes:
        raise InvalidPlanError("Trellis35 non-routed compact extent changed")
    return selected, align(destination_cursor)


def write_non_routed(
    image: Path, output: Path, plans: list[dict[str, object]], bytes_: int
) -> list[dict[str, object]]:
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
        raise InvalidPlanError("refusing to overwrite a non-routed checkpoint image")
    records: list[dict[str, object]] = []
    with image.open("rb", buffering=0) as source, partial.open("xb", buffering=0) as target:
        target.truncate(bytes_)
        for index, plan in enumerate(plans, 1):
            source.seek(plan["source_image_offset"])
            target.seek(plan["destination_offset"])
            remaining = plan["bytes"]
            digest = hashlib.sha256()
            while remaining:
                chunk = source.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise InvalidPlanError("short M08 image read while compacting non-routed weights")
                target.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            records.append({**plan, "sha256": digest.hexdigest()})
            if index % 100 == 0 or index == len(plans):
                print(f"trellis35_non_routed {index}/{len(plans)}", flush=True)
        target.flush()
        os.fsync(target.fileno())
    os.replace(partial, output)
    return records


def existing_non_routed(output: Path, manifest: Path) -> list[dict[str, object]] | None:
    if not output.exists() and not manifest.exists():
        return None
    if not output.is_file() or output.is_symlink() or not manifest.is_file() or manifest.is_symlink():
        raise InvalidPlanError("partial or unsafe existing non-routed artifact")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        metadata.get("bytes") != NON_ROUTED_BYTES
        or metadata.get("sha256") != sha256_file(output)
        or metadata.get("source_image_sha256") != M08_IMAGE_SHA256
        or not isinstance(metadata.get("tensors"), list)
        or len(metadata["tensors"]) != 1045
    ):
        raise InvalidPlanError("existing non-routed artifact failed identity validation")
    return metadata["tensors"]


def validate_experts(root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    index_path = root / "trellis35-experts.json"
    regular(index_path, "Trellis35 expert checkpoint index")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        index.get("checkpoint_profile") != PROFILE
        or index.get("layer_count") != LAYERS
        or index.get("routed_expert_bytes") != LAYERS * EXPERT_LAYER_BYTES
        or not isinstance(index.get("layers"), list)
        or len(index["layers"]) != LAYERS
    ):
        raise InvalidPlanError("Trellis35 expert checkpoint index is incomplete")
    layers = []
    for layer, record in enumerate(index["layers"]):
        if not isinstance(record, dict) or record.get("layer") != layer:
            raise InvalidPlanError("Trellis35 layer ordering is invalid")
        artifact = root / record["artifact"]
        manifest = root / record["manifest"]
        verification = root / record["verification"]
        for path, field in (
            (artifact, "artifact_sha256"), (manifest, "manifest_sha256"),
            (verification, "verification_sha256"),
        ):
            regular(path, f"Trellis35 layer {layer} file")
            if sha256_file(path) != record[field]:
                raise InvalidPlanError(f"Trellis35 layer {layer} hash mismatch")
        layers.append(dict(record))
    return index, layers


def main() -> int:
    args = arguments()
    base = args.base_model.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if not base.is_dir() or base.is_symlink() or not checkpoint.is_dir() or checkpoint.is_symlink():
        raise InvalidPlanError("base and checkpoint must be real directories")
    compilation, image = load_base(base)
    plans, non_routed_bytes = tensor_plans(compilation)
    expert_index, layers = validate_experts(checkpoint)
    non_routed = checkpoint / "non-routed.gem16"
    non_routed_manifest = checkpoint / "non-routed.json"
    tensors = existing_non_routed(non_routed, non_routed_manifest)
    if tensors is None:
        tensors = write_non_routed(image, non_routed, plans, non_routed_bytes)
        document = {
            "schema_version": 1,
            "checkpoint_profile": PROFILE,
            "status": "wp2_non_routed_import_from_accepted_direct_bf16_derivative",
            "source_lock_sha256": SOURCE_LOCK,
            "source_artifact_content_sha256": M08_ARTIFACT,
            "source_image_sha256": M08_IMAGE_SHA256,
            "source_image_bytes": M08_IMAGE_BYTES,
            "excluded_source_roles": ["routed_expert_down", "routed_expert_gate_up"],
            "tensor_count": len(tensors),
            "bytes": non_routed.stat().st_size,
            "sha256": sha256_file(non_routed),
            "tensors": tensors,
        }
        non_routed_manifest.write_bytes(canonical_json_bytes(document))
    total = NON_ROUTED_BYTES + LAYERS * EXPERT_LAYER_BYTES
    content = {
        "schema_version": 1,
        "format": "GEM16-Trellis35",
        "format_version": 1,
        "checkpoint_profile": PROFILE,
        "status": "wp2_complete_text_only_checkpoint_artifact_kernel_not_implemented",
        "runtime_supported": False,
        "source_lock_sha256": SOURCE_LOCK,
        "source_repository": "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
        "source_revision": "f1e06dc520982d9b9edd76859fdb7ab209449949",
        "non_routed": {
            "artifact": non_routed.name,
            "artifact_bytes": non_routed.stat().st_size,
            "artifact_sha256": sha256_file(non_routed),
            "manifest": non_routed_manifest.name,
            "manifest_sha256": sha256_file(non_routed_manifest),
            "tensor_count": len(tensors),
        },
        "routed_experts": {
            "index": "trellis35-experts.json",
            "index_sha256": sha256_file(checkpoint / "trellis35-experts.json"),
            "checkpoint_content_sha256": expert_index["checkpoint_content_sha256"],
            "layer_count": LAYERS,
            "bytes": LAYERS * EXPERT_LAYER_BYTES,
            "layers": layers,
        },
        "arena": {
            "alignment_bytes": ALIGNMENT,
            "one_immutable_device_representation": True,
            "nvfp4_routed_expert_bytes": 0,
            "non_routed_bytes": NON_ROUTED_BYTES,
            "trellis35_routed_expert_bytes": LAYERS * EXPERT_LAYER_BYTES,
            "total_bytes": total,
        },
    }
    content["checkpoint_content_sha256"] = hashlib.sha256(
        canonical_json_bytes(content)
    ).hexdigest()
    final_manifest = checkpoint / "trellis35-checkpoint.json"
    final_manifest.write_bytes(canonical_json_bytes(content))
    print(
        f"trellis35_package_ok bytes={total} sha256={content['checkpoint_content_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InvalidPlanError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"trellis35_package_error: {error}", file=sys.stderr)
        raise SystemExit(getattr(error, "exit_code", 1))
