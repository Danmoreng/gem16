#!/usr/bin/env python3
"""Bounded M06 Ordinary-BF16 versus Unsloth-NVFP4 convention diagnostic.

This is deliberately a diagnostic, not an Ordinary conversion or a quality
runner.  Large arithmetic is delegated to gem16-checkpoint-compiler in
``scope=fixture``.  The native fixture job converts the 20 unique full parent
sources selected by the frozen sample.  The small control-plane dequantizer
then walks only the 48 logical ranges with fixed-size read buffers so that the
report can compare both reconstructions without loading a tensor, expert, or
model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time
from typing import Any, Iterable

try:
    from tools.gem16_compile.common import (
        BoundedWorkspace, CompilerError, DataError, InvalidPlanError,
        SourceVerificationError, canonical_json_bytes, git_compiler_identity, load_json,
    )
    from tools.gem16_compile.reader import TensorDescriptor, read_source_tensors, verify_source_lock
    from tools.gem16_compile.native_nvfp4 import (
        NativeNvfp4Request,
        _query_build_info,
        _stage_executable,
        preflight_native_nvfp4,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation from tools/
    from gem16_compile.common import (
        BoundedWorkspace, CompilerError, DataError, InvalidPlanError,
        SourceVerificationError, canonical_json_bytes, git_compiler_identity, load_json,
    )
    from gem16_compile.reader import TensorDescriptor, read_source_tensors, verify_source_lock
    from gem16_compile.native_nvfp4 import (
        NativeNvfp4Request,
        _query_build_info,
        _stage_executable,
        preflight_native_nvfp4,
    )

SCHEMA_VERSION = 1
PROTOCOL = "gem16-nvfp4-direct-v1"
CONTRACT_ID = "gem16.nvfp4_bf16_group16"
PROFILE = "nvfp4-experts-partial-v1"
MAX_RECORDS = 48
DEFAULT_CHUNK_BYTES = 64 * 1024
DEFAULT_MEMORY_BYTES = 1024 * 1024 * 1024
# This is intentionally a convention screen, not a model-quality tolerance.
ACCEPTANCE = {
    "relative_l2_max": 0.25,
    "cosine_min": 0.95,
    "sqnr_min": 10.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _hex(value: Any, where: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise DataError(f"{where} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, where: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataError(f"{where} must be a positive integer")
    if maximum is not None and value > maximum:
        raise DataError(f"{where} exceeds bound {maximum}")
    return value


def _finite(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DataError(f"{where} is not finite")
    result = float(value)
    if positive and result <= 0.0:
        raise DataError(f"{where} is not positive")
    return result


def _regular(path: Path, where: str, *, executable: bool = False) -> Path:
    path = path.expanduser().absolute()
    try:
        st = path.lstat()
    except OSError as exc:
        raise SourceVerificationError(f"{where} is unavailable: {path}: {exc}") from exc
    if not path.is_file() or path.is_symlink():
        raise SourceVerificationError(f"{where} must be a non-symlink regular file: {path}")
    if executable and not os.access(path, os.X_OK, follow_symlinks=False):
        raise InvalidPlanError(f"native encoder is not executable: {path}")
    return path


def _resolve_reference(path_value: str, config_path: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    # Config paths are repository-relative in the locked artifact.  The second
    # candidate makes small self-contained fixture configs convenient.
    root = Path(__file__).resolve().parents[1]
    first = root / candidate
    return first if first.exists() else config_path.parent / candidate


def _inventory_records(inventory: dict[str, Any], where: str) -> list[dict[str, Any]]:
    records = inventory.get("tensors")
    if not isinstance(records, list) or not records:
        raise DataError(f"{where}.tensors must be a nonempty array")
    names: set[str] = set()
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise DataError(f"{where}.tensors[{index}] is not an object")
        required = {"name", "dtype", "shape", "bytes", "shard", "absolute_offset", "aliases"}
        if set(item) != required:
            raise DataError(f"{where}.tensors[{index}] has an unexpected schema")
        name = item["name"]
        if not isinstance(name, str) or not name or name in names:
            raise DataError(f"{where}.tensors[{index}] has a duplicate/invalid name")
        names.add(name)
        if not isinstance(item["dtype"], str) or not isinstance(item["shape"], list):
            raise DataError(f"{where}.tensors[{index}] has invalid dtype/shape")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in item["shape"]):
            raise DataError(f"{where}.tensors[{index}] has invalid shape")
        _positive_int(item["bytes"], f"{where}.tensors[{index}].bytes")
        if isinstance(item["absolute_offset"], bool) or not isinstance(item["absolute_offset"], int) or item["absolute_offset"] < 0:
            raise DataError(f"{where}.tensors[{index}].absolute_offset is invalid")
        if not isinstance(item["shard"], str) or not item["shard"] or not isinstance(item["aliases"], list):
            raise DataError(f"{where}.tensors[{index}] has invalid shard/aliases")
    return records


def _verify_inventory(inventory_path: Path, inventory: dict[str, Any], source: Any,
                      descriptors: dict[str, TensorDescriptor], where: str) -> None:
    entries = {str(item["name"]): item for item in _inventory_records(inventory, where)}
    if set(entries) != set(descriptors):
        raise DataError(f"{where} inventory names do not match Safetensors headers")
    for name, descriptor in descriptors.items():
        item = entries[name]
        if item["dtype"] != descriptor.dtype or tuple(item["shape"]) != descriptor.shape:
            raise DataError(f"{where} inventory dtype/shape mismatch: {name}")
        if item["bytes"] != descriptor.byte_length or item["absolute_offset"] != descriptor.absolute_offset:
            raise DataError(f"{where} inventory range mismatch: {name}")
        if item["shard"] != descriptor.shard:
            raise DataError(f"{where} inventory shard mismatch: {name}")
        if any(not isinstance(alias, str) for alias in item["aliases"]):
            raise DataError(f"{where} inventory aliases are invalid: {name}")
    # Source verification has checked each locked file.  Reject final symlinks
    # as well: the shared reader intentionally permits some legacy views.
    for locked in source.files.values():
        if locked.path.is_symlink() or not locked.path.is_file():
            raise SourceVerificationError(f"{where} locked file is a symlink/non-file: {locked.relative_path}")
    if inventory_path.is_symlink():
        raise SourceVerificationError(f"inventory is a symlink: {inventory_path}")


def _source_identity(config: dict[str, Any], inventory: dict[str, Any], kind: str) -> dict[str, str]:
    source = inventory.get("source")
    ref = config["diagnostic_references"][kind]
    if not isinstance(source, dict) or not isinstance(ref, dict):
        raise DataError(f"missing {kind} source identity")
    for field in ("repository", "revision", "lock_sha256"):
        actual = source.get(field)
        expected = ref.get("sha256") if field == "lock_sha256" else ref.get(field)
        if actual != expected:
            raise SourceVerificationError(f"{kind} {field} disagrees with frozen config")
    return {"repository": str(source["repository"]), "revision": str(source["revision"]),
            "lock_sha256": _hex(source["lock_sha256"], f"{kind}.lock_sha256")}


def _load_source(root: Path, inventory_path: Path, lock_path: Path, config: dict[str, Any],
                 inventory: dict[str, Any], kind: str, workspace: BoundedWorkspace) -> tuple[Any, dict[str, TensorDescriptor], dict[str, str]]:
    if root.is_symlink():
        raise SourceVerificationError(f"{kind} checkpoint root is a symlink")
    identity = _source_identity(config, inventory, kind)
    _regular(lock_path, f"{kind} source lock")
    if _sha256(lock_path) != identity["lock_sha256"]:
        raise SourceVerificationError(f"{kind} source lock hash mismatch")
    source = verify_source_lock(lock_path, root, workspace)
    if source.repository != identity["repository"] or source.revision != identity["revision"] or source.lock_sha256 != identity["lock_sha256"]:
        raise SourceVerificationError(f"{kind} verified lock identity mismatch")
    descriptors = read_source_tensors(source, workspace)
    _verify_inventory(inventory_path, inventory, source, descriptors, kind)
    return source, descriptors, identity


def _shape_product(shape: Iterable[int]) -> int:
    product = 1
    for dimension in shape:
        product *= dimension
    return product


def _descriptor_bytes(item: dict[str, Any], where: str) -> int:
    shape = tuple(int(x) for x in item["shape"])
    width = {"BF16": 2, "U8": 1, "F8_E4M3": 1, "F32": 4}.get(str(item["dtype"]))
    if width is None or _shape_product(shape) * width != item["bytes"]:
        raise DataError(f"{where} dtype/byte count mismatch")
    return int(item["bytes"])


def _validate_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("schema_version") != 1 or config.get("milestone") != "M06":
        raise DataError("unsupported M06 compiler config")
    contract = config.get("quantizer", {}).get("contract", {})
    if contract.get("group_size") != 16 or contract.get("packed_dtype") != "U8" or contract.get("global_scale_role") != "divisor":
        raise DataError("ambiguous or unsupported NVFP4 divisor contract")
    if contract.get("local_scale_dtype") != "F8_E4M3" or contract.get("value_codec") != "E2M1":
        raise DataError("unsupported NVFP4 codec contract")
    sample = config.get("diagnostic_sample", {})
    records = sample.get("records")
    if sample.get("logical_matrix_count") != MAX_RECORDS or not isinstance(records, list) or len(records) != MAX_RECORDS:
        raise DataError("frozen M06 sample must contain exactly 48 records")
    layers = {0, 5, 24, 29}; projections = {"gate", "up", "down"}
    identities: set[tuple[Any, ...]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise DataError(f"diagnostic record {index} is not an object")
        for field in ("kind", "layer", "projection", "source_tensor", "unsloth_component", "source_slice", "unsloth_component_range"):
            if field not in record:
                raise DataError(f"diagnostic record {index} lacks {field}")
        if record["layer"] not in layers or record["projection"] not in projections or record["kind"] not in {"shared", "routed"}:
            raise DataError(f"invalid frozen diagnostic identity at {index}")
        key = (record["kind"], record["layer"], record["projection"], record.get("expert"))
        if key in identities:
            raise DataError(f"duplicate frozen diagnostic identity: {key}")
        identities.add(key)
        source_slice = record["source_slice"]
        if not isinstance(source_slice, dict) or source_slice.get("axis") not in {"none", "expert", "expert,projection"}:
            raise DataError(f"invalid source slice at {index}")
        if not isinstance(source_slice.get("start"), list) or not isinstance(source_slice.get("stop"), list):
            raise DataError(f"invalid source slice bounds at {index}")
        if len(source_slice["start"]) != len(source_slice["stop"]) or any(
            isinstance(x, bool) or not isinstance(x, int) or x < 0
            for x in source_slice["start"] + source_slice["stop"]):
            raise DataError(f"invalid source slice integers at {index}")
        component_range = record["unsloth_component_range"]
        if not isinstance(component_range, dict) or component_range.get("range") != "full_component":
            raise DataError(f"ambiguous Unsloth component range at {index}")
    expected = {(kind, layer, projection, None if kind == "shared" else expert)
                for layer in layers for projection in projections for kind, expert in (("shared", None), ("routed", 0), ("routed", 63), ("routed", 127))}
    if identities != expected:
        raise DataError("frozen sample does not cover shared and selected routed matrices")
    _thresholds(config)
    return records


def _descriptor_from_inventory(inventory: dict[str, Any], name: str) -> dict[str, Any]:
    for item in inventory["tensors"]:
        if item["name"] == name:
            return item
    raise DataError(f"missing tensor in inventory: {name}")


def _record_geometry(record: dict[str, Any], ordinary: dict[str, TensorDescriptor], unsloth_inventory: dict[str, Any]) -> dict[str, Any]:
    source_name = str(record["source_tensor"])
    if source_name not in ordinary:
        raise DataError(f"missing Ordinary source tensor: {source_name}")
    source = ordinary[source_name]
    if source.dtype != "BF16":
        raise DataError(f"Ordinary source is not BF16: {source_name}")
    sl = record["source_slice"]; start, stop = sl["start"], sl["stop"]
    if len(start) != len(source.shape) or len(stop) != len(source.shape) or any(stop[i] <= start[i] or stop[i] > source.shape[i] for i in range(len(start))):
        raise DataError(f"source slice is outside tensor: {source_name}")
    shape = tuple(stop[i] - start[i] for i in range(len(start)))
    if record["kind"] == "shared":
        if sl["axis"] != "none" or tuple(start) != (0, 0) or tuple(stop) != source.shape:
            raise DataError(f"shared source slice is not the complete matrix: {source_name}")
        rows, columns = shape
        source_offset = source.absolute_offset
    else:
        if sl["axis"] not in {"expert", "expert,projection"} or len(shape) != 3 or shape[0] != 1:
            raise DataError(f"routed source slice is not expert-major: {source_name}")
        expert = record.get("expert")
        if expert not in {0, 63, 127} or start[0] != expert or stop[0] != expert + 1:
            raise DataError(f"routed expert split is invalid: {source_name}")
        if record["projection"] in {"gate", "up"}:
            if tuple(source.shape) != (128, 1408, 2816) or shape[1] != 704 or shape[2] != 2816:
                raise DataError(f"gate/up fused split shape is invalid: {source_name}")
            wanted = 0 if record["projection"] == "gate" else 704
            if start[1] != wanted or stop[1] != wanted + 704:
                raise DataError(f"gate/up split order is invalid: {source_name}")
        elif tuple(source.shape) != (128, 2816, 704) or shape[1:] != (2816, 704):
            raise DataError(f"down expert shape is invalid: {source_name}")
        rows, columns = shape[1], shape[2]
        source_offset = source.absolute_offset + ((start[0] * source.shape[1] + start[1]) * source.shape[2]) * 2
    component = str(record["unsloth_component"])
    expected_names = {
        "packed": component,
        "local": component.removesuffix(".weight_packed") + ".weight_scale",
        "weight_global": component.removesuffix(".weight_packed") + ".weight_global_scale",
        "input_global": component.removesuffix(".weight_packed") + ".input_global_scale",
    }
    components: dict[str, dict[str, Any]] = {}
    for role, name in expected_names.items():
        item = _descriptor_from_inventory(unsloth_inventory, name)
        components[role] = item
    if components["packed"]["dtype"] != "U8" or tuple(components["packed"]["shape"]) != (rows, columns // 2):
        raise DataError(f"Unsloth packed shape mismatch: {component}")
    if components["local"]["dtype"] != "F8_E4M3" or tuple(components["local"]["shape"]) != (rows, columns // 16):
        raise DataError(f"Unsloth local-scale shape mismatch: {component}")
    for role in ("weight_global", "input_global"):
        if components[role]["dtype"] != "F32" or tuple(components[role]["shape"]) != (1,):
            raise DataError(f"Unsloth scalar shape mismatch: {component}")
        _descriptor_bytes(components[role], f"Unsloth {role} {component}")
    return {"source": source, "source_offset": source_offset, "rows": rows, "columns": columns,
            "components": components, "component": component, "source_shape": list(source.shape),
            "slice_shape": list(shape)}


def _build_parent_geometries(
    records: list[dict[str, Any]],
    ordinary: dict[str, TensorDescriptor],
    unsloth_inventory: dict[str, Any],
    workspace: BoundedWorkspace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build full-source native jobs, then attach the 48 logical slices.

    The production contract derives one divisor from the complete source tensor.
    In particular, a routed gate/up tensor is one [128, 1408, 2816] source and
    is never replaced by an individual expert or projection slice for this
    diagnostic.  Logical records retain their selected row ranges for bounded
    comparison against Unsloth.
    """
    parents: dict[str, dict[str, Any]] = {}
    logical: list[dict[str, Any]] = []
    for record in records:
        geometry = _record_geometry(record, ordinary, unsloth_inventory)
        source = geometry["source"]
        parent = parents.get(source.name)
        if parent is None:
            shape = tuple(source.shape)
            columns = shape[-1]
            rows = _shape_product(shape[:-1])
            if len(shape) not in (2, 3) or columns <= 0 or columns % 16:
                raise DataError(f"invalid full M06 diagnostic source shape: {source.name}")
            if source.byte_length != rows * columns * 2:
                raise DataError(f"full M06 diagnostic source byte mismatch: {source.name}")
            if source.name.endswith(".experts.gate_up_proj"):
                role = "routed_expert_gate_up"
            elif source.name.endswith(".experts.down_proj"):
                role = "routed_expert_down"
            elif ".mlp.gate_proj.weight" in source.name:
                role = "shared_mlp_gate"
            elif ".mlp.up_proj.weight" in source.name:
                role = "shared_mlp_up"
            elif ".mlp.down_proj.weight" in source.name:
                role = "shared_mlp_down"
            else:
                raise DataError(f"unsupported diagnostic parent source: {source.name}")
            parent = {
                "source": source,
                "source_name": source.name,
                "source_offset": source.absolute_offset,
                "source_bytes": source.byte_length,
                "source_shape": list(shape),
                "rows": rows,
                "columns": columns,
                "role": role,
                "records": [],
                "source_hash": workspace.hash_range(
                    source.path, source.absolute_offset, source.byte_length
                ),
            }
            parents[source.name] = parent
        start = record["source_slice"]["start"]
        if record["kind"] == "shared":
            row_start = 0
            slice_rows = parent["rows"]
        else:
            row_start = start[0] * parent["source_shape"][1] + start[1]
            slice_rows = geometry["rows"]
        geometry.update({
            "record": record,
            "parent": parent,
            "parent_name": parent["source_name"],
            "parent_shape": list(parent["source_shape"]),
            "parent_rows": parent["rows"],
            "parent_columns": parent["columns"],
            "parent_role": parent["role"],
            "parent_row_start": row_start,
            "slice_rows": slice_rows,
            "source_hash": parent["source_hash"],
        })
        parent["records"].append(geometry)
        logical.append(geometry)
    parent_values = sorted(parents.values(), key=lambda item: item["source_name"])
    if len(parent_values) != 20 or len(logical) != MAX_RECORDS:
        raise DataError(
            f"M06 diagnostic parent/sample count mismatch: parents={len(parent_values)} "
            f"logical={len(logical)}"
        )
    return parent_values, logical


class _Cursor:
    """Fixed-buffer sequential reader; no read returns more than chunk_bytes."""
    def __init__(self, path: Path, offset: int, length: int, chunk_bytes: int):
        self.path, self.offset, self.remaining = path, offset, length
        self.buffer = bytearray(chunk_bytes)
        self.view = memoryview(self.buffer)
        self.pos = self.end = 0
        self.stream = path.open("rb", buffering=0)
        self.stream.seek(offset)
    def _byte(self) -> int:
        if self.pos == self.end:
            if self.remaining == 0:
                raise DataError(f"short bounded read from {self.path}")
            wanted = min(len(self.buffer), self.remaining)
            count = self.stream.readinto(self.view[:wanted])
            if count != wanted:
                raise DataError(f"short bounded read from {self.path}")
            self.remaining -= count; self.pos = 0; self.end = count
        value = self.buffer[self.pos]; self.pos += 1
        return value
    def u16(self) -> int:
        return self._byte() | (self._byte() << 8)
    def u32(self) -> int:
        return self._byte() | (self._byte() << 8) | (self._byte() << 16) | (self._byte() << 24)
    def close(self) -> None:
        self.stream.close(); self.view.release()
        if self.remaining or self.pos != self.end:
            raise DataError(f"bounded reader did not consume declared range: {self.path}")


def _bf16(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _e2m1(code: int) -> float:
    value = (0.0, .5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[code & 7]
    return -value if code & 8 else value


def _e4m3(code: int) -> float:
    magnitude = code & 0x7f
    if magnitude == 0x7f:
        return math.nan
    exponent, mantissa = (magnitude >> 3) & 0xf, magnitude & 7
    value = math.ldexp(float(mantissa), -9) if exponent == 0 else math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return -value if code & 0x80 else value


def _metric_accumulator() -> dict[str, Any]:
    return {"elements": 0, "source_sq": 0.0, "ref_sq": 0.0, "diff_sq": 0.0, "dot": 0.0,
            "max_abs": 0.0, "codes": [0] * 16, "scales": [0] * 256}


def _add_metric(acc: dict[str, Any], source: float, reference: float, code: int, scale_code: int) -> None:
    if not math.isfinite(source) or not math.isfinite(reference):
        raise DataError("nonfinite source or dequantized value")
    diff = reference - source
    acc["elements"] += 1; acc["source_sq"] += source * source; acc["ref_sq"] += reference * reference
    acc["diff_sq"] += diff * diff; acc["dot"] += source * reference; acc["max_abs"] = max(acc["max_abs"], abs(diff))
    acc["codes"][code] += 1; acc["scales"][scale_code] += 1


def _finish_metrics(acc: dict[str, Any], label: str) -> dict[str, Any]:
    n = acc["elements"]
    if n <= 0 or not acc["source_sq"] or not acc["ref_sq"]:
        raise DataError(f"{label} has zero-norm metrics")
    relative = math.sqrt(acc["diff_sq"] / acc["source_sq"])
    cosine = acc["dot"] / math.sqrt(acc["source_sq"] * acc["ref_sq"])
    # A perfect finite reconstruction is represented by the declared finite
    # diagnostic cap rather than JSON Infinity.
    sqnr = 1.0e6 if acc["diff_sq"] == 0.0 else 10.0 * math.log10(acc["source_sq"] / acc["diff_sq"])
    values = {"elements": n, "relative_l2": relative, "cosine": cosine,
              "max_absolute_error": acc["max_abs"], "sqnr": sqnr,
              "code_histogram": list(acc["codes"]), "scale_histogram": list(acc["scales"])}
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise DataError(f"{label}.{key} is nonfinite")
    return values


def _decode_component(geometry: dict[str, Any], output: dict[str, dict[str, Any]], chunk_bytes: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows, columns = geometry["rows"], geometry["columns"]
    source = geometry["source"]
    components = geometry["components"]
    # Open only bounded logical ranges.  Parent outputs are shared by all
    # selected experts, but each logical cursor starts at its exact row range.
    source_cursor = _Cursor(source.path, geometry["source_offset"], rows * columns * 2, chunk_bytes)
    compiled_p = _Cursor(output["packed"]["path"], output["packed"]["offset"], rows * (columns // 2), chunk_bytes)
    compiled_s = _Cursor(output["local"]["path"], output["local"]["offset"], rows * (columns // 16), chunk_bytes)
    uns_p = _Cursor(components["packed_path"], components["packed_offset"], components["packed_bytes"], chunk_bytes)
    uns_s = _Cursor(components["local_path"], components["local_offset"], components["local_bytes"], chunk_bytes)
    compiled = _metric_accumulator(); unsloth = _metric_accumulator(); relationship = _metric_accumulator()
    try:
        weight_global = None
        # Scalars are read through the same bounded cursor and validated before
        # any value is accepted.  Output scalar paths are carried separately.
        with open(output["weight_global"]["path"], "rb", buffering=0) as stream:
            stream.seek(output["weight_global"]["offset"]); raw = stream.read(4)
        if len(raw) != 4: raise DataError("short Ordinary weight scalar")
        weight_global = struct.unpack("<f", raw)[0]
        with open(output["input_global"]["path"], "rb", buffering=0) as stream:
            stream.seek(output["input_global"]["offset"]); raw = stream.read(4)
        if len(raw) != 4: raise DataError("short Ordinary input scalar")
        input_global = struct.unpack("<f", raw)[0]
        with open(components["weight_global_path"], "rb", buffering=0) as stream:
            stream.seek(components["weight_global_offset"]); raw = stream.read(4)
        uns_weight_global = struct.unpack("<f", raw)[0] if len(raw) == 4 else math.nan
        with open(components["input_global_path"], "rb", buffering=0) as stream:
            stream.seek(components["input_global_offset"]); raw = stream.read(4)
        uns_input_global = struct.unpack("<f", raw)[0] if len(raw) == 4 else math.nan
        for value, label in ((weight_global, "Ordinary weight divisor"), (input_global, "Ordinary input divisor"),
                             (uns_weight_global, "Unsloth weight divisor"), (uns_input_global, "Unsloth input divisor")):
            _finite(value, label, positive=True)
        for row in range(rows):
            compiled_scale_codes = [compiled_s._byte() for _ in range(columns // 16)]
            unsloth_scale_codes = [uns_s._byte() for _ in range(columns // 16)]
            for block in range(columns // 16):
                if not math.isfinite(_e4m3(compiled_scale_codes[block])) or not math.isfinite(_e4m3(unsloth_scale_codes[block])):
                    raise DataError("nonfinite E4M3 local scale")
            for pair in range(columns // 2):
                compiled_byte = compiled_p._byte(); uns_byte = uns_p._byte()
                for nibble in (0, 1):
                    source_value = _bf16(source_cursor.u16())
                    c_code = (compiled_byte >> (4 * nibble)) & 0xf; u_code = (uns_byte >> (4 * nibble)) & 0xf
                    block = (pair * 2) // 16
                    c_value = _e2m1(c_code) * _e4m3(compiled_scale_codes[block]) / weight_global
                    u_value = _e2m1(u_code) * _e4m3(unsloth_scale_codes[block]) / uns_weight_global
                    _add_metric(compiled, source_value, c_value, c_code, compiled_scale_codes[block])
                    _add_metric(unsloth, source_value, u_value, u_code, unsloth_scale_codes[block])
                    _add_metric(relationship, c_value, u_value, c_code, unsloth_scale_codes[block])
        return _finish_metrics(compiled, "Ordinary compiled"), _finish_metrics(unsloth, "Unsloth"), _finish_metrics(relationship, "compiled versus Unsloth")
    finally:
        for cursor in (source_cursor, compiled_p, compiled_s, uns_p, uns_s):
            cursor.close()


def _reserve(path: Path, size: int) -> None:
    with path.open("w+b") as stream:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(stream.fileno(), 0, size)
        else:
            stream.truncate(size)
        stream.flush()
        os.fsync(stream.fileno())


def _native_job(parents: list[dict[str, Any]], output_dir: Path, threads: int) -> dict[str, Any]:
    operations = []
    for index, parent in enumerate(parents):
        source = parent["source"]
        stem = source.name.removesuffix(".weight")
        columns = parent["columns"]
        elements = parent["rows"] * columns
        full_sizes = {
            "packed": elements // 2,
            "local_scale": elements // 16,
            "weight_global": 4,
            "input_global": 4,
        }
        outputs: dict[str, dict[str, Any]] = {}
        suffixes = {
            "packed": ".weight_packed",
            "local_scale": ".weight_scale",
            "weight_global": ".weight_global_scale",
            "input_global": ".input_global_scale",
        }
        for component, size in full_sizes.items():
            path = output_dir / f"parent-{index:02d}-{component}.bin"
            _reserve(path, size)
            outputs[component] = {
                "component": component,
                "name": stem + suffixes[component],
                "path": str(path.absolute()),
                "offset": 0,
                "bytes": size,
            }
        parent["native_outputs"] = outputs
        for geometry in parent["records"]:
            row_start = geometry["parent_row_start"]
            geometry["output"] = {
                "packed": {"path": Path(outputs["packed"]["path"]), "offset": row_start * (columns // 2)},
                "local": {"path": Path(outputs["local_scale"]["path"]), "offset": row_start * (columns // 16)},
                "weight_global": {"path": Path(outputs["weight_global"]["path"]), "offset": 0},
                "input_global": {"path": Path(outputs["input_global"]["path"]), "offset": 0},
            }
        operations.append({
            "operation_id": "fixture:" + stem,
            "source_name": source.name,
            "source_path": str(source.path.absolute()),
            "source_sha256": parent["source_hash"],
            "source_offset": source.absolute_offset,
            "source_bytes": source.byte_length,
            "source_dtype": source.dtype,
            "logical_shape": list(source.shape),
            "rows": parent["rows"],
            "columns": columns,
            "role": parent["role"],
            "axis_transformation": "identity",
            "disk_layout": "canonical_row_major_low_nibble_first",
            "runtime_layout": "expert_major_sm120_row8_k64" if parent["role"].startswith("routed_") else "sm120_row8_k64",
            "packed": outputs["packed"],
            "local_scale": outputs["local_scale"],
            "weight_global": outputs["weight_global"],
            "input_global": outputs["input_global"],
        })
    return {"schema_version": 1, "protocol": PROTOCOL, "artifact_profile": PROFILE,
            "scope": "fixture", "contract_id": CONTRACT_ID, "contract_version": 1,
            "threads": threads, "operations": operations}


def _run_native(executable: Path, job: dict[str, Any], telemetry: Path, timeout: int) -> dict[str, Any]:
    job_path = telemetry.parent / "native-job.json"
    with job_path.open("x", encoding="utf-8") as stream:
        json.dump(job, stream, indent=2, sort_keys=True); stream.write("\n")
    try:
        completed = subprocess.run([str(executable), "--nvfp4-job", str(job_path), "--telemetry", str(telemetry)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   check=False, timeout=timeout, text=True, env={**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DataError(f"native NVFP4 diagnostic failed to run: {exc}") from exc
    if completed.stdout:
        raise DataError("native NVFP4 compiler wrote unexpected stdout")
    if completed.returncode != 0:
        raise DataError(f"native NVFP4 compiler failed ({completed.returncode}): {completed.stderr[-2000:]}")
    if not telemetry.is_file() or telemetry.is_symlink():
        raise DataError("native NVFP4 compiler did not publish telemetry")
    report = load_json(telemetry, 16 * 1024 * 1024)
    if report.get("schema_version") != 1 or report.get("protocol") != PROTOCOL or report.get("scope") != "fixture":
        raise DataError("native telemetry identity mismatch")
    operations = report.get("operations")
    if not isinstance(operations, list) or len(operations) != len(job["operations"]):
        raise DataError("native telemetry operation count mismatch")
    if len(operations) != 20:
        raise DataError("M06 diagnostic must compile exactly 20 full parent sources")
    wanted = {(x["operation_id"], x["source_sha256"]) for x in job["operations"]}
    got = {(x.get("operation_id"), x.get("source_sha256")) for x in operations}
    if got != wanted or len(got) != len(wanted):
        raise DataError("native telemetry operation identity mismatch")
    return report


def _thresholds(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    thresholds = config.get("diagnostic_sample", {}).get("acceptance", {}).get("thresholds")
    if not isinstance(thresholds, dict):
        raise DataError("M06 diagnostic acceptance thresholds are missing")
    required = ("ordinary_compiled", "unsloth_reference", "compiled_vs_unsloth")
    result: dict[str, dict[str, float]] = {}
    for name in required:
        item = thresholds.get(name)
        if not isinstance(item, dict) or set(item) != {"relative_l2_max", "cosine_min", "sqnr_min"}:
            raise DataError(f"invalid M06 diagnostic thresholds: {name}")
        result[name] = {
            "relative_l2_max": _finite(item["relative_l2_max"], f"{name}.relative_l2_max"),
            "cosine_min": _finite(item["cosine_min"], f"{name}.cosine_min"),
            "sqnr_min": _finite(item["sqnr_min"], f"{name}.sqnr_min"),
        }
        if (result[name]["relative_l2_max"] > ACCEPTANCE["relative_l2_max"] or
                result[name]["cosine_min"] < ACCEPTANCE["cosine_min"] or
                result[name]["sqnr_min"] < ACCEPTANCE["sqnr_min"]):
            raise DataError(f"M06 diagnostic threshold is weaker than the frozen minimum: {name}")
    return result


def _build_report(config: dict[str, Any], config_hash: str, command: list[str], command_hash: str,
                  runner_sha256: str, compiler_identity: dict[str, Any], ordinary_id: dict[str, str],
                  unsloth_id: dict[str, str], parents: list[dict[str, Any]], geometries: list[dict[str, Any]],
                  telemetry: dict[str, Any], chunk_bytes: int, workspace: BoundedWorkspace) -> dict[str, Any]:
    thresholds = _thresholds(config)
    matrices = []; passed = True
    native_ops = {item["operation_id"]: item for item in telemetry["operations"]}
    parent_records = []
    for parent in parents:
        stem = parent["source_name"].removesuffix(".weight")
        native = native_ops["fixture:" + stem]
        parent_records.append({
            "source_tensor": parent["source_name"], "source_shape": parent["source_shape"],
            "source_sha256": parent["source_hash"], "rows": parent["rows"], "columns": parent["columns"],
            "role": parent["role"], "weight_divisor": native["weight_divisor"],
            "input_divisor": native["input_divisor"], "operation_id": "fixture:" + stem,
        })
    parent_divisors: dict[str, tuple[float, float]] = {}
    for geometry in geometries:
        stem = geometry["parent_name"].removesuffix(".weight")
        native = native_ops["fixture:" + stem]
        divisor_pair = (_finite(native["weight_divisor"], "parent weight divisor"),
                        _finite(native["input_divisor"], "parent input divisor"))
        previous = parent_divisors.setdefault(geometry["parent_name"], divisor_pair)
        if previous != divisor_pair:
            raise DataError(f"sampled records disagree on parent divisor: {geometry['parent_name']}")
        compiled, unsloth, relation = _decode_component(geometry, geometry["output"], chunk_bytes)
        for metric in (compiled, unsloth, relation):
            if not all(math.isfinite(float(metric[key])) for key in ("relative_l2", "cosine", "max_absolute_error", "sqnr")):
                raise DataError("nonfinite diagnostic metric")
        def accepts(metric: dict[str, Any], limit: dict[str, float]) -> bool:
            return (metric["relative_l2"] <= limit["relative_l2_max"] and
                    metric["cosine"] >= limit["cosine_min"] and
                    metric["sqnr"] >= limit["sqnr_min"])
        convention_pass = (accepts(compiled, thresholds["ordinary_compiled"]) and
                           accepts(unsloth, thresholds["unsloth_reference"]) and
                           accepts(relation, thresholds["compiled_vs_unsloth"]))
        passed &= convention_pass
        record = geometry["record"]
        components = geometry["components"]
        matrices.append({
            "kind": record["kind"], "layer": record["layer"], "projection": record["projection"],
            "expert": record.get("expert"), "source_tensor": record["source_tensor"],
            "source_dtype": "BF16", "source_shape": geometry["source_shape"],
            "parent_source_tensor": geometry["parent_name"], "parent_source_shape": geometry["parent_shape"],
            "parent_source_sha256": geometry["parent"]["source_hash"],
            "parent_row_start": geometry["parent_row_start"], "parent_rows": geometry["parent_rows"],
            "source_slice": record["source_slice"], "shape": [geometry["rows"], geometry["columns"]],
            "packed_shape": [geometry["rows"], geometry["columns"] // 2],
            "scale_shape": [geometry["rows"], geometry["columns"] // 16],
            "compiled_output_offsets": {key: value["offset"] for key, value in geometry["output"].items()},
            "ordinary_compiled_component": geometry["parent"]["native_outputs"]["packed"]["name"],
            "ordinary_compiled_scale_component": geometry["parent"]["native_outputs"]["local_scale"]["name"],
            "unsloth_component": geometry["component"],
            "ordinary_compiled_to_frozen_contract": {
                "direction": "divisor", "weight_global_scale": native["weight_divisor"],
                "input_global_scale": native["input_divisor"], "metrics": compiled,
            },
            "unsloth_reference": {
                "direction": "divisor", "component": geometry["component"],
                "packed": {"dtype": components["packed"]["dtype"], "shape": components["packed"]["shape"], "offset": components["packed"]["absolute_offset"], "bytes": components["packed"]["bytes"]},
                "local_scale": {"dtype": components["local"]["dtype"], "shape": components["local"]["shape"], "offset": components["local"]["absolute_offset"], "bytes": components["local"]["bytes"]},
                "weight_global_scale": {"dtype": components["weight_global"]["dtype"], "shape": components["weight_global"]["shape"], "offset": components["weight_global"]["absolute_offset"], "bytes": components["weight_global"]["bytes"]},
                "input_global_scale": {"dtype": components["input_global"]["dtype"], "shape": components["input_global"]["shape"], "offset": components["input_global"]["absolute_offset"], "bytes": components["input_global"]["bytes"]},
                "metrics": unsloth,
            },
            "ordinary_compiled_vs_unsloth": {"metrics": relation}, "pass": convention_pass,
        })
    return {
        "schema_version": SCHEMA_VERSION, "milestone": "M06",
        "status": "pass" if passed else "fail", "pass": passed,
        "quality_claim": False, "diagnostic_only": True, "artifact_profile": PROFILE,
        "contract_id": CONTRACT_ID, "identities": {
            "ordinary_bf16": ordinary_id, "unsloth_nvfp4": unsloth_id,
            "config_sha256": config_hash, "native_protocol": PROTOCOL,
            "native_encoder_sha256": telemetry.get("native_encoder_sha256"),
            "native_build": telemetry.get("native_build"), "runner_sha256": runner_sha256,
            "compiler_commit": compiler_identity["commit"], "compiler_dirty": compiler_identity["dirty"],
        },
        "command": command, "command_config_hash": command_hash,
        "parent_operations": parent_records, "matrices": matrices,
        "acceptance": {"type": "convention_only", "thresholds": thresholds, "byte_identity_required": False},
        "native": {"protocol": PROTOCOL, "scope": "fixture", "threads": telemetry.get("threads"),
                   "native_build": telemetry.get("native_build"), "source_passes": telemetry.get("source_passes"),
                   "operation_count": len(telemetry["operations"])},
        "memory": {"host_memory_cap_bytes": workspace.host_memory_cap_bytes, "staging_buffer_bytes": chunk_bytes,
                   "bounded_chunk_bytes": chunk_bytes, "maximum_python_read_bytes": chunk_bytes,
                   "native_maximum_source_row_bytes": telemetry.get("maximum_source_row_bytes"),
                   "whole_tensor_materialization": False, "telemetry_is_bounded_not_peak_rss": True},
        "limitations": ["This is the frozen 48-matrix Ordinary-versus-Unsloth convention diagnostic, not a full Ordinary conversion.", "Native compiler owns conversion arithmetic; Python only performs bounded diagnostic decode/statistics because no native comparator contract exists for these selected ranges.", "No model repository code, framework, activation comparison, quality claim, or QAT attribution is performed."],
    }


def _reverify_source(
    kind: str, root: Path, lock: Path, expected: Any, workspace: BoundedWorkspace
) -> None:
    current = verify_source_lock(lock, root, workspace)
    if (current.lock_sha256 != expected.lock_sha256 or
            current.repository != expected.repository or
            current.revision != expected.revision or
            {name: (item.size, item.sha256) for name, item in current.files.items()} !=
            {name: (item.size, item.sha256) for name, item in expected.files.items()}):
        raise SourceVerificationError(f"{kind} source identity changed during diagnostic")


def run_diagnostic(*, ordinary_root: Path, ordinary_inventory: Path, unsloth_root: Path, unsloth_inventory: Path,
                   config_path: Path, native_encoder: Path, output: Path, max_host_memory: int = DEFAULT_MEMORY_BYTES,
                   chunk_bytes: int = DEFAULT_CHUNK_BYTES, threads: int = 1, timeout_seconds: int = 1800) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise DataError(f"refusing to overwrite existing output: {output}")
    _positive_int(chunk_bytes, "chunk_bytes", 1 << 20)
    _positive_int(threads, "threads", 64); _positive_int(timeout_seconds, "timeout_seconds")
    config_path = _regular(config_path, "M06 config")
    native_encoder = _regular(native_encoder, "native encoder", executable=True)
    workspace = BoundedWorkspace(max_host_memory, chunk_bytes)
    # This diagnostic still scans tens of gigabytes of locked sources. Reject
    # an accidental Debug compiler before any checkpoint hashing, then bind the
    # executable staged for conversion to the same SHA-256 and build identity.
    native_preflight = preflight_native_nvfp4(
        NativeNvfp4Request(native_encoder, timeout_seconds, threads), workspace
    )
    ordinary_inventory = _regular(ordinary_inventory, "Ordinary inventory")
    unsloth_inventory = _regular(unsloth_inventory, "Unsloth inventory")
    config = load_json(config_path)
    records = _validate_config(config)
    ordinary_doc = load_json(ordinary_inventory); unsloth_doc = load_json(unsloth_inventory)
    ordinary_lock = _resolve_reference(config["diagnostic_references"]["ordinary_bf16"]["path"], config_path)
    unsloth_lock = _resolve_reference(config["diagnostic_references"]["unsloth_nvfp4"]["path"], config_path)
    ordinary_source, ordinary, ordinary_id = _load_source(ordinary_root, ordinary_inventory, ordinary_lock, config, ordinary_doc, "ordinary_bf16", workspace)
    unsloth_source, unsloth, unsloth_id = _load_source(unsloth_root, unsloth_inventory, unsloth_lock, config, unsloth_doc, "unsloth_nvfp4", workspace)
    parents, geometries = _build_parent_geometries(records, ordinary, unsloth_doc, workspace)
    for geometry in geometries:
        for role, item in list(geometry["components"].items()):
            descriptor = unsloth[item["name"]]
            geometry["components"][role + "_path"] = descriptor.path
            geometry["components"][role + "_offset"] = descriptor.absolute_offset
            geometry["components"][role + "_bytes"] = descriptor.byte_length
    config_hash = _sha256(config_path)
    command = [
        "python3", "tools/run_gemma4_26b_nvfp4_diagnostic.py",
        "--ordinary-root", str(ordinary_root.expanduser().absolute()),
        "--ordinary-inventory", str(ordinary_inventory.expanduser().absolute()),
        "--unsloth-root", str(unsloth_root.expanduser().absolute()),
        "--unsloth-inventory", str(unsloth_inventory.expanduser().absolute()),
        "--config", str(config_path.expanduser().absolute()),
        "--native-encoder", str(native_encoder.expanduser().absolute()),
        "--output", str(output.expanduser().absolute()),
        "--threads", str(threads), "--chunk-bytes", str(chunk_bytes),
        "--max-host-memory", str(max_host_memory), "--timeout-seconds", str(timeout_seconds),
    ]
    command_hash = hashlib.sha256(canonical_json_bytes(command)).hexdigest()
    compiler_commit, compiler_dirty = git_compiler_identity(Path(__file__).resolve().parents[1])
    compiler_identity = {"commit": compiler_commit, "dirty": compiler_dirty}
    runner_hash = _sha256(Path(__file__).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".m06-nvfp4-diagnostic-", dir=str(output.parent)) as raw:
        staging = Path(raw); telemetry_path = staging / "native-telemetry.json"
        staged_encoder, staged_hash = _stage_executable(native_encoder, staging, workspace)
        staged_build = _query_build_info(staged_encoder)
        if (staged_hash != native_preflight.binary_sha256 or
                staged_build != native_preflight.native_build):
            raise SourceVerificationError(
                "native M06 compiler changed between diagnostic preflight and execution"
            )
        job = _native_job(parents, staging, threads)
        telemetry = _run_native(staged_encoder, job, telemetry_path, timeout_seconds)
        if telemetry.get("native_build") != native_preflight.native_build:
            raise DataError("native M06 diagnostic build identity changed during execution")
        telemetry["native_encoder_sha256"] = staged_hash
        report = _build_report(config, config_hash, command, command_hash, runner_hash, compiler_identity,
                               ordinary_id, unsloth_id, parents, geometries, telemetry, chunk_bytes, workspace)
    # The comparison consumed both locked source families.  Re-verify every
    # locked file before publishing so post-verification mutation cannot be
    # attributed to the recorded identities.
    for kind, root, lock, expected in (
        ("ordinary_bf16", ordinary_root, ordinary_lock, ordinary_source),
        ("unsloth_nvfp4", unsloth_root, unsloth_lock, unsloth_source),
    ):
        _reverify_source(kind, root, lock, expected, workspace)
    payload = canonical_json_bytes(report)
    try:
        with output.open("xb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise DataError(f"refusing to overwrite existing output: {output}") from exc
    return report


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary-root", type=Path, required=True)
    parser.add_argument("--ordinary-inventory", type=Path, required=True)
    parser.add_argument("--unsloth-root", type=Path, required=True)
    parser.add_argument("--unsloth-inventory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native-encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-host-memory", type=int, default=DEFAULT_MEMORY_BYTES)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def _diagnostic_exit_code(report: dict[str, Any]) -> int:
    return 0 if report.get("status") == "pass" and report.get("pass") is True else 4


def main() -> int:
    try:
        args = _args()
        report = run_diagnostic(
            ordinary_root=args.ordinary_root, ordinary_inventory=args.ordinary_inventory,
            unsloth_root=args.unsloth_root, unsloth_inventory=args.unsloth_inventory,
            config_path=args.config, native_encoder=args.native_encoder, output=args.output,
            max_host_memory=args.max_host_memory, chunk_bytes=args.chunk_bytes,
            threads=args.threads, timeout_seconds=args.timeout_seconds,
        )
        exit_code = _diagnostic_exit_code(report)
        if exit_code != 0:
            print("M06 diagnostic failed acceptance thresholds", file=os.sys.stderr)
        return exit_code
    except CompilerError as exc:
        print(f"M06 diagnostic failed: {exc}", file=os.sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"M06 diagnostic failed: {exc}", file=os.sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())