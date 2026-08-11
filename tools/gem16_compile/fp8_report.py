"""Bounded deterministic comparison of compiled and reference FP8 attention tensors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any

from .common import (
    BoundedWorkspace,
    DataError,
    InvalidPlanError,
    OutputError,
    SourceVerificationError,
    canonical_json_bytes,
    load_json,
    reject_duplicate_keys,
    safe_relative_path,
    write_file_atomic,
)
from .profiles import (
    M05_APPROVED_SOURCE_LOCKS,
    M05_ATTENTION_TABLE,
    M05_DEQUANTIZATION_EQUATION,
    M05_SOURCE_CONTRACT,
)
from .quantize_fp8 import decode_bf16, decode_e4m3fn, E4M3FN_NAN_CODE
from .reader import (
    TensorDescriptor,
    read_artifact_tensors,
    read_source_tensors,
    verify_source_lock,
)


REPORT_SCHEMA_VERSION = 1
PROFILE = "fp8-attention-partial-v1"
ARTIFACT_STATUS = "m05_fp8_attention_partial_not_runtime_loadable"
GLOBAL_LAYERS = frozenset({5, 11, 17, 23, 29})
ATTENTION_RE = re.compile(
    r"^model\.language_model\.layers\.(0|[1-9][0-9]*)\.self_attn\.([qkvo])_proj\.(weight|weight_scale)$"
)
EXPECTED_WEIGHT_BYTES = 1_110_179_840
EXPECTED_SCALE_BYTES = 670_720
_EXPECTED_UNSLOTH_REPOSITORY = "unsloth/gemma-4-26B-A4B-it-NVFP4"
_EXPECTED_UNSLOTH_REVISION = "20df0542b1a86ce19f495ac2eca2c7c12bce82f9"

# Decode once.  Invalid NaN encodings remain None and are rejected before use.
_E4M3_VALUES: tuple[float | None, ...] = tuple(
    None if (code & 0x7F) == E4M3FN_NAN_CODE else decode_e4m3fn(code)
    for code in range(256)
)


@dataclass(frozen=True)
class AttentionPair:
    layer: int
    role: str
    stem: str
    weight: TensorDescriptor
    scale: TensorDescriptor


class _Neumaier:
    """Stable, deterministic binary64 sum with fixed left-to-right updates."""

    __slots__ = ("total", "correction")

    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        value = float(value)
        trial = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - trial) + value
        else:
            self.correction += (value - trial) + self.total
        self.total = trial

    def value(self) -> float:
        return self.total + self.correction


def _ensure_finite_json(value: Any, path: str = "report") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DataError(f"non-finite JSON value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _ensure_finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _ensure_finite_json(item, f"{path}[{index}]")


def _finite(value: float, description: str) -> float:
    if not math.isfinite(value):
        raise DataError(f"non-finite comparison value: {description}")
    return float(value)


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, _finite(value, "clamped metric")))


def _relative(error_energy: float, reference_energy: float) -> float | None:
    if reference_energy == 0.0:
        return 0.0 if error_energy == 0.0 else None
    return _finite(math.sqrt(error_energy / reference_energy), "relative L2")


def _cosine(dot: float, left_energy: float, right_energy: float) -> float:
    if left_energy == 0.0 and right_energy == 0.0:
        return 1.0
    if left_energy == 0.0 or right_energy == 0.0:
        return 0.0
    return _clamp(dot / math.sqrt(left_energy * right_energy))


def _pearson(
    count: int,
    left_sum: float,
    right_sum: float,
    left_energy: float,
    right_energy: float,
    dot: float,
    equal: bool,
) -> float:
    left_variance = count * left_energy - left_sum * left_sum
    right_variance = count * right_energy - right_sum * right_sum
    covariance = count * dot - left_sum * right_sum
    if left_variance <= 0.0 or right_variance <= 0.0:
        return 1.0 if equal else 0.0
    return _clamp(covariance / math.sqrt(left_variance * right_variance))


def _sqnr(signal_energy: float, error_energy: float) -> float | None:
    if error_energy == 0.0 or signal_energy == 0.0:
        return None
    return _finite(10.0 * math.log10(signal_energy / error_energy), "SQNR")


def _read_exact(stream: Any, view: memoryview, description: str) -> None:
    position = 0
    while position < len(view):
        try:
            count = stream.readinto(view[position:])
        except OSError as error:
            raise DataError(f"cannot read {description}: {error}") from error
        if not isinstance(count, int) or count <= 0:
            raise DataError(
                f"short read for {description}: expected {len(view)}, got {position}"
            )
        position += count


def _check_range(descriptor: TensorDescriptor) -> None:
    try:
        size = descriptor.path.stat().st_size
    except OSError as error:
        raise SourceVerificationError(
            f"cannot inspect tensor file {descriptor.path}: {error}"
        ) from error
    if descriptor.absolute_offset < 0 or descriptor.byte_length < 0 or (
        descriptor.absolute_offset > size - descriptor.byte_length
    ):
        raise SourceVerificationError(
            f"tensor range is outside file for {descriptor.name}: "
            f"offset={descriptor.absolute_offset} length={descriptor.byte_length} size={size}"
        )


def _read_stream_bytes(
    stream: Any,
    descriptor: TensorDescriptor,
    offset: int,
    length: int,
    buffer: memoryview,
    description: str,
) -> bytes:
    if offset < 0 or length < 0 or offset + length > descriptor.byte_length:
        raise DataError(f"tensor subrange is invalid for {description}")
    if length > len(buffer):
        raise DataError(f"comparison chunk exceeds staging buffer for {description}")
    try:
        stream.seek(descriptor.absolute_offset + offset)
        view = buffer[:length]
        _read_exact(stream, view, description)
        return bytes(view)
    except DataError:
        raise
    except (OSError, ValueError) as error:
        raise DataError(f"cannot access {description}: {error}") from error


def _descriptor_hash(descriptor: TensorDescriptor, workspace: BoundedWorkspace) -> str:
    _check_range(descriptor)
    return workspace.hash_tensor_range(
        descriptor.path, descriptor.absolute_offset, descriptor.byte_length
    )


def _attention_pairs(
    tensors: dict[str, TensorDescriptor], side: str, production: bool = False
) -> dict[str, AttentionPair]:
    grouped: dict[str, dict[str, TensorDescriptor]] = {}
    for name, descriptor in tensors.items():
        match = ATTENTION_RE.fullmatch(name)
        if match is None:
            continue
        layer = int(match.group(1))
        role = match.group(2)
        if layer >= 30 or (role == "v" and layer in GLOBAL_LAYERS):
            raise DataError(f"{side} contains forbidden attention projection: {name}")
        stem = name.rsplit(".", 1)[0]
        grouped.setdefault(stem, {})[match.group(3)] = descriptor

    expected_names = set(M05_ATTENTION_TABLE) if production else {
        stem + ".weight" for stem, group in grouped.items()
        if set(group) == {"weight", "weight_scale"}
    }
    expected: dict[str, AttentionPair] = {}
    for name in sorted(expected_names):
        stem = name.removesuffix(".weight")
        group = grouped.get(stem)
        if group is None or set(group) != {"weight", "weight_scale"}:
            raise DataError(f"{side} lacks complete attention pair: {stem}")
        weight = group["weight"]
        scale = group["weight_scale"]
        if weight.dtype != "F8_E4M3" or scale.dtype != "BF16":
            raise DataError(f"{side} attention dtype mismatch: {stem}")
        if production:
            spec = M05_ATTENTION_TABLE[name]
            if tuple(weight.shape) != spec.shape or tuple(scale.shape) != (spec.shape[0], 1):
                raise DataError(f"{side} attention shape mismatch: {stem}")
            if weight.byte_length != spec.shape[0] * spec.shape[1] or scale.byte_length != spec.shape[0] * 2:
                raise DataError(f"{side} attention byte mismatch: {stem}")
            role_name = spec.role.removeprefix("attention_").removesuffix("_projection")
        else:
            if len(weight.shape) != 2 or tuple(scale.shape) != (weight.shape[0], 1):
                raise DataError(f"{side} attention shape mismatch: {stem}")
            role_name = ATTENTION_RE.fullmatch(name).group(2)  # type: ignore[union-attr]
        expected[stem] = AttentionPair(
            int(name.split(".layers.", 1)[1].split(".", 1)[0]),
            role_name,
            stem,
            weight,
            scale,
        )
    if production and set(grouped) != {name.removesuffix(".weight") for name in expected_names}:
        extra = sorted(set(grouped) - {name.removesuffix(".weight") for name in expected_names})
        raise DataError(f"{side} contains unexpected attention projections: {extra[:3]}")
    if production and (
        len(expected) != 115
        or sum(pair.weight.byte_length for pair in expected.values()) != EXPECTED_WEIGHT_BYTES
        or sum(pair.scale.byte_length for pair in expected.values()) != EXPECTED_SCALE_BYTES
    ):
        raise DataError(f"{side} does not match the Gemma 4 26B FP8 attention dimensions")
    return dict(sorted(expected.items()))


def _validate_fp8_payload(
    descriptor: TensorDescriptor, workspace: BoundedWorkspace, side: str
) -> tuple[str, int, int, int]:
    digest = _descriptor_hash(descriptor, workspace)
    chunk_bytes = min(descriptor.byte_length, max(1, workspace.staging_bytes // 2))
    workspace.record_transform_row(chunk_bytes, f"validating {side} FP8 payload")
    view = memoryview(workspace.buffer)[:chunk_bytes]
    nan_count = endpoint_7e = endpoint_fe = 0
    try:
        with descriptor.path.open("rb", buffering=0) as stream:
            stream.seek(descriptor.absolute_offset)
            remaining = descriptor.byte_length
            while remaining:
                count = min(remaining, chunk_bytes)
                _read_exact(stream, view[:count], f"{side} FP8 payload {descriptor.name}")
                for code in view[:count]:
                    nan_count += int((code & 0x7F) == E4M3FN_NAN_CODE)
                    endpoint_7e += int(code == 0x7E)
                    endpoint_fe += int(code == 0xFE)
                remaining -= count
                workspace.check(f"validating {side} FP8 payload")
    except OSError as error:
        raise DataError(f"cannot validate {side} FP8 payload {descriptor.name}: {error}") from error
    finally:
        view.release()
    if nan_count:
        raise DataError(f"{side} FP8 tensor contains NaN codes: {descriptor.name}")
    return digest, endpoint_7e, endpoint_fe, nan_count


def _validate_scale_values(
    descriptor: TensorDescriptor, workspace: BoundedWorkspace, side: str
) -> tuple[str, list[float], list[int]]:
    digest = _descriptor_hash(descriptor, workspace)
    values: list[float] = []
    bits: list[int] = []
    scratch = memoryview(workspace.buffer)[:2]
    try:
        with descriptor.path.open("rb", buffering=0) as stream:
            stream.seek(descriptor.absolute_offset)
            for row in range(descriptor.shape[0]):
                _read_exact(stream, scratch, f"{side} BF16 scale {descriptor.name} row {row}")
                bit_pattern = struct.unpack_from("<H", scratch)[0]
                value = decode_bf16(bit_pattern)
                if not math.isfinite(value) or value <= 0.0:
                    raise DataError(f"{side} attention scale is not finite and positive: {descriptor.name}")
                bits.append(bit_pattern)
                values.append(value)
                workspace.check(f"validating {side} BF16 scales")
    except OSError as error:
        raise DataError(f"cannot validate {side} scale {descriptor.name}: {error}") from error
    finally:
        scratch.release()
    return digest, values, bits


def _read_json_snapshot(path: Path, maximum: int = 64 * 1024 * 1024) -> tuple[dict[str, Any], str, tuple[int, int]]:
    try:
        stat = path.stat()
        if stat.st_size > maximum:
            raise DataError(f"JSON exceeds {maximum} bytes: {path}")
        payload = path.read_bytes()
        if len(payload) != stat.st_size:
            raise DataError(f"JSON changed while reading: {path}")
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except DataError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataError(f"cannot parse JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise DataError(f"JSON root must be an object: {path}")
    return value, hashlib.sha256(payload).hexdigest(), (stat.st_size, stat.st_mtime_ns)


def _artifact_tensors(
    artifact: Path,
    workspace: BoundedWorkspace,
    production: bool = False,
    allow_dirty_compiler: bool = False,
) -> tuple[dict[str, TensorDescriptor], dict[str, Any], str, Path]:
    if artifact.is_symlink():
        raise SourceVerificationError("compiled artifact root must not be a symlink")
    try:
        root = artifact.resolve(strict=True)
    except OSError as error:
        raise SourceVerificationError(f"cannot resolve compiled artifact: {error}") from error
    if not root.is_dir():
        raise SourceVerificationError(f"compiled artifact is not a directory: {artifact}")
    manifest_path = root / "gem16_compilation.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SourceVerificationError("compiled artifact lacks gem16_compilation.json")
    manifest, manifest_hash, identity = _read_json_snapshot(manifest_path)
    expected_manifest_fields = {
        "schema_version", "artifact_profile", "artifact_status", "source", "compiler",
        "plan", "quantization", "head_format", "text_only", "omitted_families",
        "omitted_tensor_groups", "excluded_tensors", "tensors", "files",
        "file_hash_scope", "byte_totals", "compiler_settings",
    }
    if set(manifest) != expected_manifest_fields or manifest.get("schema_version") != 1:
        raise DataError("compiled gem16_compilation.json schema mismatch")
    if manifest.get("artifact_profile") != PROFILE or manifest.get("artifact_status") != ARTIFACT_STATUS:
        raise DataError("compiled artifact is not the M05 FP8 attention partial profile")
    if manifest.get("head_format") != "deferred" or manifest.get("text_only") is not True:
        raise DataError("compiled artifact has an invalid M05 profile boundary")
    compiler = manifest.get("compiler")
    source_record = manifest.get("source")
    plan_record = manifest.get("plan")
    if not isinstance(compiler, dict) or not isinstance(source_record, dict) or not isinstance(plan_record, dict):
        raise DataError("compiled provenance records are malformed")
    if production and compiler.get("implementation") != "gem16_compile_m05_native_v1":
        raise DataError("compiled artifact compiler identity is not the native M05 implementation")
    if production and compiler.get("dirty") is not False and not allow_dirty_compiler:
        raise DataError("compiled artifact compiler identity is not a clean M05 build")
    if production:
        native_record = compiler.get("native_encoder")
        if not isinstance(native_record, dict) or set(native_record) != {"protocol", "sha256", "threads", "build"}:
            raise DataError("compiled artifact native encoder provenance is malformed")
        from .native_fp8 import _native_build
        _native_build(native_record.get("build"), "compiled artifact native build")
    commit = compiler.get("commit")
    if production and (not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit)):
        raise DataError("compiled artifact compiler commit is invalid")
    if production and source_record.get("lock_sha256") not in M05_APPROVED_SOURCE_LOCKS:
        raise DataError("compiled artifact is not bound to an approved Ordinary/QAT source lock")
    if production and plan_record.get("source_contract") != M05_SOURCE_CONTRACT:
        raise DataError("compiled artifact source contract is not M05")
    if manifest.get("quantization") != {
        "profile": PROFILE, "attention": "fp8-per-output-row-v1",
        "experts": "deferred-to-m06", "embedding_head": "deferred-to-m07",
        "production_quantization_implemented": False,
    }:
        raise DataError("compiled artifact quantization contract is not M05 FP8 attention")
    if production and (len(manifest.get("tensors", [])) != 230 or len(manifest.get("excluded_tensors", [])) != 898):
        raise DataError("compiled artifact provenance does not cover the complete M05 source")
    if production:
        totals = manifest.get("byte_totals")
        if not isinstance(totals, dict) or totals.get("output_tensor_count") != 230 or totals.get("output_tensor_bytes") != EXPECTED_WEIGHT_BYTES + EXPECTED_SCALE_BYTES:
            raise DataError("compiled artifact byte totals do not match M05 attention")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise DataError("compiled artifact file manifest is missing")
    records: dict[str, dict[str, Any]] = {}
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "kind", "size", "sha256"}:
            raise DataError("compiled artifact file record is malformed")
        relative = record.get("path")
        if not isinstance(relative, str) or relative in records:
            raise DataError("compiled artifact file path is invalid or duplicated")
        safe_relative_path(relative, "compiled artifact file")
        if relative == "gem16_compilation.json":
            raise DataError("compilation manifest must not list itself")
        records[relative] = record
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SourceVerificationError(f"compiled artifact contains symlink: {path.name}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise SourceVerificationError(f"compiled artifact contains non-file: {path}")
    if actual != set(records) | {"gem16_compilation.json"}:
        raise DataError("compiled artifact file set differs from gem16_compilation.json")
    for relative, record in records.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise DataError(f"compiled artifact file is unsafe or missing: {relative}")
        size = path.stat().st_size
        digest = workspace.hash_range(path, 0, size)
        if record.get("size") != size or record.get("sha256") != digest:
            raise DataError(f"compiled artifact file hash/size mismatch: {relative}")
    shards = tuple(sorted(relative for relative, record in records.items() if record.get("kind") == "safetensors_shard"))
    indexes = [relative for relative, record in records.items() if record.get("kind") == "safetensors_index"]
    if len(indexes) != 1 or not shards:
        raise DataError("compiled artifact must declare shards and one index")
    tensors = read_artifact_tensors(root, shards, indexes[0], workspace)
    manifest_tensors = manifest.get("tensors")
    if not isinstance(manifest_tensors, list) or len(manifest_tensors) != len(tensors):
        raise DataError("compiled artifact tensor provenance count mismatch")
    names = {record.get("output_name") for record in manifest_tensors if isinstance(record, dict)}
    if names != set(tensors):
        raise DataError("compiled artifact tensor provenance names mismatch")
    if manifest_path.stat().st_size != identity[0] or manifest_path.stat().st_mtime_ns != identity[1]:
        raise SourceVerificationError("compiled manifest changed during comparison")
    return tensors, manifest, manifest_hash, root


def _reverify_artifact_files(
    root: Path, manifest: dict[str, Any], manifest_hash: str, workspace: BoundedWorkspace
) -> None:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise DataError("compiled artifact files are malformed during final verification")
    expected = {record.get("path") for record in files if isinstance(record, dict)} | {"gem16_compilation.json"}
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise SourceVerificationError("compiled artifact symlink appeared during comparison")
    actual = {path.relative_to(root).as_posix() for path in paths if path.is_file()}
    if actual != expected:
        raise SourceVerificationError("compiled artifact file set changed during comparison")
    manifest_path = root / "gem16_compilation.json"
    manifest_stat = manifest_path.stat()
    if workspace.hash_range(manifest_path, 0, manifest_stat.st_size) != manifest_hash:
        raise SourceVerificationError("compiled manifest changed during comparison")
    for record in files:
        if not isinstance(record, dict):
            raise DataError("compiled artifact file record changed during comparison")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise DataError("compiled artifact file path changed during comparison")
        path = root / relative
        stat = path.stat()
        digest = workspace.hash_range(path, 0, stat.st_size)
        if stat.st_size != record.get("size") or digest != record.get("sha256"):
            raise SourceVerificationError(f"compiled artifact file changed during comparison: {relative}")


def _matrix_report(
    pair: AttentionPair, right: AttentionPair, workspace: BoundedWorkspace,
    left_weight_hash: str, right_weight_hash: str, left_scale_hash: str, right_scale_hash: str,
    left_endpoints: tuple[int, int], right_endpoints: tuple[int, int],
    left_nan_count: int, right_nan_count: int,
    left_scales: list[float], right_scales: list[float],
    left_scale_bits: list[int], right_scale_bits: list[int],
) -> dict[str, Any]:
    rows, columns = pair.weight.shape
    chunk = min(columns, max(1, workspace.staging_bytes // 2))
    workspace.record_transform_row(chunk * 2, f"comparing {pair.stem}")
    left_view = memoryview(workspace.buffer)[:chunk]
    right_view = memoryview(workspace.buffer)[chunk:2 * chunk]
    raw_mismatch = 0
    left_energy = _Neumaier(); right_energy = _Neumaier(); diff_energy = _Neumaier(); dot = _Neumaier()
    left_min = right_min = math.inf; left_max = right_max = -math.inf; max_error = 0.0
    try:
        _check_range(pair.weight); _check_range(right.weight)
        with pair.weight.path.open("rb", buffering=0) as left_stream, right.weight.path.open("rb", buffering=0) as right_stream:
            for row in range(rows):
                ls, rs = left_scales[row], right_scales[row]
                for begin in range(0, columns, chunk):
                    count = min(chunk, columns - begin)
                    left = _read_stream_bytes(left_stream, pair.weight, row * columns + begin, count, left_view, "compiled FP8 row")
                    right_bytes = _read_stream_bytes(right_stream, right.weight, row * columns + begin, count, right_view, "Unsloth FP8 row")
                    for left_code, right_code in zip(left, right_bytes, strict=True):
                        raw_mismatch += int(left_code != right_code)
                        left_dec = _E4M3_VALUES[left_code]; right_dec = _E4M3_VALUES[right_code]
                        if left_dec is None or right_dec is None:
                            raise DataError("NaN FP8 code reached reconstruction")
                        left_value = left_dec * ls; right_value = right_dec * rs
                        left_energy.add(left_value * left_value); right_energy.add(right_value * right_value)
                        dot.add(left_value * right_value)
                        error = left_value - right_value; diff_energy.add(error * error)
                        max_error = max(max_error, abs(error)); left_min = min(left_min, left_value); right_min = min(right_min, right_value)
                        left_max = max(left_max, left_value); right_max = max(right_max, right_value)
                    workspace.check(f"comparing {pair.stem} row {row}")
    except OSError as error:
        raise DataError(f"cannot open FP8 comparison tensors for {pair.stem}: {error}") from error
    finally:
        left_view.release(); right_view.release()
    le, re, de, dp = left_energy.value(), right_energy.value(), diff_energy.value(), dot.value()
    elements = rows * columns
    scale_l2 = _Neumaier(); scale_le = _Neumaier(); scale_re = _Neumaier(); scale_dot = _Neumaier(); scale_ls = _Neumaier(); scale_rs = _Neumaier()
    for ls, rs in zip(left_scales, right_scales, strict=True):
        scale_l2.add((ls-rs)*(ls-rs)); scale_le.add(ls*ls); scale_re.add(rs*rs); scale_dot.add(ls*rs); scale_ls.add(ls); scale_rs.add(rs)
    scale_equal = left_scale_bits == right_scale_bits
    return {
        "name": pair.stem, "layer": pair.layer, "role": pair.role,
        "weight": {"left_name": pair.weight.name, "right_name": right.weight.name, "left_dtype": pair.weight.dtype, "right_dtype": right.weight.dtype, "left_shard": pair.weight.shard, "right_shard": right.weight.shard, "left_data_offset": pair.weight.data_offset, "right_data_offset": right.weight.data_offset, "shape": [rows, columns], "left_bytes": pair.weight.byte_length, "right_bytes": right.weight.byte_length, "left_sha256": left_weight_hash, "right_sha256": right_weight_hash, "raw_mismatch_count": raw_mismatch, "raw_mismatch_rate": raw_mismatch / elements, "left_endpoint_code_counts": {"0x7e": left_endpoints[0], "0xfe": left_endpoints[1]}, "right_endpoint_code_counts": {"0x7e": right_endpoints[0], "0xfe": right_endpoints[1]}, "left_nan_code_count": left_nan_count, "right_nan_code_count": right_nan_count},
        "scale": {"left_name": pair.scale.name, "right_name": right.scale.name, "left_dtype": pair.scale.dtype, "right_dtype": right.scale.dtype, "left_shard": pair.scale.shard, "right_shard": right.scale.shard, "left_data_offset": pair.scale.data_offset, "right_data_offset": right.scale.data_offset, "shape": [rows, 1], "left_bytes": pair.scale.byte_length, "right_bytes": right.scale.byte_length, "left_sha256": left_scale_hash, "right_sha256": right_scale_hash, "mismatch_count": sum(a != b for a,b in zip(left_scale_bits,right_scale_bits,strict=True)), "left_min": min(left_scales), "left_max": max(left_scales), "right_min": min(right_scales), "right_max": max(right_scales), "left_sum": scale_ls.value(), "right_sum": scale_rs.value(), "left_sum_squares": scale_le.value(), "right_sum_squares": scale_re.value(), "difference_sum_squares": scale_l2.value(), "dot": scale_dot.value(), "relative_l2": _relative(scale_l2.value(), scale_re.value()), "pearson_correlation": _pearson(rows, scale_ls.value(), scale_rs.value(), scale_le.value(), scale_re.value(), scale_dot.value(), scale_equal)},
        "reconstruction": {"left_min": left_min, "left_max": left_max, "left_rms": math.sqrt(le/elements), "left_sum_squares": le, "right_min": right_min, "right_max": right_max, "right_rms": math.sqrt(re/elements), "right_sum_squares": re, "difference_sum_squares": de, "dot": dp, "relative_l2": _relative(de,re), "cosine_similarity": _cosine(dp,le,re), "max_absolute_error": max_error, "sqnr_db": _sqnr(re,de), "perfect_reconstruction": de == 0.0, "zero_reference": re == 0.0},
    }


def _aggregate(matrix_reports: list[dict[str, Any]]) -> dict[str, Any]:
    raw = scale = elements = weight_bytes = scale_bytes = 0
    endpoint_left = {"0x7e": 0, "0xfe": 0}; endpoint_right = {"0x7e": 0, "0xfe": 0}; nan_left = nan_right = 0
    left_energy = _Neumaier(); right_energy = _Neumaier(); diff_energy = _Neumaier(); dot = _Neumaier(); scale_l2 = _Neumaier(); scale_re = _Neumaier(); scale_le = _Neumaier(); scale_dot = _Neumaier(); scale_ls = _Neumaier(); scale_rs = _Neumaier()
    max_error = 0.0; left_min = right_min = math.inf; left_max = right_max = -math.inf; scale_count = 0
    for report in matrix_reports:
        w,s,r = report["weight"],report["scale"],report["reconstruction"]; raw += w["raw_mismatch_count"]; scale += s["mismatch_count"]; rows,columns = w["shape"]; elements += rows*columns; weight_bytes += w["left_bytes"]; scale_bytes += s["left_bytes"]; scale_count += rows
        for key in endpoint_left: endpoint_left[key] += w["left_endpoint_code_counts"][key]; endpoint_right[key] += w["right_endpoint_code_counts"][key]
        nan_left += w["left_nan_code_count"]; nan_right += w["right_nan_code_count"]
        left_min=min(left_min,r["left_min"]); left_max=max(left_max,r["left_max"]); right_min=min(right_min,r["right_min"]); right_max=max(right_max,r["right_max"]); max_error=max(max_error,r["max_absolute_error"])
        left_energy.add(r["left_sum_squares"]); right_energy.add(r["right_sum_squares"]); diff_energy.add(r["difference_sum_squares"]); dot.add(r["dot"])
        scale_l2.add(s["difference_sum_squares"]); scale_re.add(s["right_sum_squares"]); scale_le.add(s["left_sum_squares"]); scale_dot.add(s["dot"]); scale_ls.add(s["left_sum"]); scale_rs.add(s["right_sum"])
    le,re,de,dp=left_energy.value(),right_energy.value(),diff_energy.value(),dot.value(); sle,sre,sdl,sd=scale_le.value(),scale_re.value(),scale_l2.value(),scale_dot.value()
    return {"matrix_count":len(matrix_reports),"weight_bytes":weight_bytes,"scale_bytes":scale_bytes,"elements":elements,"raw_mismatch_count":raw,"raw_mismatch_rate":raw/elements if elements else 0.0,"scale_mismatch_count":scale,"left_endpoint_code_counts":endpoint_left,"right_endpoint_code_counts":endpoint_right,"left_nan_code_count":nan_left,"right_nan_code_count":nan_right,"max_absolute_error":max_error,"left_min":left_min,"left_max":left_max,"right_min":right_min,"right_max":right_max,"reconstruction_left_sum_squares":le,"reconstruction_right_sum_squares":re,"reconstruction_difference_sum_squares":de,"reconstruction_dot":dp,"reconstruction_relative_l2":_relative(de,re),"reconstruction_cosine_similarity":_cosine(dp,le,re),"reconstruction_sqnr_db":_sqnr(re,de),"scale_relative_l2":_relative(sdl,sre),"scale_pearson_correlation":_pearson(scale_count,scale_ls.value(),scale_rs.value(),sle,sre,sd,scale==0),"reconstruction_metrics_are_per_matrix":False}


def compare_attention(compiled: Path, unsloth_lock: Path, unsloth_source: Path, workspace: BoundedWorkspace, *, verified_report: dict[str, Any] | None = None, production: bool = False, native_encoder: Path | None = None, threads: int = 1, native_timeout_seconds: int = 600, allow_dirty_compiler: bool = False) -> dict[str, Any]:
    source = verify_source_lock(unsloth_lock, unsloth_source, workspace)
    if production and (source.repository != _EXPECTED_UNSLOTH_REPOSITORY or source.revision != _EXPECTED_UNSLOTH_REVISION):
        raise SourceVerificationError("Unsloth source is not the pinned M05 reference")
    right_tensors = read_source_tensors(source, workspace)
    left_tensors, manifest, manifest_hash, artifact_root = _artifact_tensors(
        compiled, workspace, production, allow_dirty_compiler
    )
    if production and verified_report is None:
        raise DataError("production comparison requires public compiler verification")
    right_pairs = _attention_pairs(right_tensors, "Unsloth", production)
    left_pairs = _attention_pairs(left_tensors, "compiled", production)
    if set(left_pairs) != set(right_pairs):
        raise DataError("compiled and Unsloth attention pair sets differ")

    if production:
        if native_encoder is None:
            raise InvalidPlanError("production M05 comparison requires --native-encoder")
        from .native_fp8_compare import build_compare_job, project_report, run_native_compare
        job, pairs = build_compare_job(left_pairs, right_pairs, workspace, threads=threads)
        staging_parent = compiled.expanduser().absolute().parent / ".incomplete"
        raw, binary_hash, native_build = run_native_compare(
            native_encoder, job, workspace, staging_parent,
            timeout_seconds=native_timeout_seconds,
        )
        # This is a source/artifact mutation check after the native pass, not a
        # second conversion or elementwise comparison.
        _reverify_artifact_files(artifact_root, manifest, manifest_hash, workspace)
        verify_source_lock(unsloth_lock, unsloth_source, workspace)
        result = project_report(
            raw, pairs, binary_hash=binary_hash, native_build=native_build, threads=threads,
            manifest=manifest, manifest_hash=manifest_hash,
            source=source, tensor_count=len(right_tensors),
            staging_bytes=workspace.staging_bytes, production=True,
        )
        if manifest.get("compiler", {}).get("dirty") is True:
            result["limitations"].append(
                "The compiled input was produced from a dirty worktree and is diagnostic evidence only."
            )
        _ensure_finite_json(result)
        return result

    # The Python implementation remains a small-fixture numerical oracle only;
    # production M05 always takes the native branch above.
    reports=[]
    for name in sorted(right_pairs):
        left,right=left_pairs[name],right_pairs[name]
        left_hash,l7e,lfe,lnan=_validate_fp8_payload(left.weight,workspace,"compiled"); right_hash,r7e,rfe,rnan=_validate_fp8_payload(right.weight,workspace,"Unsloth")
        left_sh,left_sc,left_bits=_validate_scale_values(left.scale,workspace,"compiled"); right_sh,right_sc,right_bits=_validate_scale_values(right.scale,workspace,"Unsloth")
        reports.append(_matrix_report(left,right,workspace,left_hash,right_hash,left_sh,right_sh,(l7e,lfe),(r7e,rfe),lnan,rnan,left_sc,right_sc,left_bits,right_bits))
    output = _aggregate(reports)
    semantic_memory = {"staging_buffer_bytes": workspace.staging_bytes, "maximum_chunk_bytes": min(max(1, workspace.staging_bytes//2), max(p.weight.shape[1] for p in left_pairs.values())), "maximum_transform_row_bytes": workspace.maximum_transform_row_bytes, "accumulation": "binary64_neumaier_left_to_right", "report_excludes_rss": True}
    compiled_record = manifest.get("compiler",{}); plan_record=manifest.get("plan",{}); source_record=manifest.get("source",{})
    result = {"schema_version":REPORT_SCHEMA_VERSION,"family":"attention","status":"pass","artifact_profile":PROFILE,"artifact_status":ARTIFACT_STATUS,"source":{"unsloth_lock_sha256":source.lock_sha256,"repository":source.repository,"revision":source.revision,"tensor_count":len(right_tensors)},"compiled":{"compilation_manifest_sha256":manifest_hash,"tensor_count":len(left_tensors),"source_lock_sha256":source_record.get("lock_sha256"),"source_contract":plan_record.get("source_contract"),"compiler_commit":compiled_record.get("commit"),"compiler_dirty":compiled_record.get("dirty"),"compiler_implementation":compiled_record.get("implementation"),"compiler_manifest_sha256":plan_record.get("compiler_manifest_sha256"),"resolved_plan_sha256":plan_record.get("resolved_plan_sha256")},"contract":{"comparison_schema_version":REPORT_SCHEMA_VERSION,"weight_dtype":"F8_E4M3","scale_dtype":"BF16","codec":"E4M3FN","dequantization_equation":M05_DEQUANTIZATION_EQUATION,"operator_output_comparison":"not_performed_in_M05_weight_comparison","accumulation":"binary64_neumaier_left_to_right","scale_mismatch_unit":"BF16 row bit-pattern","reference_environment":{"byteorder":"little","system":"Linux","machine":"x86_64"}},"matrices":reports,"aggregates":output,"memory":semantic_memory,"limitations":["This report compares stored weights and row scales only.","No activation or CUDA operator-output comparison is performed.","Differences are not model-quality or QAT attribution evidence."]}
    _ensure_finite_json(result)
    return result


def write_report(path: Path, report: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise OutputError(f"comparison report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_file_atomic(path, canonical_json_bytes(report))


__all__ = ["compare_attention", "write_report", "_attention_pairs", "_aggregate", "_matrix_report", "_pearson"]
