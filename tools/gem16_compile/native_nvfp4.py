"""Native direct-range NVFP4 compiler adapter for M06 experts and M07 tied head.

The Python side only builds descriptor-bound jobs, validates native telemetry and
records hashes.  Numerical conversion is exclusively performed by the native
``gem16-nvfp4-direct-v1`` executable into preallocated Safetensors shard ranges.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any, BinaryIO

from .common import (
    BoundedWorkspace,
    CompilerError,
    DataError,
    InvalidPlanError,
    OutputError,
    SourceVerificationError,
    canonical_json_bytes,
    reject_duplicate_keys,
    write_file_atomic,
)
from .native_fp8 import (
    MAX_DIAGNOSTIC_BYTES,
    _native_build,
    _process_group_pids,
    _process_group_rss_bytes,
    _terminate_process,
    _wait_process_group_gone,
)
from .plan import QuantizationPlan, TensorCompilePlan
from .reader import TensorDescriptor
from .writer import DirectShardLayout

PROTOCOL = "gem16-nvfp4-direct-v1"
CONTRACT_ID = "gem16.nvfp4_bf16_group16"
CONTRACT_VERSION = 1


@dataclass(frozen=True)
class NativeNvfp4Request:
    executable: Path
    timeout_seconds: int = 14_400
    threads: int = 1
    profile: str = "nvfp4-experts-partial-v1"


@dataclass(frozen=True)
class NativeNvfp4Preflight:
    binary_sha256: str
    native_build: dict[str, str]


@dataclass(frozen=True)
class NativeNvfp4Output:
    output_name: str
    source_sha256: str
    output_sha256: str
    output_bytes: int
    statistics: dict[str, object]


@dataclass(frozen=True)
class NativeNvfp4Result:
    outputs: dict[str, NativeNvfp4Output]
    binary_sha256: str
    threads: int
    maximum_source_row_bytes: int
    child_peak_rss_bytes: int
    native_build: dict[str, str]
    source_passes: int
    analysis_seconds: float
    conversion_seconds: float
    telemetry: dict[str, Any]


def _regular_executable(path: Path) -> Path:
    path = path.expanduser().absolute()
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise OutputError(f"native M06 compiler is not a regular file: {path}")
        if not os.access(path, os.X_OK, follow_symlinks=False):
            raise OutputError(f"native M06 compiler is not executable: {path}")
    except OSError as error:
        raise OutputError(f"cannot inspect native M06 compiler {path}: {error}") from error
    return path


def _stage_executable(source: Path, staging: Path, workspace: BoundedWorkspace) -> tuple[Path, str]:
    source = _regular_executable(source)
    staged = staging / ".nvfp4_encoder"
    descriptor = output_descriptor = -1
    try:
        before = source.stat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(source, flags)
        output_descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o500)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, min(workspace.staging_bytes, 1024 * 1024))
            if not chunk:
                break
            digest.update(chunk)
            position = 0
            while position < len(chunk):
                written = os.write(output_descriptor, chunk[position:])
                if written <= 0:
                    raise OutputError("short write while staging native M06 compiler")
                position += written
            workspace.check("staging native M06 compiler")
        os.fsync(output_descriptor)
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise SourceVerificationError("native M06 compiler changed while staging")
        staged_digest = hashlib.sha256()
        with staged.open("rb", buffering=0) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                staged_digest.update(chunk)
        staged_hash = staged_digest.hexdigest()
        if staged_hash != digest.hexdigest():
            raise SourceVerificationError("staged native M06 compiler hash mismatch")
        os.chmod(staged, 0o500)
        return staged, staged_hash
    except CompilerError:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    except OSError as error:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass
        raise OutputError(f"cannot stage native M06 compiler: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if output_descriptor >= 0:
            os.close(output_descriptor)


def _query_build_info(executable: Path, timeout_seconds: float = 10.0) -> dict[str, str]:
    """Read build identity before any source hashing or native conversion."""
    try:
        process = subprocess.Popen(
            [str(executable), "--build-info"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            close_fds=True, start_new_session=(os.name == "posix"),
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            _terminate_process(process, process.pid if os.name == "posix" else None)
            raise OutputError("native M06 build-info query timed out") from error
    except OSError as error:
        raise OutputError(f"native M06 build-info query failed: {error}") from error
    if len(stdout) > 64 * 1024 or len(stderr) > MAX_DIAGNOSTIC_BYTES:
        raise OutputError("native M06 build-info output exceeds safety limit")
    if process.returncode != 0 or stderr:
        detail = stderr.decode("utf-8", errors="replace")
        raise OutputError(f"native M06 build-info query failed with exit {process.returncode}: {detail}")
    try:
        document = json.loads(stdout.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DataError(f"native M06 build-info JSON is invalid: {error}") from error
    if not isinstance(document, dict) or set(document) != {"protocol", "native_build"}:
        raise DataError("native M06 build-info schema mismatch")
    if document["protocol"] != PROTOCOL:
        raise DataError("native M06 build-info protocol mismatch")
    return _native_build(document["native_build"], "native M06 build-info native_build")


def preflight_native_nvfp4(
    request: NativeNvfp4Request, workspace: BoundedWorkspace
) -> NativeNvfp4Preflight:
    """Stage and identify the native binary before reading the checkpoint.

    This deliberately uses an isolated temporary directory rather than the
    eventual artifact staging directory, so a rejected Debug/mismatched binary
    cannot create or modify an output artifact.
    """
    with tempfile.TemporaryDirectory(prefix="gem16-m06-native-") as temporary:
        staging = Path(temporary)
        staged, binary_sha256 = _stage_executable(request.executable, staging, workspace)
        native_build = _query_build_info(staged)
        if native_build.get("build_type") != "Release":
            if request.profile == "nvfp4-tied-head-partial-v1":
                message = "M07 tied-head conversion requires a native Release build"
            else:
                message = "M06 full conversion requires a native Release build"
            raise InvalidPlanError(f"{message}; got {native_build.get('build_type')!r}")
        return NativeNvfp4Preflight(binary_sha256, native_build)


def _sha256_hex(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise DataError(f"native M06 telemetry hash is invalid: {field}")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DataError(f"native M06 telemetry scalar is invalid: {field}")
    return float(value)


def _statistics(item: dict[str, Any], component: str) -> dict[str, object]:
    elements = int(item["rows"]) * int(item["columns"])
    source_energy = _finite(item["source_sum_squares"], "source_sum_squares")
    error_energy = _finite(item["error_sum_squares"], "error_sum_squares")
    reconstruction_energy = _finite(item["reconstruction_sum_squares"], "reconstruction_sum_squares")
    dot = _finite(item["source_reconstruction_dot"], "source_reconstruction_dot")
    if min(source_energy, error_energy, reconstruction_energy) < 0:
        raise DataError(f"native M06 telemetry energy is negative: {item['source_name']}")
    source_norm = math.sqrt(source_energy)
    reconstructed_norm = math.sqrt(reconstruction_energy)
    cosine = dot / (source_norm * reconstructed_norm) if source_norm and reconstructed_norm else (1.0 if source_energy == reconstruction_energy == 0 else 0.0)
    cosine = max(-1.0, min(1.0, cosine))
    result: dict[str, object] = {
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "component": component,
        "rows": int(item["rows"]),
        "columns": int(item["columns"]),
        "elements": elements,
        "source_min": _finite(item["source_min"], "source_min"),
        "source_max": _finite(item["source_max"], "source_max"),
        "source_rms": math.sqrt(source_energy / elements),
        "relative_l2_error": math.sqrt(error_energy / source_energy) if source_energy else 0.0,
        "cosine_similarity": cosine,
        "max_absolute_error": _finite(item["max_absolute_error"], "max_absolute_error"),
        "sqnr_db": None if error_energy == 0 or source_energy == 0 else 10.0 * math.log10(source_energy / error_energy),
        "perfect_reconstruction": error_energy == 0,
        "tensor_amax": _finite(item["tensor_amax"], "tensor_amax"),
        "weight_divisor": _finite(item["weight_divisor"], "weight_divisor"),
        "input_divisor": _finite(item["input_divisor"], "input_divisor"),
        "scale_min": _finite(item["scale_min"], "scale_min"),
        "scale_max": _finite(item["scale_max"], "scale_max"),
        "zero_blocks": int(item["zero_blocks"]),
        "underflow_blocks": int(item["underflow_blocks"]),
        "saturation_count": int(item["saturation_count"]),
        "code_histogram": list(item["code_histogram"]),
        "scale_histogram": list(item["scale_histogram"]),
    }
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise DataError(f"native M06 derived statistic is non-finite: {key}")
    return result


def _group_plans(plan: QuantizationPlan) -> list[dict[str, TensorCompilePlan]]:
    groups: dict[str, dict[str, TensorCompilePlan]] = {}
    component_by_encoder = {
        "nvfp4-packed-v1": "packed",
        "nvfp4-local-scale-v1": "local_scale",
        "nvfp4-weight-divisor-v1": "weight_global",
        "nvfp4-input-divisor-v1": "input_global",
    }
    for tensor in plan.tensors:
        component = component_by_encoder.get(tensor.encoder)
        if component is None:
            raise DataError(f"unsupported native M06 encoder: {tensor.encoder}")
        group = groups.setdefault(tensor.operation_id, {})
        if component in group:
            raise DataError(f"duplicate M06 component: {tensor.operation_id}:{component}")
        group[component] = tensor
    result = []
    for operation_id in sorted(groups):
        group = groups[operation_id]
        if set(group) != {"packed", "local_scale", "weight_global", "input_global"}:
            raise DataError(f"M06 operation does not contain exactly four components: {operation_id}")
        source_sets = {item.source_names for item in group.values()}
        if len(source_sets) != 1 or len(next(iter(source_sets))) != 1:
            raise DataError(f"M06 operation source mismatch: {operation_id}")
        result.append(group)
    return result


def _make_job(
    plan: QuantizationPlan,
    source_tensors: dict[str, TensorDescriptor],
    layout: DirectShardLayout,
    workspace: BoundedWorkspace,
    threads: int,
) -> tuple[dict[str, Any], list[dict[str, TensorCompilePlan]], dict[str, str]]:
    if threads < 1 or threads > 64:
        raise InvalidPlanError("--threads must be in the range 1..64")
    groups = _group_plans(plan)
    source_hashes: dict[str, str] = {}
    operations: list[dict[str, Any]] = []
    for group in groups:
        packed = group["packed"]
        source_name = packed.source_names[0]
        source = source_tensors.get(source_name)
        if source is None or source.dtype != "BF16" or len(source.shape) not in (2, 3):
            raise DataError(f"invalid M06 source tensor: {source_name}")
        if tuple(packed.logical_shape) != tuple(source.shape):
            raise DataError(f"M06 source shape mismatch: {source_name}")
        elements = 1
        for dimension in source.shape:
            elements *= dimension
        if source.byte_length != elements * 2:
            raise DataError(f"M06 source byte count mismatch: {source_name}")
        source_path = source.path.expanduser().absolute()
        if source_path.is_symlink():
            raise SourceVerificationError(f"M06 source shard is a symlink: {source_path}")
        try:
            source_path = source_path.resolve(strict=True)
            if not source_path.is_file() or source_path.is_symlink():
                raise SourceVerificationError(f"M06 source shard is not a regular file: {source_path}")
        except OSError as error:
            raise SourceVerificationError(f"cannot resolve M06 source shard {source_path}: {error}") from error
        source_hash = source_hashes.get(source_name)
        if source_hash is None:
            source_hash = workspace.hash_tensor_range(source_path, source.absolute_offset, source.byte_length)
            source_hashes[source_name] = source_hash
        rows = 1
        for dimension in source.shape[:-1]:
            rows *= dimension
        columns = source.shape[-1]
        elements = rows * columns
        expected_shapes = {
            "packed": tuple(source.shape[:-1]) + (columns // 2,),
            "local_scale": tuple(source.shape[:-1]) + (columns // 16,),
            "weight_global": (1,),
            "input_global": (1,),
        }
        expected_bytes = {
            "packed": elements // 2,
            "local_scale": elements // 16,
            "weight_global": 4,
            "input_global": 4,
        }
        outputs: dict[str, Any] = {}
        for component, tensor in group.items():
            if tensor.physical_shape != expected_shapes[component] or tensor.output_bytes != expected_bytes[component]:
                raise DataError(f"M06 output shape/bytes mismatch: {tensor.output_name}")
            if tensor.output_name not in layout.outputs:
                raise DataError(f"M06 output has no prepared shard range: {tensor.output_name}")
            path, offset = layout.outputs[tensor.output_name]
            outputs[component] = {
                "component": component,
                "name": tensor.output_name,
                "path": str(path),
                "offset": offset,
                "bytes": tensor.output_bytes,
            }
        operations.append({
            "operation_id": packed.operation_id,
            "source_name": source_name,
            "source_path": str(source_path),
            "source_sha256": source_hash,
            "source_offset": source.absolute_offset,
            "source_bytes": source.byte_length,
            "source_dtype": source.dtype,
            "logical_shape": list(source.shape),
            "rows": rows,
            "columns": columns,
            "role": packed.role,
            "axis_transformation": packed.axis_transformation,
            "disk_layout": packed.disk_layout,
            "runtime_layout": packed.runtime_layout,
            **outputs,
        })
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "artifact_profile": plan.artifact_profile,
        "scope": ("tied_head" if plan.artifact_profile == "nvfp4-tied-head-partial-v1"
                  else "full" if len(operations) == 150 else "fixture"),
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "threads": threads,
        "operations": operations,
    }, groups, source_hashes


def _parse_telemetry(
    path: Path,
    job: dict[str, Any],
    layout: DirectShardLayout,
    workspace: BoundedWorkspace,
) -> tuple[dict[str, NativeNvfp4Output], int, dict[str, str], int, float, float, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DataError(f"cannot read native M06 telemetry: {error}") from error
    if len(raw) > 64 * 1024 * 1024:
        raise DataError("native M06 telemetry exceeds safety limit")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DataError(f"native M06 telemetry is invalid JSON: {error}") from error
    required = {"schema_version", "protocol", "artifact_profile", "scope", "contract_id", "contract_version", "threads", "source_passes", "maximum_source_row_bytes", "analysis_seconds", "conversion_seconds", "native_build", "operations"}
    if not isinstance(document, dict) or set(document) != required:
        raise DataError("native M06 telemetry root schema mismatch")
    if document["schema_version"] != 1 or document["protocol"] != PROTOCOL or document["artifact_profile"] != job["artifact_profile"] or document["scope"] != job["scope"] or document["contract_id"] != CONTRACT_ID or document["contract_version"] != CONTRACT_VERSION or document["threads"] != job["threads"] or document["source_passes"] != 2:
        raise DataError("native M06 telemetry contract mismatch")
    if not isinstance(document["threads"], int) or not 1 <= document["threads"] <= 64:
        raise DataError("native M06 telemetry thread count is invalid")
    source_passes = document["source_passes"]
    if not isinstance(source_passes, int) or source_passes != 2:
        raise DataError("native M06 source pass count is invalid")
    analysis_seconds = _finite(document["analysis_seconds"], "analysis_seconds")
    conversion_seconds = _finite(document["conversion_seconds"], "conversion_seconds")
    if analysis_seconds < 0.0 or conversion_seconds < 0.0:
        raise DataError("native M06 phase durations must be nonnegative")
    maximum_row = document["maximum_source_row_bytes"]
    if not isinstance(maximum_row, int) or maximum_row < 0:
        raise DataError("native M06 maximum source row is invalid")
    expected_maximum = max((int(item["columns"]) * 2 for item in job["operations"]), default=0)
    if maximum_row != expected_maximum:
        raise DataError("native M06 maximum source row mismatch")
    build = _native_build(document["native_build"], "native M06 telemetry native_build")
    operations = document["operations"]
    if not isinstance(operations, list) or len(operations) != len(job["operations"]):
        raise DataError("native M06 telemetry operation count mismatch")
    output_records: dict[str, NativeNvfp4Output] = {}
    for item, expected in zip(operations, job["operations"], strict=True):
        required_item = {"operation_id", "source_name", "source_sha256", "source_bytes", "source_dtype", "role", "axis_transformation", "disk_layout", "runtime_layout", "logical_shape", "rows", "columns", "packed_name", "packed_sha256", "local_scale_name", "local_scale_sha256", "weight_global_name", "weight_global_sha256", "input_global_name", "input_global_sha256", "tensor_amax", "weight_divisor", "input_divisor", "source_min", "source_max", "source_sum_squares", "reconstruction_sum_squares", "source_reconstruction_dot", "error_sum_squares", "max_absolute_error", "scale_min", "scale_max", "zero_blocks", "underflow_blocks", "saturation_count", "code_histogram", "scale_histogram"}
        if not isinstance(item, dict) or set(item) != required_item:
            raise DataError("native M06 telemetry operation schema mismatch")
        if item["operation_id"] != expected["operation_id"] or item["source_name"] != expected["source_name"] or item["source_sha256"] != expected["source_sha256"] or item["source_bytes"] != expected["source_bytes"] or item["source_dtype"] != expected["source_dtype"] or item["role"] != expected["role"] or item["axis_transformation"] != expected["axis_transformation"] or item["disk_layout"] != expected["disk_layout"] or item["runtime_layout"] != expected["runtime_layout"] or item["logical_shape"] != expected["logical_shape"] or item["rows"] != expected["rows"] or item["columns"] != expected["columns"]:
            raise DataError(f"native M06 telemetry identity mismatch: {expected['source_name']}")
        _sha256_hex(item["source_sha256"], "source_sha256")
        elements = int(item["rows"]) * int(item["columns"])
        if not isinstance(item["rows"], int) or not isinstance(item["columns"], int) or item["rows"] <= 0 or item["columns"] <= 0:
            raise DataError("native M06 telemetry shape is invalid")
        for key in ("source_bytes", "zero_blocks", "underflow_blocks", "saturation_count"):
            if not isinstance(item[key], int) or item[key] < 0:
                raise DataError(f"native M06 telemetry count is invalid: {key}")
        if item["source_bytes"] != elements * 2 or item["zero_blocks"] > elements // 16 or item["underflow_blocks"] > elements // 16 or item["saturation_count"] > elements:
            raise DataError(f"native M06 telemetry count bounds are invalid: {item['source_name']}")
        for key in ("tensor_amax", "weight_divisor", "input_divisor", "source_min", "source_max", "source_sum_squares", "reconstruction_sum_squares", "source_reconstruction_dot", "error_sum_squares", "max_absolute_error", "scale_min", "scale_max"):
            _finite(item[key], key)
        if item["tensor_amax"] < 0 or item["weight_divisor"] <= 0 or item["input_divisor"] <= 0 or item["source_min"] > item["source_max"] or item["scale_min"] > item["scale_max"]:
            raise DataError(f"native M06 telemetry numeric invariants are invalid: {item['source_name']}")
        for key in ("packed_sha256", "local_scale_sha256", "weight_global_sha256", "input_global_sha256"):
            _sha256_hex(item[key], key)
        code_hist = item["code_histogram"]
        scale_hist = item["scale_histogram"]
        if not isinstance(code_hist, list) or len(code_hist) != 16 or any(not isinstance(v, int) or v < 0 for v in code_hist) or sum(code_hist) != elements:
            raise DataError(f"native M06 code histogram is invalid: {item['source_name']}")
        if not isinstance(scale_hist, list) or len(scale_hist) != 256 or any(not isinstance(v, int) or v < 0 for v in scale_hist) or sum(scale_hist) != elements // 16:
            raise DataError(f"native M06 scale histogram is invalid: {item['source_name']}")
        component_hashes = {
            "packed": item["packed_sha256"], "local_scale": item["local_scale_sha256"],
            "weight_global": item["weight_global_sha256"], "input_global": item["input_global_sha256"],
        }
        for component, output_name in (("packed", item["packed_name"]), ("local_scale", item["local_scale_name"]), ("weight_global", item["weight_global_name"]), ("input_global", item["input_global_name"])):
            expected_output = next((op[component] for op in job["operations"] if op["operation_id"] == expected["operation_id"]), None)
            if expected_output is None or output_name != expected_output["name"]:
                raise DataError(f"native M06 telemetry output identity mismatch: {output_name}")
            path_value, offset = layout.outputs[output_name]
            actual = workspace.hash_range(path_value, offset, expected_output["bytes"])
            if actual != component_hashes[component]:
                raise DataError(f"native M06 output hash mismatch: {output_name}")
            statistics = _statistics(item, component)
            output_records[output_name] = NativeNvfp4Output(output_name, item["source_sha256"], actual, expected_output["bytes"], statistics)
    return output_records, maximum_row, build, source_passes, analysis_seconds, conversion_seconds, document


def prepare_native_direct(
    request: NativeNvfp4Request,
    plan: QuantizationPlan,
    source_tensors: dict[str, TensorDescriptor],
    workspace: BoundedWorkspace,
    staging: Path,
    layout: DirectShardLayout,
    expected_preflight: NativeNvfp4Preflight | None = None,
) -> NativeNvfp4Result:
    if request.timeout_seconds <= 0:
        raise InvalidPlanError("--native-timeout-seconds must be positive")
    job_path = staging / ".nvfp4_job.json"
    telemetry_path = staging / ".nvfp4_telemetry.json"
    for path in (job_path, telemetry_path):
        if path.exists() or path.is_symlink():
            raise OutputError(f"native M06 temporary output already exists: {path}")
    staged_executable, binary_hash = _stage_executable(request.executable, staging, workspace)
    try:
        native_build = _query_build_info(staged_executable)
        if expected_preflight is not None:
            if binary_hash != expected_preflight.binary_sha256:
                raise SourceVerificationError(
                    "native M06 compiler changed between preflight and execution"
                )
            if native_build != expected_preflight.native_build:
                raise DataError(
                    "native M06 build identity changed between preflight and execution"
                )
        operation_count = sum(1 for tensor in plan.tensors if tensor.encoder == "nvfp4-packed-v1")
        requires_release = plan.artifact_profile in {
            "nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1"
        } and (operation_count == 150 or plan.artifact_profile == "nvfp4-tied-head-partial-v1")
        if requires_release and native_build.get("build_type") != "Release":
            if plan.artifact_profile == "nvfp4-tied-head-partial-v1":
                message = "M07 tied-head conversion requires a native Release build"
            else:
                message = "M06 full conversion requires a native Release build"
            raise InvalidPlanError(f"{message}; got {native_build.get('build_type')!r}")
        # This is deliberately after build preflight: a rejected Debug full run
        # must not hash or otherwise scan the multi-gigabyte source checkpoint.
        job, _groups, _source_hashes = _make_job(
            plan, source_tensors, layout, workspace, request.threads
        )
        write_file_atomic(job_path, canonical_json_bytes(job))
    except Exception:
        try:
            staged_executable.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    child_peak_rss = 0
    stdout_data = bytearray()
    stderr_data = bytearray()
    overflow = threading.Event()

    def drain(stream: BinaryIO, destination: bytearray) -> None:
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    return
                if len(destination) + len(chunk) > MAX_DIAGNOSTIC_BYTES:
                    overflow.set()
                    return
                destination.extend(chunk)
        except OSError:
            overflow.set()

    try:
        env = os.environ.copy()
        env["LC_ALL"] = "C.UTF-8"
        env["LANG"] = "C.UTF-8"
        available = max(64 * 1024 * 1024, workspace.host_memory_cap_bytes - workspace.baseline_rss_bytes)

        def limit_child() -> None:
            if os.name == "posix":
                try:
                    import resource
                    resource.setrlimit(resource.RLIMIT_AS, (available, available))
                except (ImportError, OSError, ValueError) as error:
                    raise RuntimeError(f"cannot apply native M06 address-space limit: {error}") from error

        process = subprocess.Popen(
            [str(staged_executable), "--nvfp4-job", str(job_path), "--telemetry", str(telemetry_path)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, close_fds=True, start_new_session=(os.name == "posix"),
            preexec_fn=limit_child if os.name == "posix" else None,
        )
        stdout_thread = threading.Thread(target=drain, args=(process.stdout, stdout_data), daemon=True)
        stderr_thread = threading.Thread(target=drain, args=(process.stderr, stderr_data), daemon=True)
        stdout_thread.start(); stderr_thread.start()
        pgid = process.pid if os.name == "posix" else None
        deadline = time.monotonic() + request.timeout_seconds
        timed_out = memory_exceeded = False
        while True:
            child_peak_rss = max(child_peak_rss, _process_group_rss_bytes(pgid) if pgid is not None else 0)
            leader_done = process.poll() is not None
            readers_done = not stdout_thread.is_alive() and not stderr_thread.is_alive()
            if child_peak_rss + workspace.max_observed_rss_bytes > workspace.host_memory_cap_bytes:
                memory_exceeded = True; _terminate_process(process, pgid)
            elif overflow.is_set():
                _terminate_process(process, pgid)
            elif time.monotonic() >= deadline:
                timed_out = True; _terminate_process(process, pgid)
            if (leader_done and readers_done) or time.monotonic() >= deadline + 2.0:
                if not leader_done or not readers_done:
                    _terminate_process(process, pgid)
                break
            time.sleep(0.02)
        process.wait(); stdout_thread.join(timeout=2.0); stderr_thread.join(timeout=2.0)
        if process.stdout is not None: process.stdout.close()
        if process.stderr is not None: process.stderr.close()
        if not timed_out and not memory_exceeded and not overflow.is_set() and pgid is not None:
            remaining = _process_group_pids(pgid)
            if remaining:
                gone = _terminate_process(process, pgid)
                suffix = "" if gone else "; process group survived SIGKILL wait"
                raise OutputError(f"native M06 compiler left process-group descendants: {sorted(remaining)}{suffix}")
        if timed_out:
            raise OutputError(f"native M06 compiler timed out after {request.timeout_seconds}s")
        if memory_exceeded:
            raise DataError("native M06 compiler exceeded combined host-memory cap")
        if overflow.is_set():
            raise OutputError("native M06 compiler diagnostics exceed safety limit")
        if stdout_data:
            raise DataError("native M06 compiler wrote unexpected stdout")
        detail = bytes(stderr_data).decode("utf-8", errors="replace")
        if process.returncode != 0:
            message = f"native M06 compiler failed with exit {process.returncode}: {detail}"
            if process.returncode == InvalidPlanError.exit_code: raise InvalidPlanError(message)
            if process.returncode == SourceVerificationError.exit_code: raise SourceVerificationError(message)
            if process.returncode == DataError.exit_code: raise DataError(message)
            raise OutputError(message)
        if not telemetry_path.is_file() or telemetry_path.is_symlink():
            raise DataError("native M06 compiler did not publish telemetry")
        outputs, maximum_row, build, source_passes, analysis_seconds, conversion_seconds, telemetry = _parse_telemetry(telemetry_path, job, layout, workspace)
        if build != native_build:
            raise DataError("native M06 build identity changed between preflight and conversion")
        workspace.record_transform_row(maximum_row, "native M06 source row")
        return NativeNvfp4Result(outputs, binary_hash, request.threads, maximum_row, child_peak_rss, build, source_passes, analysis_seconds, conversion_seconds, telemetry)
    except subprocess.SubprocessError as error:
        raise OutputError(f"native M06 compiler could not start: {error}") from error
    except OSError as error:
        raise OutputError(f"native M06 compiler process failed: {error}") from error
    finally:
        try: staged_executable.unlink(missing_ok=True)
        except OSError: pass
        for path in (job_path, telemetry_path):
            try: path.unlink(missing_ok=True)
            except OSError: pass


__all__ = [
    "NativeNvfp4Preflight", "NativeNvfp4Request", "NativeNvfp4Result",
    "NativeNvfp4Output", "PROTOCOL", "preflight_native_nvfp4",
    "prepare_native_direct",
]
