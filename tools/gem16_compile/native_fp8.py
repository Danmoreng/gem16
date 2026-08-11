"""Native M05 FP8 batch-encoder protocol adapter.

The Python compiler owns plans, publication, and provenance.  It never performs
M05 conversion here; the versioned host executable performs one bounded pass
per source matrix and writes a temporary bundle consumed by this adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
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
from .encoders import EncoderResult, TensorEncoder
from .plan import QuantizationPlan, TensorCompilePlan
from .reader import TensorDescriptor

PROTOCOL = "gem16-fp8-batch-v1"
CONTRACT_ID = "gem16.fp8_attention_rowwise"
CONTRACT_VERSION = 1
MAX_DIAGNOSTIC_BYTES = 1 << 20
NATIVE_BUILD_KEYS = frozenset({
    "compiler_id", "compiler_version", "build_type", "cxx_standard",
    "system", "processor",
})


def _native_build(value: Any, where: str = "native_build") -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != NATIVE_BUILD_KEYS:
        raise DataError(f"{where} schema mismatch")
    if any(not isinstance(item, str) or not item for item in value.values()):
        raise DataError(f"{where} values must be non-empty strings")
    return {key: value[key] for key in sorted(NATIVE_BUILD_KEYS)}


@dataclass(frozen=True)
class NativeMatrix:
    source_name: str
    weight_name: str
    scale_name: str
    weight_offset: int
    weight_bytes: int
    scale_offset: int
    scale_bytes: int
    rows: int
    columns: int
    source_sha256: str
    telemetry: dict[str, Any]


@dataclass(frozen=True)
class NativeBundle:
    payload: Path
    telemetry_path: Path
    matrices: dict[str, NativeMatrix]
    binary_sha256: str
    threads: int
    maximum_source_row_bytes: int
    child_peak_rss_bytes: int
    native_build: dict[str, str]

    def cleanup(self) -> None:
        for path in (self.payload, self.telemetry_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


@dataclass(frozen=True)
class NativeRequest:
    executable: Path
    timeout_seconds: int = 600
    threads: int = 1


def _regular_executable(path: Path) -> Path:
    path = path.expanduser().absolute()
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise OutputError(f"native M05 encoder is not a regular file: {path}")
        if not os.access(path, os.X_OK, follow_symlinks=False):
            raise OutputError(f"native M05 encoder is not executable: {path}")
    except OSError as error:
        raise OutputError(f"cannot inspect native M05 encoder {path}: {error}") from error
    return path


def _stage_executable(source: Path, staging: Path, workspace: BoundedWorkspace) -> tuple[Path, str]:
    """Copy one verified executable inode into private staging before execution."""
    source = _regular_executable(source)
    staged = staging / ".m05_fp8_encoder"
    descriptor = -1
    output_descriptor = -1
    try:
        before = source.stat()
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) |
                             getattr(os, "O_NOFOLLOW", 0))
        output_descriptor = os.open(
            staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o500
        )
        digest = hashlib.sha256()
        try:
            while True:
                chunk = os.read(descriptor, min(workspace.staging_bytes, 1024 * 1024))
                if not chunk:
                    break
                digest.update(chunk)
                workspace.check("staging native M05 encoder")
                position = 0
                while position < len(chunk):
                    written = os.write(output_descriptor, chunk[position:])
                    if written <= 0:
                        raise OutputError("short write while staging native M05 encoder")
                    position += written
            os.fsync(output_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if output_descriptor >= 0:
                os.close(output_descriptor)
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise SourceVerificationError("native M05 encoder changed while staging")
        staged_hash = binary_sha256(staged)
        if staged_hash != digest.hexdigest():
            raise SourceVerificationError("staged native M05 encoder hash mismatch")
        os.chmod(staged, 0o500)
        return staged, staged_hash
    except CompilerError:
        try:
            staged.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except OSError:
            pass
        raise
    except OSError as error:
        try:
            staged.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except OSError:
            pass
        raise OutputError(f"cannot stage native M05 encoder: {error}") from error


def _process_group_pids(pgid: int) -> set[int]:
    if os.name != "posix" or not Path("/proc").is_dir():
        return set()
    result: set[int] = set()
    for candidate in Path("/proc").glob("[0-9]*"):
        try:
            # The comm field may contain spaces; fields after the final ')'
            # are stable: state, ppid, pgrp, ... .
            stat_text = (candidate / "stat").read_text(encoding="ascii")
            suffix = stat_text.rsplit(")", 1)[1].split()
            if len(suffix) >= 3 and int(suffix[2]) == pgid:
                result.add(int(candidate.name))
        except (OSError, ValueError, IndexError):
            continue
    return result


def _process_group_rss_bytes(pgid: int) -> int:
    total = 0
    for pid in _process_group_pids(pgid):
        try:
            for line in (Path("/proc") / str(pid) / "status").read_text(encoding="ascii").splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total


def _wait_process_group_gone(pgid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_pids(pgid):
            return True
        time.sleep(0.02)
    return not _process_group_pids(pgid)


def _terminate_process(process: subprocess.Popen[bytes], pgid: int | None) -> bool:
    """Terminate the complete native process group, including descendants."""
    posix_group = os.name == "posix" and pgid is not None
    try:
        if posix_group:
            os.killpg(pgid, signal.SIGTERM)
        elif process.poll() is None:
            process.terminate()
    except OSError:
        pass

    group_gone = _wait_process_group_gone(pgid, 1.0) if posix_group else False
    if posix_group and not group_gone:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
        group_gone = _wait_process_group_gone(pgid, 2.0)
    elif not posix_group and process.poll() is None:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass

    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
    return group_gone if posix_group else process.poll() is not None


def binary_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise OutputError(f"cannot hash native M05 encoder {path}: {error}") from error
    return digest.hexdigest()


def _pair_plans(plan: QuantizationPlan) -> list[tuple[TensorCompilePlan, TensorCompilePlan]]:
    groups: dict[str, list[TensorCompilePlan]] = {}
    for item in plan.tensors:
        groups.setdefault(item.operation_id, []).append(item)
    pairs: list[tuple[TensorCompilePlan, TensorCompilePlan]] = []
    for operation, values in sorted(groups.items()):
        weights = [v for v in values if v.encoder == "fp8-rowwise-weight-v1"]
        scales = [v for v in values if v.encoder == "fp8-rowwise-scale-v1"]
        if len(values) != 2 or len(weights) != 1 or len(scales) != 1:
            raise DataError(f"M05 operation is not one weight/scale pair: {operation}")
        if weights[0].source_names != scales[0].source_names:
            raise DataError(f"M05 pair source mismatch: {operation}")
        pairs.append((weights[0], scales[0]))
    if len(pairs) * 2 != len(plan.tensors):
        raise DataError("M05 plan tensor count is not pair-aligned")
    return pairs


def _make_job(
    plan: QuantizationPlan,
    source_tensors: dict[str, TensorDescriptor],
    payload_bytes: int,
    threads: int,
) -> tuple[dict[str, Any], list[tuple[TensorCompilePlan, TensorCompilePlan]], dict[str, str]]:
    pairs = _pair_plans(plan)
    matrices: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    offset = 0
    for weight, scale in pairs:
        if len(weight.source_names) != 1:
            raise DataError(f"M05 pair must have one source: {weight.output_name}")
        source_name = weight.source_names[0]
        source = source_tensors.get(source_name)
        if source is None or len(source.shape) != 2 or source.dtype != "BF16":
            raise DataError(f"invalid M05 source tensor: {source_name}")
        rows, columns = source.shape
        if tuple(weight.physical_shape) != tuple(source.shape) or tuple(scale.physical_shape) != (rows, 1):
            raise DataError(f"M05 source/output shape mismatch: {source_name}")
        source_hashes[source_name] = ""  # Filled by prepare_native_bundle.
        resolved_source = source.path.expanduser().resolve(strict=True)
        if not resolved_source.is_file() or resolved_source.is_symlink():
            raise SourceVerificationError(f"M05 source shard is not a regular file: {resolved_source}")
        if source.absolute_offset < 0 or source.byte_length < 0:
            raise DataError(f"invalid M05 source range: {source_name}")
        weight_bytes = weight.output_bytes
        scale_bytes = scale.output_bytes
        matrices.append({
            "source_name": source_name,
            "source_path": str(resolved_source),
            "source_offset": source.absolute_offset,
            "source_bytes": source.byte_length,
            "rows": rows,
            "columns": columns,
            "weight_output_name": weight.output_name,
            "weight_offset": offset,
            "weight_bytes": weight_bytes,
            "scale_output_name": scale.output_name,
            "scale_offset": offset + weight_bytes,
            "scale_bytes": scale_bytes,
        })
        offset += weight_bytes + scale_bytes
    if offset != payload_bytes:
        raise DataError(f"native M05 payload size mismatch: expected {payload_bytes}, got {offset}")
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "threads": threads,
        "payload_bytes": payload_bytes,
        "matrices": matrices,
    }, pairs, source_hashes


def _read_telemetry(path: Path, expected: dict[str, Any]) -> tuple[dict[str, Any], int]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DataError(f"cannot read native M05 telemetry: {error}") from error
    if len(raw) > 64 * 1024 * 1024:
        raise DataError("native M05 telemetry exceeds safety limit")
    try:
        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DataError(f"native M05 telemetry is invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise DataError("native M05 telemetry root must be an object")
    required = {"schema_version", "contract_id", "contract_version", "payload_bytes", "threads", "maximum_source_row_bytes", "native_build", "matrices"}
    if set(document) != required:
        raise DataError("native M05 telemetry root schema mismatch")
    if document["schema_version"] != 1 or document["contract_id"] != CONTRACT_ID or document["contract_version"] != CONTRACT_VERSION:
        raise DataError("native M05 telemetry contract mismatch")
    for key in ("schema_version", "contract_version", "payload_bytes", "threads", "maximum_source_row_bytes"):
        if isinstance(document[key], bool) or not isinstance(document[key], int):
            raise DataError(f"native M05 telemetry root field is not an integer: {key}")
    if (document["payload_bytes"] < 0 or document["threads"] < 1 or document["threads"] > 64
            or document["maximum_source_row_bytes"] < 0):
        raise DataError("native M05 telemetry root bounds are invalid")
    if document["payload_bytes"] != expected["payload_bytes"] or document["threads"] != expected["threads"]:
        raise DataError("native M05 telemetry job mismatch")
    native_build = _native_build(document["native_build"], "native M05 telemetry native_build")
    expected_max_row = max((int(matrix["columns"]) * 2 for matrix in expected["matrices"]), default=0)
    if document["maximum_source_row_bytes"] != expected_max_row:
        raise DataError("native M05 telemetry maximum source row mismatch")
    matrices = document["matrices"]
    if not isinstance(matrices, list) or len(matrices) != len(expected["matrices"]):
        raise DataError("native M05 telemetry matrix count mismatch")
    required_matrix = {
        "source_name", "weight_output_name", "scale_output_name", "rows", "columns", "elements",
        "source_sha256", "weight_sha256", "scale_sha256", "source_min", "source_max",
        "source_sum_squares", "reconstruction_sum_squares", "source_reconstruction_dot",
        "error_sum_squares", "max_absolute_error", "scale_min", "scale_max", "saturation_count",
        "zero_rows", "underflow_clamped_rows", "histogram",
    }
    integer_fields = {"rows", "columns", "elements", "saturation_count", "zero_rows", "underflow_clamped_rows"}
    float_fields = {"source_min", "source_max", "source_sum_squares", "reconstruction_sum_squares",
                    "source_reconstruction_dot", "error_sum_squares", "max_absolute_error", "scale_min", "scale_max"}
    result: dict[str, dict[str, Any]] = {}
    for item, expected_matrix in zip(matrices, expected["matrices"], strict=True):
        if not isinstance(item, dict) or set(item) != required_matrix:
            raise DataError("native M05 telemetry matrix schema mismatch")
        source_name = item["source_name"]
        if source_name != expected_matrix["source_name"]:
            raise DataError(f"native M05 telemetry matrix order/name mismatch: {source_name}")
        if (item["weight_output_name"] != expected_matrix["weight_output_name"] or
            item["scale_output_name"] != expected_matrix["scale_output_name"] or
            item["rows"] != expected_matrix["rows"] or item["columns"] != expected_matrix["columns"]):
            raise DataError(f"native M05 telemetry matrix identity mismatch: {source_name}")
        for key in integer_fields:
            if isinstance(item[key], bool) or not isinstance(item[key], int) or item[key] < 0:
                raise DataError(f"native M05 telemetry integer is invalid: {source_name}:{key}")
        for key in float_fields:
            if isinstance(item[key], bool) or not isinstance(item[key], (int, float)) or not math.isfinite(float(item[key])):
                raise DataError(f"native M05 telemetry scalar is invalid: {source_name}:{key}")
        elements = expected_matrix["rows"] * expected_matrix["columns"]
        if (item["elements"] != elements or item["saturation_count"] > elements
            or item["zero_rows"] > expected_matrix["rows"]
            or item["underflow_clamped_rows"] > expected_matrix["rows"]):
            raise DataError(f"native M05 telemetry counts are invalid: {source_name}")
        if (item["source_min"] > item["source_max"]
            or item["source_sum_squares"] < 0.0
            or item["reconstruction_sum_squares"] < 0.0
            or item["error_sum_squares"] < 0.0
            or item["max_absolute_error"] < 0.0
            or item["scale_min"] <= 0.0
            or item["scale_min"] > item["scale_max"]):
            raise DataError(f"native M05 telemetry numeric invariants are invalid: {source_name}")
        for key in ("source_sha256", "weight_sha256", "scale_sha256"):
            value = item[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise DataError(f"native M05 telemetry hash is invalid: {source_name}:{key}")
        if item["source_sha256"] != expected_matrix["source_sha256"]:
            raise SourceVerificationError(f"native M05 source hash mismatch: {source_name}")
        histogram = item["histogram"]
        if (not isinstance(histogram, list) or len(histogram) != 256 or
            any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in histogram) or
            sum(histogram) != elements):
            raise DataError(f"native M05 histogram is invalid: {source_name}")
        result[source_name] = item
    return result, int(document["maximum_source_row_bytes"]), native_build


def prepare_native_bundle(
    request: NativeRequest,
    plan: QuantizationPlan,
    source_tensors: dict[str, TensorDescriptor],
    workspace: BoundedWorkspace,
    staging: Path,
) -> NativeBundle:
    if request.timeout_seconds <= 0 or request.threads < 1 or request.threads > 64:
        raise DataError("invalid native M05 timeout or thread count")
    job, pairs, source_hashes = _make_job(plan, source_tensors, plan.output_tensor_bytes, request.threads)
    for weight, _scale in pairs:
        source_name = weight.source_names[0]
        if source_hashes[source_name]:
            continue
        source = source_tensors[source_name]
        resolved = source.path.expanduser().resolve(strict=True)
        source_hashes[source_name] = workspace.hash_tensor_range(
            resolved, source.absolute_offset, source.byte_length
        )
    for item in job["matrices"]:
        item["source_sha256"] = source_hashes[item["source_name"]]
    job_path = staging / ".m05_fp8_job.json"
    payload_path = staging / ".m05_fp8_payload.bin"
    telemetry_path = staging / ".m05_fp8_telemetry.json"
    for path in (job_path, payload_path, telemetry_path):
        if path.exists() or path.is_symlink():
            raise OutputError(f"native M05 temporary output already exists: {path}")
    staged_executable, binary_hash = _stage_executable(request.executable, staging, workspace)
    write_file_atomic(job_path, canonical_json_bytes(job))
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
                    raise RuntimeError(f"cannot apply native M05 address-space limit: {error}") from error

        process = subprocess.Popen(
            [str(staged_executable), "--job", str(job_path), "--payload", str(payload_path), "--telemetry", str(telemetry_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            close_fds=True,
            start_new_session=(os.name == "posix"),
            preexec_fn=limit_child if os.name == "posix" else None,
        )
        stdout_thread = threading.Thread(target=drain, args=(process.stdout, stdout_data), daemon=True)
        stderr_thread = threading.Thread(target=drain, args=(process.stderr, stderr_data), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        pgid = process.pid if os.name == "posix" else None
        deadline = time.monotonic() + request.timeout_seconds
        timed_out = False
        memory_exceeded = False
        while True:
            child_peak_rss = max(child_peak_rss, _process_group_rss_bytes(pgid) if pgid is not None else 0)
            leader_done = process.poll() is not None
            readers_done = not stdout_thread.is_alive() and not stderr_thread.is_alive()
            if child_peak_rss + workspace.max_observed_rss_bytes > workspace.host_memory_cap_bytes:
                memory_exceeded = True
                _terminate_process(process, pgid)
            elif overflow.is_set():
                _terminate_process(process, pgid)
            elif time.monotonic() >= deadline:
                timed_out = True
                _terminate_process(process, pgid)
            if (leader_done and readers_done) or time.monotonic() >= deadline + 2.0:
                if not leader_done or not readers_done:
                    _terminate_process(process, pgid)
                break
            time.sleep(0.02)
        process.wait()
        stdout_thread.join(timeout=2.0)
        stderr_thread.join(timeout=2.0)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        # A successful leader exit is not sufficient: detached descendants
        # can still hold descriptors or mutate the staged inputs.  Refuse to
        # publish until the complete process group is gone.
        if not timed_out and not memory_exceeded and not overflow.is_set() and pgid is not None:
            remaining = _process_group_pids(pgid)
            if remaining:
                gone = _terminate_process(process, pgid)
                suffix = "" if gone else "; process group survived SIGKILL wait"
                raise OutputError(
                    f"native M05 encoder left process-group descendants: {sorted(remaining)}{suffix}"
                )
        if timed_out:
            raise OutputError(f"native M05 encoder timed out after {request.timeout_seconds}s")
        if memory_exceeded:
            raise DataError("native M05 encoder exceeded combined host-memory cap")
        if overflow.is_set():
            raise OutputError("native M05 encoder diagnostics exceed safety limit")
        detail = bytes(stderr_data).decode("utf-8", errors="replace")
        if process.returncode != 0:
            message = f"native M05 encoder failed with exit {process.returncode}: {detail}"
            if process.returncode == InvalidPlanError.exit_code:
                raise InvalidPlanError(message)
            if process.returncode == SourceVerificationError.exit_code:
                raise SourceVerificationError(message)
            if process.returncode == DataError.exit_code:
                raise DataError(message)
            raise OutputError(message)
        if stdout_data:
            raise DataError("native M05 encoder wrote unexpected stdout")
        if not payload_path.is_file() or payload_path.is_symlink() or not telemetry_path.is_file() or telemetry_path.is_symlink():
            raise DataError("native M05 encoder did not publish payload and telemetry")
        if payload_path.stat().st_size != plan.output_tensor_bytes:
            raise DataError("native M05 payload has unexpected size")
        telemetry, max_row, native_build = _read_telemetry(telemetry_path, job)
        matrices: dict[str, NativeMatrix] = {}
        for item in job["matrices"]:
            source_name = item["source_name"]
            matrices[item["weight_output_name"]] = NativeMatrix(
                source_name, item["weight_output_name"], item["scale_output_name"],
                item["weight_offset"], item["weight_bytes"], item["scale_offset"], item["scale_bytes"],
                item["rows"], item["columns"], source_hashes[source_name], telemetry[source_name],
            )
            matrices[item["scale_output_name"]] = matrices[item["weight_output_name"]]
        workspace.record_transform_row(max_row, "native M05 source row")
        return NativeBundle(payload_path, telemetry_path, matrices, binary_hash, request.threads, max_row, child_peak_rss, native_build)
    except subprocess.SubprocessError as error:
        for path in (payload_path, telemetry_path, job_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise OutputError(f"native M05 encoder could not start: {error}") from error
    except OSError as error:
        for path in (payload_path, telemetry_path, job_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise OutputError(f"native M05 encoder process failed: {error}") from error
    except Exception:
        for path in (payload_path, telemetry_path, job_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        try:
            staged_executable.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            job_path.unlink(missing_ok=True)
        except OSError:
            pass


def _weight_statistics(item: dict[str, Any]) -> dict[str, object]:
    import math
    elements = int(item["elements"])
    source_energy = float(item["source_sum_squares"])
    error_energy = float(item["error_sum_squares"])
    reconstruction_energy = float(item["reconstruction_sum_squares"])
    dot = float(item["source_reconstruction_dot"])
    source_norm = math.sqrt(source_energy)
    reconstruction_norm = math.sqrt(reconstruction_energy)
    cosine = dot / (source_norm * reconstruction_norm) if source_norm and reconstruction_norm else (1.0 if source_energy == reconstruction_energy == 0.0 else 0.0)
    cosine = max(-1.0, min(1.0, cosine))
    result = {
        "contract_id": CONTRACT_ID, "contract_version": CONTRACT_VERSION, "component": "weight",
        "rows": item["rows"], "columns": item["columns"], "elements": elements,
        "source_min": item["source_min"], "source_max": item["source_max"],
        "source_rms": math.sqrt(source_energy / elements), "scale_min": item["scale_min"], "scale_max": item["scale_max"],
        "relative_l2_error": math.sqrt(error_energy / source_energy) if source_energy else 0.0,
        "cosine_similarity": cosine, "max_absolute_error": item["max_absolute_error"],
        "sqnr_db": None if error_energy == 0.0 or source_energy == 0.0 else 10.0 * math.log10(source_energy / error_energy),
        "perfect_reconstruction": error_energy == 0.0,
        "saturation_count": item["saturation_count"], "saturation_rate": item["saturation_count"] / elements,
        "zero_rows": item["zero_rows"], "underflow_clamped_rows": item["underflow_clamped_rows"],
        "histogram": item["histogram"],
    }
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise DataError(f"native M05 derived statistic is non-finite: {key}")
    return result


def _scale_statistics(item: dict[str, Any]) -> dict[str, object]:
    return {
        "contract_id": CONTRACT_ID, "contract_version": CONTRACT_VERSION, "component": "scale",
        "rows": item["rows"], "columns": 1, "scale_min": item["scale_min"], "scale_max": item["scale_max"],
        "zero_rows": item["zero_rows"], "underflow_clamped_rows": item["underflow_clamped_rows"],
    }


class NativeBundleEncoder:
    """Read-only adapter over one native bundle; no conversion occurs here."""
    def __init__(self, bundle: NativeBundle, component: str):
        self.bundle = bundle
        self.component = component
        self.name = "fp8-rowwise-weight-v1" if component == "weight" else "fp8-rowwise-scale-v1"
        self.version = 1

    def compile_tensor(self, plan: TensorCompilePlan, sources: tuple[TensorDescriptor, ...], output: BinaryIO, workspace: BoundedWorkspace) -> EncoderResult:
        if len(sources) != 1 or plan.output_name not in self.bundle.matrices:
            raise DataError(f"native M05 bundle lacks output: {plan.output_name}")
        entry = self.bundle.matrices[plan.output_name]
        if entry.source_name != sources[0].name:
            raise DataError(f"native M05 bundle source mismatch: {plan.output_name}")
        offset, length = (entry.weight_offset, entry.weight_bytes) if self.component == "weight" else (entry.scale_offset, entry.scale_bytes)
        _payload_hash, output_hash = workspace.copy_range(self.bundle.payload, offset, length, output, track_tensor=False)
        item = entry.telemetry
        expected_hash = item["weight_sha256"] if self.component == "weight" else item["scale_sha256"]
        if output_hash != expected_hash:
            raise DataError(f"native M05 {self.component} output hash mismatch: {plan.output_name}")
        return EncoderResult(
            source_sha256=(entry.source_sha256,), output_sha256=output_hash, output_bytes=length,
            statistics=_weight_statistics(item) if self.component == "weight" else _scale_statistics(item),
        )

    def cleanup(self) -> None:
        self.bundle.cleanup()


__all__ = ["NativeBundle", "NativeBundleEncoder", "NativeRequest", "PROTOCOL", "binary_sha256", "prepare_native_bundle"]
