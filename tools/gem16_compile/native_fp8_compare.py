"""Native M05 FP8 attention comparison protocol adapter.

The native comparator owns all elementwise work.  This module only constructs a
bounded descriptor job, supervises the process, validates its scalar protocol,
and projects the result into the historical comparison report.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any

from .common import (
    BoundedWorkspace, DataError, InvalidPlanError, OutputError,
    SourceVerificationError, canonical_json_bytes, reject_duplicate_keys,
    write_file_atomic,
)
from .native_fp8 import (
    MAX_DIAGNOSTIC_BYTES, _native_build,
    _process_group_pids, _process_group_rss_bytes, _regular_executable,
    _stage_executable, _terminate_process,
    binary_sha256,
)
from .reader import TensorDescriptor
from .profiles import M05_DEQUANTIZATION_EQUATION

COMPARE_PROTOCOL = "gem16-fp8-attention-compare-v1"
COMPARE_CONTRACT_ID = "gem16.fp8_attention_compare"
COMPARE_CONTRACT_VERSION = 1
MAX_COMPARE_MATRICES = 115
MAX_COMPARE_JSON_BYTES = 16 * 1024 * 1024
_HASH_FIELDS = (
    "left_weight_sha256", "right_weight_sha256",
    "left_scale_sha256", "right_scale_sha256",
)


def _hash(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise DataError(f"invalid SHA-256 at {where}")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DataError(f"invalid integer at {where}")
    if maximum is not None and value > maximum:
        raise DataError(f"integer exceeds bound at {where}")
    return value


def _finite(value: Any, where: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DataError(f"invalid finite number at {where}")
    result = float(value)
    if nonnegative and result < 0.0:
        raise DataError(f"negative number at {where}")
    return result


def _range(path: Path, descriptor: TensorDescriptor, workspace: BoundedWorkspace) -> dict[str, Any]:
    if descriptor.absolute_offset < 0 or descriptor.byte_length < 0:
        raise SourceVerificationError(f"invalid range for {descriptor.name}")
    try:
        resolved = path.expanduser().resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file():
            raise SourceVerificationError(f"comparison input is not a regular file: {path}")
        size = resolved.stat().st_size
    except OSError as error:
        raise SourceVerificationError(f"cannot inspect comparison input {path}: {error}") from error
    if descriptor.absolute_offset > size or descriptor.byte_length > size - descriptor.absolute_offset:
        raise SourceVerificationError(f"comparison range exceeds file: {descriptor.name}")
    return {
        "path": str(resolved),
        "offset": descriptor.absolute_offset,
        "bytes": descriptor.byte_length,
        "sha256": workspace.hash_range(resolved, descriptor.absolute_offset, descriptor.byte_length),
    }


def build_compare_job(left: dict[str, Any], right: dict[str, Any], workspace: BoundedWorkspace,
                      *, threads: int) -> tuple[dict[str, Any], list[tuple[Any, Any]]]:
    """Build the canonical native job from already validated attention pairs."""
    if threads < 1 or threads > 64:
        raise InvalidPlanError("comparison threads must be in the range 1..64")
    if len(left) != len(right) or not left:
        raise DataError("compiled and reference attention pair sets differ")
    if len(left) > MAX_COMPARE_MATRICES:
        raise DataError("comparison matrix count exceeds M05 bound")
    matrices: list[dict[str, Any]] = []
    pairs: list[tuple[Any, Any]] = []
    for name in sorted(left):
        pair = left[name]
        reference = right.get(name)
        if reference is None:
            raise DataError(f"reference lacks attention matrix: {name}")
        rows, columns = pair.weight.shape
        if tuple(reference.weight.shape) != (rows, columns) or tuple(pair.scale.shape) != (rows, 1) or tuple(reference.scale.shape) != (rows, 1):
            raise DataError(f"attention comparison shape mismatch: {name}")
        if pair.weight.dtype != "F8_E4M3" or reference.weight.dtype != "F8_E4M3" or pair.scale.dtype != "BF16" or reference.scale.dtype != "BF16":
            raise DataError(f"attention comparison dtype mismatch: {name}")
        if pair.weight.byte_length != rows * columns or reference.weight.byte_length != rows * columns:
            raise DataError(f"attention comparison weight byte mismatch: {name}")
        if pair.scale.byte_length != rows * 2 or reference.scale.byte_length != rows * 2:
            raise DataError(f"attention comparison scale byte mismatch: {name}")
        role = pair.role.removeprefix("attention_").removesuffix("_projection")
        matrices.append({
            "name": name,
            "layer": pair.layer,
            "role": role,
            "rows": rows,
            "columns": columns,
            "left_weight": _range(pair.weight.path, pair.weight, workspace),
            "left_scale": _range(pair.scale.path, pair.scale, workspace),
            "right_weight": _range(reference.weight.path, reference.weight, workspace),
            "right_scale": _range(reference.scale.path, reference.scale, workspace),
        })
        pairs.append((pair, reference))
    job = {
        "schema_version": 1,
        "contract_id": COMPARE_CONTRACT_ID,
        "contract_version": COMPARE_CONTRACT_VERSION,
        "threads": threads,
        "matrices": matrices,
    }
    if len(canonical_json_bytes(job)) > MAX_COMPARE_JSON_BYTES:
        raise DataError("native comparison job exceeds JSON bound")
    return job, pairs


def _read_metrics(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DataError(f"cannot read native comparison metrics: {error}") from error
    if len(raw) > MAX_COMPARE_JSON_BYTES:
        raise DataError("native comparison metrics exceed JSON bound")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataError(f"invalid native comparison metrics JSON: {error}") from error
    if not isinstance(document, dict):
        raise DataError("native comparison metrics root must be an object")
    required = {"schema_version", "contract_id", "contract_version", "threads", "maximum_chunk_bytes", "native_build", "matrices"}
    if set(document) != required or document.get("schema_version") != 1 or document.get("contract_id") != COMPARE_CONTRACT_ID or document.get("contract_version") != 1:
        raise DataError("native comparison metrics contract mismatch")
    _integer(document["threads"], "metrics.threads", minimum=1, maximum=64)
    try:
        expected_threads = expected["threads"]
    except (KeyError, TypeError) as error:
        raise DataError("comparison job thread count is missing") from error
    _integer(expected_threads, "job.threads", minimum=1, maximum=64)
    if document["threads"] != expected_threads:
        raise DataError("native comparison metrics thread mismatch")
    _integer(document["maximum_chunk_bytes"], "metrics.maximum_chunk_bytes", minimum=1, maximum=1 << 20)
    native_build = _native_build(document["native_build"], "native comparison metrics native_build")
    records = document["matrices"]
    try:
        expected_matrices = expected["matrices"]
    except (KeyError, TypeError) as error:
        raise DataError("comparison job matrices are missing") from error
    if not isinstance(records, list) or not isinstance(expected_matrices, list) or len(records) != len(expected_matrices):
        raise DataError("native comparison metrics matrix count mismatch")
    result: list[dict[str, Any]] = []
    # Keep this list exact: accepting additional fields would make the raw
    # protocol ambiguous and makes provenance harder to audit.
    keys = {
        "name", "layer", "role", "rows", "columns", "elements", *_HASH_FIELDS,
        "raw_mismatch_count", "left_endpoint_7e", "left_endpoint_fe", "right_endpoint_7e", "right_endpoint_fe", "left_nan_count", "right_nan_count",
        "scale_mismatch_count", "left_scale_min", "left_scale_max", "right_scale_min", "right_scale_max", "left_scale_sum", "right_scale_sum", "left_scale_sum_squares", "right_scale_sum_squares", "scale_difference_sum_squares", "scale_dot", "scale_relative_l2", "scale_pearson_correlation",
        "left_min", "left_max", "left_sum_squares", "right_min", "right_max", "right_sum_squares", "difference_sum_squares", "reconstruction_dot", "reconstruction_relative_l2", "cosine_similarity", "max_absolute_error", "sqnr_db", "perfect_reconstruction", "zero_reference",
    }
    integers = {"layer", "rows", "columns", "elements", "raw_mismatch_count", "left_endpoint_7e", "left_endpoint_fe", "right_endpoint_7e", "right_endpoint_fe", "left_nan_count", "right_nan_count", "scale_mismatch_count"}
    count_fields = {"raw_mismatch_count", "left_endpoint_7e", "left_endpoint_fe", "right_endpoint_7e", "right_endpoint_fe", "left_nan_count", "right_nan_count", "scale_mismatch_count"}
    nonnegative = {"left_sum_squares", "right_sum_squares", "difference_sum_squares", "max_absolute_error"}
    floats = keys - integers - {"name", "role", "perfect_reconstruction", "zero_reference", *_HASH_FIELDS}
    for index, (record, wanted) in enumerate(zip(records, expected_matrices, strict=True)):
        if not isinstance(record, dict) or set(record) != keys:
            raise DataError(f"native comparison matrix schema mismatch at {index}")
        try:
            identity = (wanted["name"], wanted["layer"], wanted["role"], wanted["rows"], wanted["columns"])
        except (KeyError, TypeError) as error:
            raise DataError(f"comparison job matrix is malformed at {index}") from error
        if (record["name"], record["layer"], record["role"], record["rows"], record["columns"]) != identity:
            raise DataError(f"native comparison matrix identity mismatch at {index}")
        for key in integers:
            _integer(record[key], f"metrics[{index}].{key}")
        elements = record["rows"] * record["columns"]
        if record["elements"] != elements:
            raise DataError(f"native comparison element count mismatch at {index}")
        for key in count_fields:
            if record[key] > elements:
                raise DataError(f"native comparison count exceeds elements at {index}.{key}")
        for key in nonnegative:
            if record[key] < 0:
                raise DataError(f"native comparison value is negative at {index}.{key}")
        if (record["left_endpoint_7e"] + record["left_endpoint_fe"] > elements
                or record["right_endpoint_7e"] + record["right_endpoint_fe"] > elements
                or record["scale_mismatch_count"] > record["rows"]):
            raise DataError(f"native comparison endpoint/count total exceeds elements at {index}")
        for key in floats:
            if record[key] is None:
                if key not in {"reconstruction_relative_l2", "scale_relative_l2", "sqnr_db"}:
                    raise DataError(f"unexpected null metric at {index}.{key}")
            else:
                _finite(record[key], f"metrics[{index}].{key}", nonnegative=key in {"left_scale_min", "right_scale_min", "left_scale_max", "right_scale_max", "left_sum_squares", "right_sum_squares", "difference_sum_squares", "max_absolute_error"})
        if (record["left_scale_min"] <= 0 or record["right_scale_min"] <= 0
                or record["left_scale_min"] > record["left_scale_max"]
                or record["right_scale_min"] > record["right_scale_max"]
                or record["left_scale_sum"] <= 0 or record["right_scale_sum"] <= 0
                or record["left_scale_sum_squares"] <= 0 or record["right_scale_sum_squares"] <= 0
                or record["scale_difference_sum_squares"] < 0
                or record["left_min"] > record["left_max"]
                or record["right_min"] > record["right_max"]
                or record["scale_pearson_correlation"] < -1
                or record["scale_pearson_correlation"] > 1
                or record["cosine_similarity"] < -1
                or record["cosine_similarity"] > 1):
            raise DataError(f"invalid native comparison numeric invariants at {index}")
        if not isinstance(record["perfect_reconstruction"], bool) or not isinstance(record["zero_reference"], bool):
            raise DataError(f"invalid metric flags at {index}")
        expected_ranges = {
            "left_weight_sha256": "left_weight",
            "right_weight_sha256": "right_weight",
            "left_scale_sha256": "left_scale",
            "right_scale_sha256": "right_scale",
        }
        for key, range_name in expected_ranges.items():
            try:
                expected_range = wanted[range_name]
                expected_hash = expected_range["sha256"]
            except (KeyError, TypeError) as error:
                raise DataError(f"comparison job range is malformed at {index}.{range_name}") from error
            if _hash(record[key], f"metrics[{index}].{key}") != _hash(expected_hash, f"job[{index}].{range_name}.sha256"):
                raise SourceVerificationError(f"native comparison range hash mismatch at {index}.{key}")
        result.append(record)
    return {**document, "native_build": native_build, "matrices": result}


def run_native_compare(executable: Path, job: dict[str, Any], workspace: BoundedWorkspace,
                       staging_parent: Path, *, timeout_seconds: int) -> tuple[dict[str, Any], str, dict[str, str]]:
    if timeout_seconds <= 0:
        raise InvalidPlanError("native comparison timeout must be positive")
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".incomplete-compare-", dir=staging_parent) as raw:
        staging = Path(raw)
        staged, binary_hash = _stage_executable(_regular_executable(executable), staging, workspace)
        job_path = staging / "job.json"
        metrics_path = staging / "metrics.json"
        write_file_atomic(job_path, canonical_json_bytes(job))
        stdout = bytearray(); stderr = bytearray(); overflow = threading.Event()
        def drain(stream: Any, target: bytearray) -> None:
            try:
                while True:
                    block = stream.read(65536)
                    if not block:
                        return
                    if len(target) + len(block) > MAX_DIAGNOSTIC_BYTES:
                        overflow.set(); return
                    target.extend(block)
            except OSError:
                overflow.set()
        env = os.environ.copy(); env["LC_ALL"] = "C.UTF-8"; env["LANG"] = "C.UTF-8"
        available = max(64 * 1024 * 1024, workspace.host_memory_cap_bytes - workspace.baseline_rss_bytes)
        def limit_child() -> None:
            if os.name == "posix":
                try:
                    import resource
                    resource.setrlimit(resource.RLIMIT_AS, (available, available))
                except (ImportError, OSError, ValueError) as error:
                    raise RuntimeError(f"cannot apply native comparison address-space limit: {error}") from error
        try:
            process = subprocess.Popen(
                [str(staged), "--compare-job", str(job_path), "--metrics", str(metrics_path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env, close_fds=True, start_new_session=(os.name == "posix"),
                preexec_fn=limit_child if os.name == "posix" else None,
            )
        except OSError as error:
            raise OutputError(f"cannot start native comparison: {error}") from error
        out_thread = threading.Thread(target=drain, args=(process.stdout, stdout), daemon=True)
        err_thread = threading.Thread(target=drain, args=(process.stderr, stderr), daemon=True)
        out_thread.start(); err_thread.start()
        pgid = process.pid if os.name == "posix" else None
        deadline = time.monotonic() + timeout_seconds
        memory_exceeded = False
        try:
            while process.poll() is None or out_thread.is_alive() or err_thread.is_alive():
                if pgid is not None and _process_group_rss_bytes(pgid) + workspace.max_observed_rss_bytes > workspace.host_memory_cap_bytes:
                    memory_exceeded = True
                    _terminate_process(process, pgid)
                if overflow.is_set() or time.monotonic() >= deadline:
                    _terminate_process(process, pgid)
                    if overflow.is_set():
                        raise OutputError("native comparison diagnostics exceed safety limit")
                    raise OutputError(f"native comparison timed out after {timeout_seconds}s")
                if memory_exceeded:
                    raise DataError("native comparison exceeded combined host-memory cap")
                workspace.check("native comparison")
                time.sleep(0.02)
            process.wait()
            out_thread.join(timeout=2); err_thread.join(timeout=2)
            if process.returncode == 0 and pgid is not None:
                remaining = _process_group_pids(pgid)
                if remaining:
                    gone = _terminate_process(process, pgid)
                    suffix = "" if gone else "; process group survived SIGKILL wait"
                    raise OutputError(
                        f"native comparison left process-group descendants: {sorted(remaining)}{suffix}"
                    )
        finally:
            if process.poll() is None:
                _terminate_process(process, pgid)
        if stdout:
            raise DataError("native comparison wrote unexpected stdout")
        if process.returncode != 0:
            detail = bytes(stderr).decode("utf-8", errors="replace")
            message = f"native comparison failed with exit {process.returncode}: {detail}"
            if process.returncode == 2: raise InvalidPlanError(message)
            if process.returncode == 3: raise SourceVerificationError(message)
            if process.returncode == 4: raise DataError(message)
            raise OutputError(message)
        if not metrics_path.is_file() or metrics_path.is_symlink():
            raise DataError("native comparison did not publish metrics")
        metrics = _read_metrics(metrics_path, job)
        return metrics, binary_hash, metrics["native_build"]


def project_report(raw: dict[str, Any], pairs: list[tuple[Any, Any]], *, binary_hash: str,
                   native_build: dict[str, str], threads: int, manifest: dict[str, Any], manifest_hash: str,
                   source: Any, tensor_count: int, staging_bytes: int,
                   production: bool) -> dict[str, Any]:
    """Project native scalar records without touching tensor payload bytes."""
    native_build = _native_build(native_build, "native comparator build")
    reports: list[dict[str, Any]] = []
    for item, (pair, right) in zip(raw["matrices"], pairs, strict=True):
        rows, columns = item["rows"], item["columns"]
        reports.append({
            "name": item["name"], "layer": item["layer"], "role": item["role"],
            "weight": {"left_name": pair.weight.name, "right_name": right.weight.name, "left_dtype": pair.weight.dtype, "right_dtype": right.weight.dtype, "left_shard": pair.weight.shard, "right_shard": right.weight.shard, "left_data_offset": pair.weight.data_offset, "right_data_offset": right.weight.data_offset, "shape": [rows, columns], "left_bytes": rows * columns, "right_bytes": rows * columns, "left_sha256": item["left_weight_sha256"], "right_sha256": item["right_weight_sha256"], "raw_mismatch_count": item["raw_mismatch_count"], "raw_mismatch_rate": item["raw_mismatch_count"] / item["elements"], "left_endpoint_code_counts": {"0x7e": item["left_endpoint_7e"], "0xfe": item["left_endpoint_fe"]}, "right_endpoint_code_counts": {"0x7e": item["right_endpoint_7e"], "0xfe": item["right_endpoint_fe"]}, "left_nan_code_count": item["left_nan_count"], "right_nan_code_count": item["right_nan_count"]},
            "scale": {"left_name": pair.scale.name, "right_name": right.scale.name, "left_dtype": pair.scale.dtype, "right_dtype": right.scale.dtype, "left_shard": pair.scale.shard, "right_shard": right.scale.shard, "left_data_offset": pair.scale.data_offset, "right_data_offset": right.scale.data_offset, "shape": [rows, 1], "left_bytes": rows * 2, "right_bytes": rows * 2, "left_sha256": item["left_scale_sha256"], "right_sha256": item["right_scale_sha256"], "mismatch_count": item["scale_mismatch_count"], "left_min": item["left_scale_min"], "left_max": item["left_scale_max"], "right_min": item["right_scale_min"], "right_max": item["right_scale_max"], "left_sum": item["left_scale_sum"], "right_sum": item["right_scale_sum"], "left_sum_squares": item["left_scale_sum_squares"], "right_sum_squares": item["right_scale_sum_squares"], "difference_sum_squares": item["scale_difference_sum_squares"], "dot": item["scale_dot"], "relative_l2": item["scale_relative_l2"], "pearson_correlation": item["scale_pearson_correlation"]},
            "reconstruction": {"left_min": item["left_min"], "left_max": item["left_max"], "left_rms": math.sqrt(item["left_sum_squares"] / item["elements"]), "left_sum_squares": item["left_sum_squares"], "right_min": item["right_min"], "right_max": item["right_max"], "right_rms": math.sqrt(item["right_sum_squares"] / item["elements"]), "right_sum_squares": item["right_sum_squares"], "difference_sum_squares": item["difference_sum_squares"], "dot": item["reconstruction_dot"], "relative_l2": item["reconstruction_relative_l2"], "cosine_similarity": item["cosine_similarity"], "max_absolute_error": item["max_absolute_error"], "sqnr_db": item["sqnr_db"], "perfect_reconstruction": item["perfect_reconstruction"], "zero_reference": item["zero_reference"]},
        })
    # Reuse the existing scalar-only aggregate reducer; it never reads tensor bytes.
    from .fp8_report import _aggregate
    aggregates = _aggregate(reports)
    compiled_record = manifest.get("compiler", {}); plan_record = manifest.get("plan", {}); source_record = manifest.get("source", {})
    return {"schema_version": 1, "family": "attention", "status": "pass", "artifact_profile": "fp8-attention-partial-v1", "artifact_status": "m05_fp8_attention_partial_not_runtime_loadable", "source": {"unsloth_lock_sha256": source.lock_sha256, "repository": source.repository, "revision": source.revision, "tensor_count": tensor_count}, "compiled": {"compilation_manifest_sha256": manifest_hash, "tensor_count": len(manifest.get("tensors", [])), "source_lock_sha256": source_record.get("lock_sha256"), "source_contract": plan_record.get("source_contract"), "compiler_commit": compiled_record.get("commit"), "compiler_dirty": compiled_record.get("dirty"), "compiler_implementation": compiled_record.get("implementation"), "compiler_manifest_sha256": plan_record.get("compiler_manifest_sha256"), "resolved_plan_sha256": plan_record.get("resolved_plan_sha256")}, "contract": {"comparison_schema_version": 1, "weight_dtype": "F8_E4M3", "scale_dtype": "BF16", "codec": "E4M3FN", "dequantization_equation": M05_DEQUANTIZATION_EQUATION, "operator_output_comparison": "not_performed_in_M05_weight_comparison", "accumulation": "binary64_neumaier_left_to_right", "scale_mismatch_unit": "BF16 row bit-pattern", "native_comparator_protocol": COMPARE_PROTOCOL, "native_comparator_sha256": binary_hash, "native_comparator_threads": threads, "native_comparator_build": native_build, "reference_environment": {"byteorder": "little", "system": "Linux", "machine": "x86_64"}}, "matrices": reports, "aggregates": aggregates, "memory": {"staging_buffer_bytes": staging_bytes, "maximum_chunk_bytes": raw["maximum_chunk_bytes"], "maximum_transform_row_bytes": raw["maximum_chunk_bytes"] * 2, "accumulation": "binary64_neumaier_left_to_right", "report_excludes_rss": True}, "limitations": ["This report compares stored weights and row scales only.", "No activation or CUDA operator-output comparison is performed.", "Differences are not model-quality or QAT attribution evidence."]}

__all__ = ["build_compare_job", "run_native_compare", "project_report"]
