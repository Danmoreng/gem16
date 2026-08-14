#!/usr/bin/env python3
"""Generate deterministic M05 reports from optional ignored raw evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

try:
    from tools.gem16_compile.common import OutputError, canonical_json_bytes, write_file_atomic
    from tools.gem16_compile.profiles import (
        M05_ATTENTION_TABLE, M05_GLOBAL_LAYERS, M05_PROFILE, M05_QUANTIZER_PARAMETERS,
        M05_SOURCE_CONTRACT, M05_SOURCE_LOCK_SHA256,
    )
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from gem16_compile.common import OutputError, canonical_json_bytes, write_file_atomic  # type: ignore[no-redef]
    from gem16_compile.profiles import (  # type: ignore[no-redef]
        M05_ATTENTION_TABLE, M05_GLOBAL_LAYERS, M05_PROFILE, M05_QUANTIZER_PARAMETERS,
        M05_SOURCE_CONTRACT, M05_SOURCE_LOCK_SHA256,
    )

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_ROOT = ROOT / "artifacts/raw/m05"
DEFAULT_ORDINARY = DEFAULT_RAW_ROOT / "ordinary-compile.json"
DEFAULT_QAT = DEFAULT_RAW_ROOT / "qat-compile.json"
DEFAULT_ORDINARY_MANIFEST = DEFAULT_RAW_ROOT / "ordinary-gem16-compilation-clean.json"
DEFAULT_QAT_MANIFEST = DEFAULT_RAW_ROOT / "qat-gem16-compilation-clean.json"
DEFAULT_CONFIG = ROOT / "artifacts/m05/fp8-compiler-config.json"
DEFAULT_ORDINARY_PLAN = ROOT / "benchmarks/goldens/gemma4_26b/fp8/ordinary-compiler-plan.json"
DEFAULT_QAT_PLAN = ROOT / "benchmarks/goldens/gemma4_26b/fp8/qat-compiler-plan.json"
DEFAULT_SPEC = ROOT / "tools/gem16_compile/specs/fp8-attention-rowwise-v1.json"
DEFAULT_TENSOR_REPORT = DEFAULT_RAW_ROOT / "fp8-tensor-report.json"
DEFAULT_QAT_SUMMARY = ROOT / "artifacts/m05/qat-fp8-summary.json"

EXPECTED_OUTPUT_BYTES = 1_110_850_560
EXPECTED_WEIGHT_BYTES = 1_110_179_840
EXPECTED_SCALE_BYTES = 670_720
EXPECTED_TENSOR_COUNT = 230
EXPECTED_MATRIX_COUNT = 115
EXPECTED_HISTOGRAM_BINS = 256
HEX64 = set("0123456789abcdef")
WEIGHT_REQUIRED = {
    "columns", "component", "contract_id", "contract_version", "cosine_similarity",
    "elements", "histogram", "max_absolute_error", "perfect_reconstruction",
    "relative_l2_error", "rows", "saturation_count", "saturation_rate", "scale_max",
    "scale_min", "source_max", "source_min", "source_rms", "sqnr_db",
    "underflow_clamped_rows", "zero_rows",
}
SCALE_REQUIRED = {
    "columns", "component", "contract_id", "contract_version", "rows", "scale_max",
    "scale_min", "underflow_clamped_rows", "zero_rows",
}


class ReportError(ValueError):
    """Raised when a compiler report is not an accepted M05 diagnostic report."""


def _fail(message: str) -> ReportError:
    return ReportError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _fail(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise _fail(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise _fail(f"cannot hash {path}: {error}") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _fail(message)


def _integer(value: Any, description: str, *, minimum: int = 0) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
             f"invalid non-negative integer {description}")
    return value


def _finite(value: Any, description: str, *, allow_null: bool = False) -> None:
    if value is None:
        _require(allow_null, f"unexpected null {description}")
        return
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"invalid numeric value {description}")
    if isinstance(value, float):
        _require(math.isfinite(value), f"non-finite numeric value {description}")


def _hex64(value: Any, description: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and set(value) <= HEX64,
             f"invalid SHA-256 {description}")
    return value


def _hex40(value: Any, description: str) -> str:
    _require(isinstance(value, str) and len(value) == 40 and set(value) <= HEX64,
             f"invalid commit hash {description}")
    return value


def _shape(value: Any, description: str) -> tuple[int, int]:
    _require(isinstance(value, list) and len(value) == 2, f"invalid shape {description}")
    return (_integer(value[0], f"{description}[0]", minimum=1),
            _integer(value[1], f"{description}[1]", minimum=1))


def _validate_number_fields(stats: dict[str, Any], fields: Iterable[str], label: str,
                            *, nullable: frozenset[str] = frozenset()) -> None:
    for field in fields:
        _finite(stats.get(field), f"{label}.{field}", allow_null=field in nullable)


def _validate_statistics(entry: dict[str, Any], expected_component: str,
                         expected_shape: tuple[int, int]) -> dict[str, Any]:
    stats = entry.get("statistics")
    _require(isinstance(stats, dict), f"statistics is not an object: {entry.get('output_name')}")
    required = WEIGHT_REQUIRED if expected_component == "weight" else SCALE_REQUIRED
    _require(set(stats) == required, f"statistics schema mismatch: {entry.get('output_name')}")
    _require(stats["component"] == expected_component, f"wrong component: {entry.get('output_name')}")
    _require(stats["contract_id"] == "gem16.fp8_attention_rowwise" and stats["contract_version"] == 1,
             f"wrong statistics contract: {entry.get('output_name')}")
    rows, columns = expected_shape
    _integer(stats["rows"], f"{entry['output_name']}.rows", minimum=1)
    _integer(stats["columns"], f"{entry['output_name']}.columns", minimum=1)
    _require(stats["rows"] == rows and stats["columns"] == columns,
             f"statistics shape mismatch: {entry.get('output_name')}")
    if expected_component == "weight":
        elements = _integer(stats["elements"], f"{entry['output_name']}.elements", minimum=1)
        _require(elements == rows * columns, f"element count mismatch: {entry['output_name']}")
        histogram = stats["histogram"]
        _require(isinstance(histogram, list) and len(histogram) == EXPECTED_HISTOGRAM_BINS,
                 f"histogram schema mismatch: {entry['output_name']}")
        histogram_total = sum(_integer(item, f"{entry['output_name']}.histogram", minimum=0)
                              for item in histogram)
        _require(histogram_total == elements, f"histogram total mismatch: {entry['output_name']}")
        for field in ("saturation_count", "underflow_clamped_rows", "zero_rows"):
            _require(_integer(stats[field], f"{entry['output_name']}.{field}", minimum=0) <= rows,
                     f"row count exceeds rows: {entry['output_name']}.{field}")
        _require(0.0 <= stats["saturation_rate"] <= 1.0,
                 f"invalid saturation rate: {entry['output_name']}")
        _validate_number_fields(stats, (
            "cosine_similarity", "max_absolute_error", "relative_l2_error", "saturation_rate",
            "scale_max", "scale_min", "source_max", "source_min", "source_rms",
            "sqnr_db",), entry["output_name"], nullable=frozenset({"sqnr_db"}))
        _require(isinstance(stats["perfect_reconstruction"], bool),
                 f"invalid reconstruction sentinel: {entry['output_name']}")
        _require(stats["source_rms"] >= 0.0 and stats["scale_min"] >= 0.0 and stats["scale_max"] >= stats["scale_min"],
                 f"invalid source/scale range: {entry['output_name']}")
    else:
        for field in ("underflow_clamped_rows", "zero_rows"):
            _require(_integer(stats[field], f"{entry['output_name']}.{field}", minimum=0) <= rows,
                     f"row count exceeds rows: {entry['output_name']}.{field}")
        _validate_number_fields(stats, ("scale_max", "scale_min"), entry["output_name"])
        _require(stats["scale_min"] >= 0.0 and stats["scale_max"] >= stats["scale_min"],
                 f"invalid scale range: {entry['output_name']}")
    return stats


def _validate_config(config: dict[str, Any], spec_path: Path,
                     plan_paths: dict[str, Path]) -> None:
    _require(config.get("schema_version") == 1 and
             config.get("artifact_profile") == M05_PROFILE.name and
             config.get("source_contract") == M05_SOURCE_CONTRACT,
             "M05 compiler config profile/contract mismatch")
    quantizer = config.get("quantizer")
    _require(isinstance(quantizer, dict) and
             quantizer.get("path") == str(spec_path.relative_to(ROOT)) and
             quantizer.get("sha256") == _sha256(spec_path),
             "M05 quantizer config does not match the checked specification")
    attention = config.get("attention")
    _require(isinstance(attention, dict) and attention.get("matrix_count") == EXPECTED_MATRIX_COUNT and
             attention.get("output_tensor_count") == EXPECTED_TENSOR_COUNT and
             attention.get("output_tensor_bytes") == EXPECTED_OUTPUT_BYTES and
             attention.get("global_layers_without_v") == sorted(M05_GLOBAL_LAYERS),
             "M05 config attention totals do not match the frozen contract")
    spec = _load(spec_path)
    _require(spec.get("contract_id") == "gem16.fp8_attention_rowwise" and
             spec.get("version") == 1 and
             spec.get("histogram", {}).get("bins") == EXPECTED_HISTOGRAM_BINS,
             "M05 quantizer specification contract mismatch")
    plans = config.get("plans")
    _require(isinstance(plans, dict), "M05 compiler config plans are missing")
    for key, path in plan_paths.items():
        record = plans.get(key)
        _require(isinstance(record, dict) and record.get("path") == str(path.relative_to(ROOT)) and
                 record.get("sha256") == _sha256(path),
                 f"M05 config plan record does not match {key}")


def _expected_entries() -> dict[str, tuple[str, tuple[int, int]]]:
    result: dict[str, tuple[str, tuple[int, int]]] = {}
    for source_name, spec in M05_ATTENTION_TABLE.items():
        stem = source_name.removesuffix(".weight")
        result[stem + ".weight"] = ("weight", spec.shape)
        result[stem + ".weight_scale"] = ("scale", (spec.shape[0], 1))
    return result


def _validate_lane(report: dict[str, Any], report_path: Path, lane: str,
                   expected_lock: str, expected_plan: Path,
                   retained_manifest_path: Path,
                   config: dict[str, Any]) -> dict[str, Any]:
    _require(report.get("action") == "compile" and report.get("status") == "pass",
             f"{lane} report is not a successful compile")
    _require(report.get("artifact_profile") == M05_PROFILE.name,
             f"{lane} report has the wrong profile")
    _require(isinstance(report.get("compiler_dirty"), bool),
             f"{lane} compiler dirty provenance is not boolean")
    _require(report.get("source_lock_sha256") == expected_lock,
             f"{lane} source lock is not approved")
    compile_report_sha256 = _sha256(report_path)
    _hex64(expected_lock, f"{lane} source lock")
    _hex64(report.get("compiler_manifest_sha256"), f"{lane} compiler plan")
    _hex64(report.get("resolved_plan_sha256"), f"{lane} resolved plan")
    compiler_commit = _hex40(report.get("compiler_commit"), f"{lane} compiler commit")
    compiler_dirty = report.get("compiler_dirty")
    _require(report.get("output_tensor_count") == EXPECTED_TENSOR_COUNT and
             report.get("output_tensor_bytes") == EXPECTED_OUTPUT_BYTES,
             f"{lane} output totals are not the exact M05 totals")
    native = report.get("native_encoder")
    _require(isinstance(native, dict), f"{lane} native encoder provenance is malformed")
    _require(native.get("protocol") == "gem16-fp8-batch-v1" and native.get("threads") == 8,
             f"{lane} native encoder protocol/thread mismatch")
    native_hash = _hex64(native.get("sha256"), f"{lane} native encoder")
    build = native.get("build")
    _require(isinstance(build, dict), f"{lane} native build is malformed")
    _require(set(build) == {"build_type", "compiler_id", "compiler_version", "cxx_standard", "processor", "system"},
             f"{lane} native build schema mismatch")
    _require(build["build_type"] == "Release" and build["cxx_standard"] == "20" and
             build["system"] == "Linux" and build["processor"] == "x86_64",
             f"{lane} native build identity mismatch")
    _require(report.get("native_build") == build, f"{lane} native build records disagree")
    plan_key = "ordinary_bf16" if lane == "ordinary" else "qat_bf16"
    config_plan = config.get("plans", {}).get(plan_key)
    _require(isinstance(config_plan, dict), f"missing {lane} plan config")
    plan_hash = _sha256(expected_plan)
    _require(config_plan.get("path") == str(expected_plan.relative_to(ROOT)) and
             config_plan.get("sha256") == plan_hash and
             report.get("compiler_manifest_sha256") == plan_hash and
             report.get("resolved_plan_sha256") == plan_hash,
             f"{lane} plan hash/path mismatch")
    output = report.get("output")
    _require(isinstance(output, str) and output, f"{lane} output path is missing")
    compilation_manifest = retained_manifest_path
    compilation_manifest_sha256 = _sha256(compilation_manifest)
    _hex64(report.get("compilation_manifest_sha256"), f"{lane} compilation manifest")
    _hex64(compile_report_sha256, f"{lane} compile report")
    _require(report.get("compilation_manifest_sha256") == compilation_manifest_sha256,
             f"{lane} compilation manifest hash mismatch")
    manifest = _load(compilation_manifest)
    manifest_plan = manifest.get("plan")
    manifest_compiler = manifest.get("compiler")
    _require(isinstance(manifest_plan, dict) and isinstance(manifest_compiler, dict),
             f"{lane} compilation manifest provenance is malformed")
    _hex64(manifest_plan.get("compiler_manifest_sha256"), f"{lane} manifest compiler plan")
    _hex64(manifest_plan.get("resolved_plan_sha256"), f"{lane} manifest resolved plan")
    _require(manifest_plan.get("compiler_manifest_sha256") == plan_hash and
             manifest_plan.get("resolved_plan_sha256") == plan_hash and
             manifest_compiler.get("commit") == compiler_commit and
             manifest_compiler.get("dirty") is compiler_dirty,
             f"{lane} compilation manifest provenance mismatch")
    _require(config_plan.get("source_lock_sha256") == expected_lock,
             f"{lane} config source lock mismatch")
    stats_entries = report.get("fp8_tensor_statistics")
    _require(isinstance(stats_entries, list) and len(stats_entries) == EXPECTED_TENSOR_COUNT,
             f"{lane} statistics count mismatch")
    expected = _expected_entries()
    by_name: dict[str, dict[str, Any]] = {}
    for entry in stats_entries:
        _require(isinstance(entry, dict), f"{lane} tensor statistic is not an object")
        name = entry.get("output_name")
        _require(name in expected and name not in by_name, f"{lane} unexpected/duplicate output {name}")
        component, shape = expected[name]
        _require(entry.get("logical_dtype") == "BF16" and entry.get("logical_shape") == list(shape),
                 f"{lane} logical shape/dtype mismatch: {name}")
        _require(entry.get("operation_id") == "fp8-attention:" + name.removesuffix(".weight_scale").removesuffix(".weight"),
                 f"{lane} operation identity mismatch: {name}")
        _require(entry.get("quantizer_parameters") == M05_QUANTIZER_PARAMETERS,
                 f"{lane} quantizer contract mismatch: {name}")
        source_names = entry.get("source_names")
        source_hashes = entry.get("source_sha256")
        expected_source_name = (name if component == "weight" else
                                name.removesuffix(".weight_scale") + ".weight")
        _require(isinstance(source_names, list) and len(source_names) == 1 and
                 source_names[0] == expected_source_name,
                 f"{lane} source name mismatch: {name}")
        _require(isinstance(source_hashes, list) and len(source_hashes) == 1, f"{lane} source hash schema: {name}")
        _hex64(source_hashes[0], f"{lane} source tensor {name}")
        _validate_statistics(entry, component, shape)
        by_name[name] = entry
    _require(set(by_name) == set(expected), f"{lane} does not cover the frozen attention table")
    for source_name in M05_ATTENTION_TABLE:
        _require(not (source_name.endswith(".v_proj.weight") and
                      int(source_name.split(".layers.")[1].split(".", 1)[0]) in M05_GLOBAL_LAYERS),
                 "frozen table unexpectedly contains a global V")
        weight = by_name[source_name]
        scale = by_name[source_name.removesuffix(".weight") + ".weight_scale"]
        _require(weight["source_names"] == scale["source_names"] and
                 weight["source_sha256"] == scale["source_sha256"],
                 f"{lane} weight/scale source pairing mismatch: {source_name}")
    return {"report": report, "by_name": by_name, "native_hash": native_hash, "build": build,
            "plan_hash": plan_hash, "source_lock_sha256": expected_lock,
            "compiler_plan_sha256": report["compiler_manifest_sha256"],
            "resolved_plan_sha256": report["resolved_plan_sha256"],
            "compilation_manifest_sha256": compilation_manifest_sha256,
            "compile_report_sha256": compile_report_sha256,
            "compiler_commit": compiler_commit, "compiler_dirty": compiler_dirty}


def _copy_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(stats)


def _matrix_records(lanes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordinary = lanes["ordinary"]["by_name"]
    qat = lanes["qat"]["by_name"]
    records: list[dict[str, Any]] = []
    for source_name in sorted(M05_ATTENTION_TABLE):
        spec = M05_ATTENTION_TABLE[source_name]
        stem = source_name.removesuffix(".weight")
        role = source_name.split(".self_attn.", 1)[1].removesuffix("_proj.weight")
        weight_name, scale_name = stem + ".weight", stem + ".weight_scale"
        records.append({
            "source_name": source_name,
            "source_sha256_by_lane": {
                "ordinary_bf16": ordinary[weight_name]["source_sha256"][0],
                "qat_bf16": qat[weight_name]["source_sha256"][0],
            },
            "output_names": {"weight": weight_name, "scale": scale_name},
            "role": f"attention_{role}_projection",
            "shape": list(spec.shape),
            "ordinary": {"weight": _copy_stats(ordinary[weight_name]["statistics"]),
                         "scale": _copy_stats(ordinary[scale_name]["statistics"])},
            "qat": {"weight": _copy_stats(qat[weight_name]["statistics"]),
                    "scale": _copy_stats(qat[scale_name]["statistics"])},
        })
    return records


def _totals(records: list[dict[str, Any]]) -> dict[str, int]:
    weight_elements = sum(item["ordinary"]["weight"]["elements"] for item in records)
    scale_elements = sum(item["ordinary"]["scale"]["rows"] for item in records)
    return {
        "matrix_count": len(records), "output_tensor_count": len(records) * 2,
        "weight_elements": weight_elements, "weight_bytes": weight_elements,
        "scale_elements": scale_elements, "scale_bytes": scale_elements * 2,
        "output_tensor_bytes": weight_elements + scale_elements * 2,
    }


def _provenance_record(lane: dict[str, Any]) -> dict[str, Any]:
    return {key: lane[key] for key in (
        "source_lock_sha256", "compiler_plan_sha256", "resolved_plan_sha256",
        "compilation_manifest_sha256", "compile_report_sha256",
        "compiler_commit", "compiler_dirty",
    )}


def _evidence_class(lanes: dict[str, dict[str, Any]]) -> tuple[str, str]:
    dirty = lanes["ordinary"]["compiler_dirty"]
    _require(lanes["qat"]["compiler_dirty"] is dirty,
             "Ordinary and QAT compiler dirty states differ")
    if dirty:
        return (
            "diagnostic_dirty_worktree",
            "Compiler provenance records a dirty worktree; this is diagnostic evidence, not release evidence.",
        )
    _require(lanes["ordinary"]["compiler_commit"] == lanes["qat"]["compiler_commit"],
             "clean Ordinary and QAT compiler commits differ")
    return (
        "clean_revision_evidence",
        "Compiler provenance binds both lanes to one clean implementation commit.",
    )


def build_tensor_report(lanes: dict[str, dict[str, Any]], config: dict[str, Any],
                        plan_paths: dict[str, Path], spec_path: Path) -> dict[str, Any]:
    records = _matrix_records(lanes)
    totals = _totals(records)
    _require(totals == {"matrix_count": 115, "output_tensor_count": 230,
                        "weight_elements": 1_110_179_840, "weight_bytes": 1_110_179_840,
                        "scale_elements": 335_360, "scale_bytes": 670_720,
                        "output_tensor_bytes": 1_110_850_560}, "semantic totals do not reconcile")
    evidence_class, provenance_limitation = _evidence_class(lanes)
    return {
        "schema_version": 1, "report_kind": "m05_fp8_semantic_tensor_report",
        "status": "pass", "evidence_class": evidence_class,
        "artifact_profile": M05_PROFILE.name, "source_contract": M05_SOURCE_CONTRACT,
        "quantizer": {"contract_id": "gem16.fp8_attention_rowwise", "contract_version": 1,
                      "native_protocol": "gem16-fp8-batch-v1", "spec_sha256": _sha256(spec_path)},
        "native_encoder": {"protocol": "gem16-fp8-batch-v1", "sha256": lanes["ordinary"]["native_hash"],
                           "threads": 8, "build": lanes["ordinary"]["build"]},
        "sources": {
            "ordinary_bf16": _provenance_record(lanes["ordinary"]),
            "qat_bf16": _provenance_record(lanes["qat"]),
        },
        "plans": {lane: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
                  for lane, path in plan_paths.items()},
        "totals": totals, "global_layers_without_v": sorted(M05_GLOBAL_LAYERS),
        "limitations": [
            "Telemetry summarizes stored weights and row scales only.",
            "No model-quality or QAT attribution claim is made.",
            provenance_limitation,
        ],
        "matrices": records,
    }


def _weighted_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_elements = sum(item["elements"] for item in records)
    source_energy = sum(item["source_rms"] ** 2 * item["elements"] for item in records)
    error_energy = sum(item["relative_l2_error"] ** 2 * item["source_rms"] ** 2 * item["elements"]
                       for item in records)
    _require(source_energy >= 0.0 and error_energy >= 0.0, "negative aggregate energy")
    sqnr = sorted(item["sqnr_db"] for item in records if item["sqnr_db"] is not None)
    _require(sqnr, "all QAT SQNR values are null")
    return {
        "elements": total_elements,
        "weighted_source_rms": math.sqrt(source_energy / total_elements),
        "aggregate_relative_l2_error": math.sqrt(error_energy / source_energy) if source_energy else None,
        "sqnr_db": {"count": len(sqnr), "min": min(sqnr), "median": statistics.median(sqnr), "max": max(sqnr)},
    }


def build_qat_summary(lanes: dict[str, dict[str, Any]], config: dict[str, Any],
                      plan_paths: dict[str, Path], spec_path: Path) -> dict[str, Any]:
    records = _matrix_records(lanes)
    weights = [item["qat"]["weight"] for item in records]
    scales = [item["qat"]["scale"] for item in records]
    totals = _totals(records)
    by_role: dict[str, dict[str, int]] = {}
    for item in records:
        role = item["role"].removeprefix("attention_").removesuffix("_projection")
        target = by_role.setdefault(role, {"matrix_count": 0, "weight_elements": 0, "weight_bytes": 0,
                                            "scale_elements": 0, "scale_bytes": 0})
        target["matrix_count"] += 1
        target["weight_elements"] += item["qat"]["weight"]["elements"]
        target["weight_bytes"] += item["qat"]["weight"]["elements"]
        target["scale_elements"] += item["qat"]["scale"]["rows"]
        target["scale_bytes"] += item["qat"]["scale"]["rows"] * 2
    saturation = sum(x["saturation_count"] for x in weights)
    zeros = sum(x["zero_rows"] for x in weights)
    underflow = sum(x["underflow_clamped_rows"] for x in weights)
    rows = sum(x["rows"] for x in weights)
    histogram = [sum(x["histogram"][index] for x in weights) for index in range(EXPECTED_HISTOGRAM_BINS)]
    _require(sum(histogram) == totals["weight_elements"], "QAT histogram does not reconcile")
    scale_values = [x["scale_min"] for x in scales] + [x["scale_max"] for x in scales]
    source_min = min(x["source_min"] for x in weights)
    source_max = max(x["source_max"] for x in weights)
    max_error = max(x["max_absolute_error"] for x in weights)
    evidence_class, provenance_limitation = _evidence_class(lanes)
    return {
        "schema_version": 1, "report_kind": "m05_qat_fp8_summary",
        "status": "pass", "evidence_class": evidence_class,
        "artifact_profile": M05_PROFILE.name, "source_contract": M05_SOURCE_CONTRACT,
        "quantizer": {"contract_id": "gem16.fp8_attention_rowwise", "contract_version": 1,
                      "native_protocol": "gem16-fp8-batch-v1", "spec_sha256": _sha256(spec_path)},
        "native_encoder": {"protocol": "gem16-fp8-batch-v1", "sha256": lanes["qat"]["native_hash"],
                           "threads": 8, "build": lanes["qat"]["build"]},
        "source": _provenance_record(lanes["qat"]),
        "plans": {lane: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
                  for lane, path in plan_paths.items()},
        "totals": totals, "by_role": by_role,
        "aggregate_qat": {
            "rows": rows,
            "saturation": {"count": saturation, "rate": saturation / totals["weight_elements"]},
            "zero_rows": {"count": zeros, "rate": zeros / rows},
            "underflow_clamped_rows": {"count": underflow, "rate": underflow / rows},
            "scale_min": min(scale_values), "scale_max": max(scale_values),
            "source_min": source_min, "source_max": source_max,
            "max_absolute_error": max_error,
            "histogram": histogram,
            "histogram_sum": sum(histogram),
            "error_metrics": _weighted_metrics(weights),
        },
        "global_layers_without_v": sorted(M05_GLOBAL_LAYERS),
        "limitations": [
            "Telemetry summarizes stored weights and row scales only.",
            "No model-quality or QAT attribution claim is made.",
            provenance_limitation,
        ],
    }


def generate(ordinary_path: Path = DEFAULT_ORDINARY, qat_path: Path = DEFAULT_QAT,
             config_path: Path = DEFAULT_CONFIG, ordinary_plan: Path = DEFAULT_ORDINARY_PLAN,
             qat_plan: Path = DEFAULT_QAT_PLAN, spec_path: Path = DEFAULT_SPEC,
             ordinary_manifest: Path = DEFAULT_ORDINARY_MANIFEST,
             qat_manifest: Path = DEFAULT_QAT_MANIFEST) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load(config_path)
    ordinary = _load(ordinary_path)
    qat = _load(qat_path)
    plan_paths = {"ordinary_bf16": ordinary_plan, "qat_bf16": qat_plan}
    _validate_config(config, spec_path, plan_paths)
    lanes = {
        "ordinary": _validate_lane(
            ordinary, ordinary_path, "ordinary",
            M05_SOURCE_LOCK_SHA256["ordinary_bf16"], ordinary_plan,
            ordinary_manifest, config,
        ),
        "qat": _validate_lane(
            qat, qat_path, "qat",
            M05_SOURCE_LOCK_SHA256["qat_bf16"], qat_plan,
            qat_manifest, config,
        ),
    }
    _require(lanes["ordinary"]["native_hash"] == lanes["qat"]["native_hash"] and
             lanes["ordinary"]["build"] == lanes["qat"]["build"],
             "Ordinary and QAT native identities differ")
    _evidence_class(lanes)
    return (build_tensor_report(lanes, config, plan_paths, spec_path),
            build_qat_summary(lanes, config, plan_paths, spec_path))


def _write_or_check(path: Path, value: dict[str, Any], check: bool) -> None:
    expected = canonical_json_bytes(value)
    if check:
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise _fail(f"cannot read retained report {path}: {error}") from error
        _require(actual == expected, f"retained report is stale or non-canonical: {path}")
    else:
        write_file_atomic(path, expected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary", type=Path, default=DEFAULT_ORDINARY)
    parser.add_argument("--qat", type=Path, default=DEFAULT_QAT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ordinary-manifest", type=Path, default=DEFAULT_ORDINARY_MANIFEST)
    parser.add_argument("--qat-manifest", type=Path, default=DEFAULT_QAT_MANIFEST)
    parser.add_argument("--ordinary-plan", type=Path, default=DEFAULT_ORDINARY_PLAN)
    parser.add_argument("--qat-plan", type=Path, default=DEFAULT_QAT_PLAN)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--tensor-report", type=Path, default=DEFAULT_TENSOR_REPORT)
    parser.add_argument("--qat-summary", type=Path, default=DEFAULT_QAT_SUMMARY)
    parser.add_argument("--check", action="store_true", help="verify retained reports without writing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tensor, summary = generate(
            args.ordinary, args.qat, args.config, args.ordinary_plan,
            args.qat_plan, args.spec, args.ordinary_manifest, args.qat_manifest,
        )
        _write_or_check(args.tensor_report, tensor, args.check)
        _write_or_check(args.qat_summary, summary, args.check)
    except (ReportError, OutputError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
    print("M05 retained reports are current" if args.check else "M05 retained reports written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
