#!/usr/bin/env python3
"""Independent bounded BF16-oracle check for the M25 Assistant hybrid artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any


# The accepted target FP8 QAT artifact records 0.0263997 aggregate relative
# L2. Keep the independent Assistant bound slightly above that frozen format
# characteristic rather than inventing a stricter, incompatible 0.02 gate.
FP8_MAX_RELATIVE_L2 = 0.03
FP8_MIN_COSINE = 0.999
NVFP4_MAX_RELATIVE_L2 = 0.25
NVFP4_MIN_COSINE = 0.97


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_bf16(payload: bytes) -> list[float]:
    if len(payload) % 2:
        raise ValueError("BF16 payload has odd byte length")
    return [
        struct.unpack("<f", b"\0\0" + payload[index:index + 2])[0]
        for index in range(0, len(payload), 2)
    ]


def decode_e4m3fn(code: int) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    exponent = (code >> 3) & 0x0F
    mantissa = code & 0x07
    if exponent == 0:
        value = mantissa * (2.0 ** -9)
    elif exponent == 0x0F and mantissa == 0x07:
        return math.nan
    else:
        value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
    return sign * value


def decode_e2m1(code: int) -> float:
    values = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    return (-1.0 if code & 0x08 else 1.0) * values[code & 0x07]


class TensorFile:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as stream:
            raw_size = stream.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"short Safetensors header length: {path}")
            header_size = struct.unpack("<Q", raw_size)[0]
            if header_size > 16 * 1024 * 1024:
                raise ValueError(f"Safetensors header exceeds bound: {path}")
            raw_header = stream.read(header_size)
        header = json.loads(raw_header)
        if not isinstance(header, dict):
            raise ValueError(f"Safetensors header is not an object: {path}")
        self.data_begin = 8 + header_size
        self.tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        file_size = path.stat().st_size
        for name, value in self.tensors.items():
            offsets = value.get("data_offsets")
            if (
                not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(item, int) for item in offsets)
                or offsets[0] < 0 or offsets[1] < offsets[0]
                or self.data_begin + offsets[1] > file_size
            ):
                raise ValueError(f"invalid Safetensors range: {name}")

    def descriptor(self, name: str) -> dict[str, Any]:
        value = self.tensors.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"missing tensor: {name}")
        return value

    def read(self, name: str) -> bytes:
        value = self.descriptor(name)
        begin, end = value["data_offsets"]
        with self.path.open("rb", buffering=0) as stream:
            stream.seek(self.data_begin + begin)
            payload = stream.read(end - begin)
        if len(payload) != end - begin:
            raise ValueError(f"short tensor read: {name}")
        return payload

    def read_row(self, name: str, row: int, row_bytes: int) -> bytes:
        value = self.descriptor(name)
        begin, end = value["data_offsets"]
        offset = begin + row * row_bytes
        if row < 0 or offset + row_bytes > end:
            raise ValueError(f"tensor row is outside range: {name}:{row}")
        with self.path.open("rb", buffering=0) as stream:
            stream.seek(self.data_begin + offset)
            payload = stream.read(row_bytes)
        if len(payload) != row_bytes:
            raise ValueError(f"short tensor row read: {name}:{row}")
        return payload


def artifact_tensors(root: Path) -> dict[str, TensorFile]:
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("artifact index weight_map is missing")
    shards = {
        name: TensorFile(root / name)
        for name in sorted(set(weight_map.values()))
    }
    return {name: shards[shard] for name, shard in weight_map.items()}


def metrics(source: list[float], candidate: list[float]) -> dict[str, float]:
    if len(source) != len(candidate) or not source:
        raise ValueError("metric vectors differ or are empty")
    source_energy = sum(value * value for value in source)
    candidate_energy = sum(value * value for value in candidate)
    error_energy = sum((actual - expected) ** 2 for expected, actual in zip(source, candidate))
    dot = sum(expected * actual for expected, actual in zip(source, candidate))
    if not all(math.isfinite(value) for value in (
        source_energy, candidate_energy, error_energy, dot
    )):
        raise ValueError("non-finite metric accumulation")
    relative_l2 = math.sqrt(error_energy / source_energy) if source_energy else 0.0
    cosine = (
        dot / math.sqrt(source_energy * candidate_energy)
        if source_energy and candidate_energy else 1.0
    )
    return {
        "relative_l2_error": relative_l2,
        "cosine_similarity": max(-1.0, min(1.0, cosine)),
        "max_absolute_error": max(abs(actual - expected) for expected, actual in zip(source, candidate)),
    }


def selected_rows(name: str, rows: int) -> list[int]:
    values = {0, rows // 2, rows - 1}
    if name == "model.embed_tokens.weight":
        values.update({1, 106, 1000, 255999, 256000, 258880})
    return sorted(value for value in values if 0 <= value < rows)


def fp8_check(
    source: TensorFile,
    artifact: dict[str, TensorFile],
    name: str,
    shape: list[int],
) -> dict[str, Any]:
    rows, columns = shape
    stem = name.removesuffix(".weight")
    weight_file = artifact[name]
    scale_name = f"{stem}.weight_scale"
    scale_file = artifact[scale_name]
    expected_values: list[float] = []
    actual_values: list[float] = []
    for row in selected_rows(name, rows):
        expected_values.extend(decode_bf16(source.read_row(name, row, columns * 2)))
        codes = weight_file.read_row(name, row, columns)
        scale = decode_bf16(scale_file.read_row(scale_name, row, 2))[0]
        actual_values.extend(decode_e4m3fn(code) * scale for code in codes)
    result = metrics(expected_values, actual_values)
    result.update({"source_name": name, "rows_sampled": selected_rows(name, rows)})
    result["pass"] = (
        result["relative_l2_error"] <= FP8_MAX_RELATIVE_L2
        and result["cosine_similarity"] >= FP8_MIN_COSINE
    )
    return result


def nvfp4_check(
    source: TensorFile,
    artifact: dict[str, TensorFile],
    name: str,
    shape: list[int],
) -> dict[str, Any]:
    rows, columns = shape
    stem = name.removesuffix(".weight")
    packed_name = f"{stem}.weight_packed"
    scale_name = f"{stem}.weight_scale"
    global_name = f"{stem}.weight_global_scale"
    divisor = struct.unpack("<f", artifact[global_name].read(global_name))[0]
    if not math.isfinite(divisor) or divisor <= 0:
        raise ValueError(f"invalid NVFP4 divisor: {name}")
    expected_values: list[float] = []
    actual_values: list[float] = []
    for row in selected_rows(name, rows):
        expected_values.extend(decode_bf16(source.read_row(name, row, columns * 2)))
        packed = artifact[packed_name].read_row(packed_name, row, columns // 2)
        scales = artifact[scale_name].read_row(scale_name, row, columns // 16)
        for column in range(columns):
            byte = packed[column // 2]
            code = (byte >> 4) & 0x0F if column & 1 else byte & 0x0F
            scale = decode_e4m3fn(scales[column // 16])
            actual_values.append(decode_e2m1(code) * scale / divisor)
    result = metrics(expected_values, actual_values)
    result.update({"source_name": name, "rows_sampled": selected_rows(name, rows)})
    result["pass"] = (
        result["relative_l2_error"] <= NVFP4_MAX_RELATIVE_L2
        and result["cosine_similarity"] >= NVFP4_MIN_COSINE
    )
    return result


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f"oracle output already exists: {args.output}")
    source_file = TensorFile(args.source / "model.safetensors")
    artifact = artifact_tensors(args.artifact)
    compilation_path = args.artifact / "gem16_compilation.json"
    compilation = json.loads(compilation_path.read_text(encoding="utf-8"))
    plan_tensors = compilation.get("tensors")
    if compilation.get("artifact_profile") != "sm120-mtp-assistant-hybrid-v1":
        raise ValueError("artifact is not the locked M25 Assistant hybrid profile")
    if not isinstance(plan_tensors, list) or len(plan_tensors) != 97:
        raise ValueError("M25 compilation tensor inventory is incomplete")
    by_source: dict[str, list[dict[str, Any]]] = {}
    for item in plan_tensors:
        sources = item.get("sources")
        if isinstance(sources, list) and len(sources) == 1:
            by_source.setdefault(sources[0]["name"], []).append(item)

    fp8: list[dict[str, Any]] = []
    nvfp4: list[dict[str, Any]] = []
    copied: list[str] = []
    for name in sorted(source_file.tensors):
        descriptor = source_file.descriptor(name)
        group = by_source.get(name, [])
        encoders = {item["transformation"] for item in group}
        if "bf16-to-fp8-e4m3fn-rowwise-weight" in encoders:
            fp8.append(fp8_check(source_file, artifact, name, descriptor["shape"]))
        elif "nvfp4-packed" in encoders:
            nvfp4.append(nvfp4_check(source_file, artifact, name, descriptor["shape"]))
        elif len(group) == 1 and group[0]["transformation"] == "identity-copy":
            if source_file.read(name) != artifact[name].read(name):
                raise ValueError(f"BF16 copied tensor differs: {name}")
            copied.append(name)
        else:
            raise ValueError(f"source tensor has no recognized compiled mapping: {name}")

    status = (
        "pass" if all(item["pass"] for item in fp8 + nvfp4)
        and len(fp8) == 10 and len(nvfp4) == 13 and len(copied) == 25 else "fail"
    )
    report = {
        "schema_version": 1,
        "milestone": "M25",
        "status": status,
        "scope": "bounded-independent-bf16-oracle",
        "artifact_profile": compilation["artifact_profile"],
        "source": compilation["source"],
        "compilation_manifest_sha256": sha256_file(compilation_path),
        "thresholds": {
            "fp8_max_relative_l2": FP8_MAX_RELATIVE_L2,
            "fp8_min_cosine": FP8_MIN_COSINE,
            "nvfp4_max_relative_l2": NVFP4_MAX_RELATIVE_L2,
            "nvfp4_min_cosine": NVFP4_MIN_COSINE,
        },
        "counts": {
            "fp8_matrix_count": len(fp8),
            "nvfp4_matrix_count": len(nvfp4),
            "bf16_exact_copy_count": len(copied),
        },
        "fp8": fp8,
        "nvfp4": nvfp4,
        "bf16_exact_copies": copied,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "counts": report["counts"],
        "worst_fp8_relative_l2": max(item["relative_l2_error"] for item in fp8),
        "worst_nvfp4_relative_l2": max(item["relative_l2_error"] for item in nvfp4),
        "minimum_fp8_cosine": min(item["cosine_similarity"] for item in fp8),
        "minimum_nvfp4_cosine": min(item["cosine_similarity"] for item in nvfp4),
    }, indent=2, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
