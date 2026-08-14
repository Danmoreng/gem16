#!/usr/bin/env python3
"""Independent, bounded CPU oracle for the Gemma 4 26B MoE contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ModuleNotFoundError:  # Compact evidence tests do not require the offline replay dependency.
    np = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "benchmarks/goldens/gemma4_26b/qat-bf16-selected/qat-bf16-selected.json"
COMPILED = ROOT / "artifacts/raw/m08/qat-hybrid-clean-1"
SOURCE = ROOT / "models/checkpoints/google-gemma-4-26b-a4b-it-qat-bf16-f1e06dc"
OUTPUT = ROOT / "artifacts/m10/diagnostic-summary.json"
ACCEPTANCE_OUTPUT = ROOT / "artifacts/m10/acceptance.json"
LAYERS = (0, 5, 6, 29)
POSITIONS = (0, 17)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class OracleError(ValueError):
    pass


def _reject_json_constant(value: str) -> None:
    raise OracleError(f"non-finite JSON constant is forbidden: {value}")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OracleError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _json_loads(payload: str | bytes) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_json_object,
                          parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise OracleError(f"invalid JSON: {error}") from error


def _load_json_file(path: Path, maximum_bytes: int = 64 * 1024 * 1024) -> Any:
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"JSON input must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise OracleError(f"JSON input exceeds bound: {path}")
    return _json_loads(path.read_bytes())


def _finite(values: Iterable[float], where: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) for value in result):
        raise OracleError(f"{where} must contain finite values")
    return result


def bf16_round(value: float) -> float:
    if not math.isfinite(value):
        raise OracleError("BF16 input must be finite")
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def rms_norm(values: Sequence[float], scale: Sequence[float] | None, epsilon: float) -> list[float]:
    source = _finite(values, "RMSNorm input")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise OracleError("RMSNorm epsilon must be positive")
    weights = [1.0] * len(source) if scale is None else _finite(scale, "RMSNorm scale")
    if len(weights) != len(source):
        raise OracleError("RMSNorm scale shape mismatch")
    inverse = (math.fsum(value * value for value in source) / len(source) + epsilon) ** -0.5
    return [bf16_round(value * inverse * weight) for value, weight in zip(source, weights)]


def linear_bf16(matrix: Sequence[Sequence[float]], values: Sequence[float]) -> list[float]:
    source = _finite(values, "linear input")
    result = []
    for row in matrix:
        weights = _finite(row, "linear row")
        if len(weights) != len(source):
            raise OracleError("linear matrix shape mismatch")
        result.append(bf16_round(math.fsum(weight * value for weight, value in zip(weights, source))))
    if not result:
        raise OracleError("linear matrix must have rows")
    return result


def gelu_tanh(value: float) -> float:
    factor = math.sqrt(2.0 / math.pi)
    return bf16_round(0.5 * value * (1.0 + math.tanh(factor * (value + 0.044715 * value**3))))


def mlp_bf16(values: Sequence[float], gate: Sequence[Sequence[float]],
             up: Sequence[Sequence[float]], down: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    gate_values = linear_bf16(gate, values)
    up_values = linear_bf16(up, values)
    if len(gate_values) != len(up_values):
        raise OracleError("MLP gate/up shape mismatch")
    product = [bf16_round(gelu_tanh(left) * right) for left, right in zip(gate_values, up_values)]
    return {"gate": gate_values, "up": up_values, "product": product,
            "output": linear_bf16(down, product)}


def fused_expert_mlp_bf16(values: Sequence[float], gate_up: Sequence[Sequence[float]],
                          down: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    fused = list(gate_up)
    if not fused or len(fused) % 2:
        raise OracleError("fused expert gate-up axis must contain two equal halves")
    middle = len(fused) // 2
    return mlp_bf16(values, fused[:middle], fused[middle:], down)


def softmax_fp32(logits: Sequence[float]) -> list[float]:
    source = _finite(logits, "router logits")
    maximum = max(source)
    exponentials = [math.exp(value - maximum) for value in source]
    total = math.fsum(exponentials)
    if not math.isfinite(total) or total <= 0.0:
        raise OracleError("router softmax sum is invalid")
    return [value / total for value in exponentials]


def select_topk(probabilities: Sequence[float], per_expert_scale: Sequence[float],
                top_k: int = 8) -> dict[str, list[float] | list[int]]:
    probs = _finite(probabilities, "router probabilities")
    scales = _finite(per_expert_scale, "per-expert scale")
    if len(probs) != len(scales) or top_k <= 0 or top_k > len(probs):
        raise OracleError("router top-k geometry is invalid")
    if any(scale <= 0.0 for scale in scales):
        raise OracleError("per-expert scales must be positive")
    # The independent deterministic policy is probability descending, then lower ID.
    ids = sorted(range(len(probs)), key=lambda expert: (-probs[expert], expert))[:top_k]
    selected_sum = math.fsum(probs[expert] for expert in ids)
    if selected_sum <= 0.0 or not math.isfinite(selected_sum):
        raise OracleError("selected router sum is invalid")
    normalized = [probs[expert] / selected_sum for expert in ids]
    weights = [normalized[index] * scales[expert] for index, expert in enumerate(ids)]
    return {"ids": ids, "normalized_probabilities": normalized, "weights": weights}


def router_bf16(hidden: Sequence[float], learned_scale: Sequence[float],
                projection: Sequence[Sequence[float]], per_expert_scale: Sequence[float],
                epsilon: float = 1e-6, top_k: int = 8) -> dict[str, Any]:
    normalized = rms_norm(hidden, None, epsilon)
    scale = _finite(learned_scale, "router learned scale")
    if len(scale) != len(normalized):
        raise OracleError("router learned scale shape mismatch")
    root = len(normalized) ** -0.5
    transformed = [bf16_round(value * weight * root) for value, weight in zip(normalized, scale)]
    logits = linear_bf16(projection, transformed)
    probabilities = softmax_fp32(logits)
    selected = select_topk(probabilities, per_expert_scale, top_k)
    return {"normalized": normalized, "transformed": transformed, "logits": logits,
            "probabilities": probabilities, **selected}


def moe_layer_bf16(residual: Sequence[float], norms: dict[str, Sequence[float]],
                   shared: dict[str, Sequence[Sequence[float]]],
                   experts: dict[int, dict[str, Sequence[Sequence[float]]]],
                   router: dict[str, Any], layer_scalar: float = 1.0,
                   epsilon: float = 1e-6) -> dict[str, Any]:
    shared_input = rms_norm(residual, norms["pre_shared"], epsilon)
    shared_trace = mlp_bf16(shared_input, shared["gate"], shared["up"], shared["down"])
    shared_post = rms_norm(shared_trace["output"], norms["post_shared"], epsilon)
    route = router_bf16(residual, router["scale"], router["projection"],
                        router["per_expert_scale"], epsilon, router.get("top_k", 8))
    expert_input = rms_norm(residual, norms["pre_expert"], epsilon)
    contributions = []
    routed = [0.0] * len(residual)
    for rank, (expert_id, weight) in enumerate(zip(route["ids"], route["weights"])):
        if expert_id not in experts:
            raise OracleError(f"missing selected expert {expert_id}")
        trace = mlp_bf16(expert_input, experts[expert_id]["gate"],
                         experts[expert_id]["up"], experts[expert_id]["down"])
        weighted = [float(value) * float(weight) for value in trace["output"]]
        routed = [left + right for left, right in zip(routed, weighted)]
        contributions.append({"rank": rank, "expert_id": expert_id,
                              "router_weight": weight, "output": trace["output"],
                              "weighted": weighted})
    routed_post = rms_norm([bf16_round(value) for value in routed], norms["post_expert"], epsilon)
    combined = [bf16_round(left + right) for left, right in zip(shared_post, routed_post)]
    ff = rms_norm(combined, norms["post_combined"], epsilon)
    output = [bf16_round(bf16_round(left + right) * layer_scalar)
              for left, right in zip(residual, ff)]
    return {"shared_input": shared_input, "shared": shared_trace, "shared_post": shared_post,
            "router": route, "expert_input": expert_input, "expert_contributions": contributions,
            "routed_sum_fp32": routed, "routed_post": routed_post, "combined": combined,
            "feed_forward": ff, "output": output}


def decode_e4m3fn(code: int) -> float:
    if not isinstance(code, int) or code < 0 or code > 255 or (code & 0x7F) == 0x7F:
        raise OracleError("invalid E4M3FN code")
    sign = -1.0 if code & 0x80 else 1.0
    magnitude = code & 0x7F
    exponent = (magnitude >> 3) & 0xF
    mantissa = magnitude & 0x7
    value = mantissa / 512.0 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return sign * value


def decode_e2m1(code: int) -> float:
    if not isinstance(code, int) or code < 0 or code > 15:
        raise OracleError("invalid E2M1 code")
    return (-1.0 if code & 8 else 1.0) * E2M1[code & 7]


def dequantize_nvfp4_row(packed: bytes, scales: bytes, weight_divisor: float,
                         columns: int) -> list[float]:
    if columns <= 0 or columns % 16 or len(packed) * 2 != columns or len(scales) * 16 != columns:
        raise OracleError("NVFP4 row geometry is invalid")
    if not math.isfinite(weight_divisor) or weight_divisor <= 0.0:
        raise OracleError("NVFP4 weight divisor must be positive")
    result = []
    for column in range(columns):
        byte = packed[column // 2]
        code = (byte >> 4) & 0xF if column & 1 else byte & 0xF
        result.append(decode_e2m1(code) * decode_e4m3fn(scales[column // 16]) /
                      weight_divisor)
    return result


def _rows(captures: dict[str, Any], name: str, key: str) -> dict[int, list[Any]]:
    value = captures.get(name)
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise OracleError(f"missing trusted capture {name}")
    result = {int(row["position"]): list(row[key]) for row in value["rows"]}
    if set(result) != set(POSITIONS):
        raise OracleError(f"trusted capture positions differ: {name}")
    return result


def validate_trusted_bf16_golden(document: dict[str, Any]) -> dict[str, Any]:
    if (document.get("schema_version") != 1 or
            document.get("status") != "deterministic_reference_candidate" or
            document.get("checkpoint", {}).get("lock_sha256") !=
            "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230" or
            document.get("software", {}).get("lock_sha256") !=
            "577528f96ac1e64d5e258a9a50eaa60f4cb48bf0294b8c6db96bace1f921ab6e" or
            document.get("software", {}).get("trust_remote_code") is not False):
        raise OracleError("trusted BF16 golden identity is invalid")
    captures = document.get("captures")
    if not isinstance(captures, dict):
        raise OracleError("trusted BF16 captures are missing")
    ordered_exact = 0
    tie_equivalent = 0
    contribution_count = 0
    routed_max_abs = 0.0
    routed_relative_l2 = 0.0
    routed_cosine = 1.0
    for layer in LAYERS:
        probabilities = _rows(captures, f"layer_{layer}.router_probabilities", "values_f32")
        trusted_ids = _rows(captures, f"layer_{layer}.router_top_ids", "values_i64")
        routed_sum = _rows(captures, f"layer_{layer}.routed_experts_sum", "values_f32")
        contributions = captures.get(f"layer_{layer}.routed_expert_contributions")
        if not isinstance(contributions, list) or len(contributions) != 2:
            raise OracleError("trusted expert contributions are missing")
        by_position = {int(record["position"]): record["contributions"] for record in contributions}
        for position in POSITIONS:
            probs = _finite(probabilities[position], "trusted router probabilities")
            if len(probs) != 128 or abs(math.fsum(probs) - 1.0) > 2e-6:
                raise OracleError("trusted router probabilities are invalid")
            oracle_ids = sorted(range(128), key=lambda expert: (-probs[expert], expert))[:8]
            if oracle_ids == trusted_ids[position]:
                ordered_exact += 1
            elif set(oracle_ids) == set(trusted_ids[position]):
                tie_equivalent += 1
            else:
                raise OracleError("trusted top-8 set disagrees with deterministic oracle")
            records = by_position[position]
            if len(records) != 8 or [int(item["rank"]) for item in records] != list(range(8)):
                raise OracleError("trusted expert contribution slots are invalid")
            contribution_count += len(records)
            target = _finite(routed_sum[position], "trusted routed sum")
            combined = [math.fsum(float(item["values_f32"][index]) for item in records)
                        for index in range(len(target))]
            rounded = [bf16_round(value) for value in combined]
            routed_max_abs = max(routed_max_abs,
                                 max(abs(value - expected)
                                     for value, expected in zip(rounded, target)))
            difference_norm = math.sqrt(math.fsum((value - expected) ** 2
                                                  for value, expected in zip(rounded, target)))
            target_norm = math.sqrt(math.fsum(value * value for value in target))
            rounded_norm = math.sqrt(math.fsum(value * value for value in rounded))
            routed_relative_l2 = max(routed_relative_l2, difference_norm / target_norm)
            routed_cosine = min(routed_cosine,
                                math.fsum(value * expected for value, expected in zip(rounded, target)) /
                                (rounded_norm * target_norm))
    if (ordered_exact != 7 or tie_equivalent != 1 or routed_max_abs > 0.03125 or
            routed_relative_l2 > 0.005 or routed_cosine < 0.99999):
        raise OracleError("trusted BF16 boundary reconciliation exceeded its contract")
    return {"layers": list(LAYERS), "positions_per_layer": len(POSITIONS),
            "router_ordered_exact": ordered_exact, "router_tie_equivalent": tie_equivalent,
            "expert_contribution_count": contribution_count,
            "routed_sum_max_abs": routed_max_abs,
            "routed_sum_relative_l2": routed_relative_l2,
            "routed_sum_cosine": routed_cosine,
            "routed_sum_thresholds": {"max_abs": 0.03125, "relative_l2": 0.005,
                                      "cosine_min": 0.99999}}


class SafeTensorReader:
    """Minimal read-only Safetensors reader with bounded header and range checks."""

    _WIDTH = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
              "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
              "U32": 4, "I32": 4, "F32": 4,
              "U64": 8, "I64": 8, "F64": 8}

    def __init__(self, root: Path):
        self.root = root.resolve()
        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_symlink() or not index_path.is_file():
            raise OracleError("Safetensors index must be a regular non-symlink file")
        document = _load_json_file(index_path)
        self.weight_map = document.get("weight_map")
        if not isinstance(self.weight_map, dict):
            raise OracleError("Safetensors index weight_map is missing")
        self.headers: dict[Path, tuple[int, dict[str, Any]]] = {}

    def record(self, name: str) -> tuple[Path, str, tuple[int, ...], int, int]:
        shard_name = self.weight_map.get(name)
        if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
            raise OracleError(f"unsafe or missing Safetensors shard for {name}")
        shard = self.root / shard_name
        if shard.is_symlink() or not shard.is_file() or shard.resolve().parent != self.root:
            raise OracleError("Safetensors shard must be an in-root regular non-symlink file")
        if shard not in self.headers:
            with shard.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise OracleError("Safetensors header prefix is truncated")
                header_size = struct.unpack("<Q", prefix)[0]
                if header_size <= 0 or header_size > 64 * 1024 * 1024:
                    raise OracleError("Safetensors header exceeds bound")
                payload = stream.read(header_size)
                if len(payload) != header_size:
                    raise OracleError("Safetensors header is truncated")
                header = _json_loads(payload)
            if not isinstance(header, dict):
                raise OracleError("Safetensors header must be an object")
            payload_size = shard.stat().st_size - 8 - header_size
            ranges = []
            for tensor_name, candidate in header.items():
                if tensor_name == "__metadata__":
                    continue
                if not isinstance(candidate, dict):
                    raise OracleError("Safetensors tensor record must be an object")
                candidate_dtype = candidate.get("dtype")
                candidate_shape = candidate.get("shape")
                candidate_offsets = candidate.get("data_offsets")
                if (candidate_dtype not in self._WIDTH or
                        not isinstance(candidate_shape, list) or
                        any(not isinstance(value, int) or value <= 0
                            for value in candidate_shape) or
                        not isinstance(candidate_offsets, list) or len(candidate_offsets) != 2):
                    raise OracleError("Safetensors header contains invalid tensor geometry")
                candidate_start, candidate_end = candidate_offsets
                expected = math.prod(candidate_shape) * self._WIDTH[candidate_dtype]
                if (not isinstance(candidate_start, int) or not isinstance(candidate_end, int) or
                        candidate_start < 0 or candidate_end <= candidate_start or
                        candidate_end - candidate_start != expected or candidate_end > payload_size):
                    raise OracleError("Safetensors header contains invalid tensor range")
                ranges.append((candidate_start, candidate_end))
            ranges.sort()
            if any(right[0] < left[1] for left, right in zip(ranges, ranges[1:])):
                raise OracleError("Safetensors tensor ranges overlap")
            self.headers[shard] = (header_size, header)
        header_size, header = self.headers[shard]
        record = header.get(name)
        if not isinstance(record, dict):
            raise OracleError(f"Safetensors tensor is missing: {name}")
        dtype = record.get("dtype")
        offsets = record.get("data_offsets")
        shape = record.get("shape")
        if (dtype not in self._WIDTH or not isinstance(offsets, list) or len(offsets) != 2 or
                not isinstance(shape, list) or
                any(not isinstance(value, int) or value <= 0 for value in shape)):
            raise OracleError("Safetensors tensor geometry is invalid")
        start, end = offsets
        elements = math.prod(shape)
        if (not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or
                end - start != elements * self._WIDTH[dtype] or
                8 + header_size + end > shard.stat().st_size):
            raise OracleError("Safetensors tensor range is invalid")
        return shard, dtype, tuple(shape), 8 + header_size + start, end - start

    def prefix(self, name: str, count: int) -> tuple[str, tuple[int, ...], bytes]:
        shard, dtype, shape, offset, size = self.record(name)
        if count <= 0 or count > size:
            raise OracleError("Safetensors prefix request exceeds tensor")
        with shard.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read(count)
        if len(payload) != count:
            raise OracleError("Safetensors tensor is truncated")
        return dtype, shape, payload

    def bf16(self, name: str, first_axis: int | None = None) -> np.ndarray:
        shard, dtype, shape, offset, _ = self.record(name)
        if dtype != "BF16":
            raise OracleError(f"expected BF16 tensor: {name}")
        if (first_axis is not None and
                (not isinstance(first_axis, int) or first_axis < 0 or first_axis >= shape[0])):
            raise OracleError(f"BF16 first-axis selection is invalid: {name}")
        mapped = np.memmap(shard, dtype="<u2", mode="r", offset=offset, shape=shape)
        selected = mapped if first_axis is None else mapped[first_axis]
        bits = np.asarray(selected, dtype=np.uint32) << np.uint32(16)
        return bits.view(np.float32)


def _np_bf16(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _np_gelu_tanh(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    factor = np.float32(math.sqrt(2.0 / math.pi))
    return _np_bf16(np.float32(0.5) * source *
                     (np.float32(1.0) + np.tanh(factor *
                     (source + np.float32(0.044715) * source**3))))


def _np_rms_norm(values: np.ndarray, scale: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    weights = np.asarray(scale, dtype=np.float32)
    if source.shape[-1] != weights.shape[0]:
        raise OracleError("NumPy RMSNorm shape mismatch")
    mean_squared = np.mean(source * source, axis=-1, keepdims=True, dtype=np.float32)
    inverse = np.power(mean_squared + np.float32(epsilon), np.float32(-0.5))
    return _np_bf16(source * inverse * weights)


def _f32_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes(order="C")).hexdigest()


def _metrics(actual: np.ndarray, expected: Sequence[float]) -> dict[str, float]:
    left = np.asarray(actual, dtype=np.float32)
    right = np.asarray(expected, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 1 or not np.isfinite(left).all():
        raise OracleError("BF16 replay boundary shape or finiteness mismatch")
    difference = left - right
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    left_norm = float(np.linalg.norm(left64))
    right_norm = float(np.linalg.norm(right64))
    denominator = left_norm * right_norm
    return {"max_abs": float(np.max(np.abs(difference))),
            "relative_l2": float(np.linalg.norm(difference.astype(np.float64)) /
                                 max(right_norm, 1e-30)),
            "cosine": float(np.dot(left64, right64) / denominator) if denominator else 1.0,
            "exact_fraction": float(np.mean(left == right))}


def validate_real_bf16_replay(root: Path, document: dict[str, Any]) -> dict[str, Any]:
    if np is None:
        raise OracleError("NumPy is required to generate the offline BF16 acceptance replay")
    reader = SafeTensorReader(root)
    captures = document["captures"]
    shared_records = []
    norm_records = []
    router_records = []
    expert_records = []
    for layer in LAYERS:
        prefix = f"model.language_model.layers.{layer}"
        shared_inputs = _rows(captures, f"layer_{layer}.pre_feedforward_layernorm", "values_f32")
        shared_targets = _rows(captures, f"layer_{layer}.shared_mlp", "values_f32")
        batch = np.asarray([shared_inputs[position] for position in POSITIONS], dtype=np.float32)
        gate_weight = reader.bf16(prefix + ".mlp.gate_proj.weight")
        up_weight = reader.bf16(prefix + ".mlp.up_proj.weight")
        down_weight = reader.bf16(prefix + ".mlp.down_proj.weight")
        gate = _np_bf16(batch @ gate_weight.T)
        up = _np_bf16(batch @ up_weight.T)
        product = _np_bf16(_np_gelu_tanh(gate) * up)
        shared_output = _np_bf16(product @ down_weight.T)
        post_shared_scale = reader.bf16(prefix + ".post_feedforward_layernorm_1.weight")
        post_expert_scale = reader.bf16(prefix + ".post_feedforward_layernorm_2.weight")
        post_combined_scale = reader.bf16(prefix + ".post_feedforward_layernorm.weight")
        shared_post_targets = _rows(captures, f"layer_{layer}.post_feedforward_layernorm_1", "values_f32")
        routed_sum_targets = _rows(captures, f"layer_{layer}.routed_experts_sum", "values_f32")
        routed_post_targets = _rows(captures, f"layer_{layer}.post_feedforward_layernorm_2", "values_f32")
        final_norm_targets = _rows(captures, f"layer_{layer}.post_feedforward_layernorm", "values_f32")
        for index, position in enumerate(POSITIONS):
            shared_records.append({"layer": layer, "position": position,
                                   "product_sha256_f32le": _f32_sha256(product[index]),
                                   "output_sha256_f32le": _f32_sha256(shared_output[index]),
                                   "metrics": _metrics(shared_output[index], shared_targets[position])})
            shared_post = _np_rms_norm(shared_output[index], post_shared_scale)
            routed_post = _np_rms_norm(np.asarray(routed_sum_targets[position], np.float32),
                                       post_expert_scale)
            combined = _np_bf16(shared_post + routed_post)
            final_norm = _np_rms_norm(combined, post_combined_scale)
            norm_records.extend((
                {"layer": layer, "position": position, "boundary": "post_shared",
                 "metrics": _metrics(shared_post, shared_post_targets[position])},
                {"layer": layer, "position": position, "boundary": "post_expert",
                 "metrics": _metrics(routed_post, routed_post_targets[position])},
                {"layer": layer, "position": position, "boundary": "post_combined",
                 "metrics": _metrics(final_norm, final_norm_targets[position])},
            ))

        pre_norm = reader.bf16(prefix + ".pre_feedforward_layernorm.weight")
        router_scale = reader.bf16(prefix + ".router.scale")
        router_projection = reader.bf16(prefix + ".router.proj.weight")
        expert_scale = reader.bf16(prefix + ".router.per_expert_scale")
        trusted_probabilities = _rows(captures, f"layer_{layer}.router_probabilities", "values_f32")
        trusted_ids = _rows(captures, f"layer_{layer}.router_top_ids", "values_i64")
        trusted_weights = _rows(captures, f"layer_{layer}.router_top_weights", "values_f32")
        expert_inputs = _rows(captures, f"layer_{layer}.pre_feedforward_layernorm_2", "values_f32")
        contribution_groups = captures[f"layer_{layer}.routed_expert_contributions"]
        contributions = {int(group["position"]): group["contributions"]
                         for group in contribution_groups}
        for position in POSITIONS:
            # Dividing the captured pre-norm output by its learned scale reconstructs
            # the scale-invariant router RMS input up to the documented BF16 rounding.
            normalized_residual = np.asarray(shared_inputs[position], dtype=np.float32) / pre_norm
            transformed = _np_bf16(normalized_residual * router_scale *
                                    np.float32(2816 ** -0.5))
            logits = _np_bf16(router_projection @ transformed)
            shifted = logits - np.max(logits)
            exponential = np.exp(shifted, dtype=np.float32)
            probabilities = exponential / np.sum(exponential, dtype=np.float32)
            ids = sorted(range(128), key=lambda expert: (-float(probabilities[expert]), expert))[:8]
            reference_ids = [int(value) for value in trusted_ids[position]]
            aligned = np.asarray([probabilities[expert] for expert in reference_ids], dtype=np.float32)
            weights = aligned / np.sum(aligned, dtype=np.float32)
            weights *= expert_scale[reference_ids]
            router_records.append({"layer": layer, "position": position,
                                   "top8_set_match": set(ids) == set(reference_ids),
                                   "top8_order_match": ids == reference_ids,
                                   "probability_max_abs": float(np.max(np.abs(
                                       probabilities - np.asarray(trusted_probabilities[position], np.float32)))),
                                   "weight_max_abs": float(np.max(np.abs(
                                       weights - np.asarray(trusted_weights[position], np.float32))))})

            expert_input = np.asarray(expert_inputs[position], dtype=np.float32)
            for rank, contribution in enumerate(contributions[position]):
                expert_id = int(contribution["expert_id"])
                if expert_id != reference_ids[rank]:
                    raise OracleError("trusted contribution order disagrees with router slots")
                gate_up_weight = reader.bf16(prefix + ".experts.gate_up_proj", expert_id)
                down_expert_weight = reader.bf16(prefix + ".experts.down_proj", expert_id)
                gate_up = _np_bf16(gate_up_weight @ expert_input)
                gate_expert, up_expert = np.split(gate_up, 2)
                expert_product = _np_bf16(_np_gelu_tanh(gate_expert) * up_expert)
                expert_down = _np_bf16(down_expert_weight @ expert_product)
                weighted = _np_bf16(expert_down * np.float32(trusted_weights[position][rank]))
                expert_records.append({"layer": layer, "position": position, "rank": rank,
                                       "expert_id": expert_id,
                                       "product_sha256_f32le": _f32_sha256(expert_product),
                                       "down_sha256_f32le": _f32_sha256(expert_down),
                                       "weighted_sha256_f32le": _f32_sha256(weighted),
                                       "metrics": _metrics(weighted, contribution["values_f32"])})

    shared_relative = max(record["metrics"]["relative_l2"] for record in shared_records)
    shared_cosine = min(record["metrics"]["cosine"] for record in shared_records)
    expert_relative = max(record["metrics"]["relative_l2"] for record in expert_records)
    expert_cosine = min(record["metrics"]["cosine"] for record in expert_records)
    norm_relative = max(record["metrics"]["relative_l2"] for record in norm_records)
    norm_cosine = min(record["metrics"]["cosine"] for record in norm_records)
    probability_error = max(record["probability_max_abs"] for record in router_records)
    weight_error = max(record["weight_max_abs"] for record in router_records)
    passed = (all(record["top8_set_match"] for record in router_records) and
              shared_relative <= 0.001 and shared_cosine >= 0.99999 and
              expert_relative <= 0.01 and expert_cosine >= 0.9999 and
              norm_relative <= 0.005 and norm_cosine >= 0.99999 and
              probability_error <= 0.003 and weight_error <= 0.005)
    if not passed:
        raise OracleError("real BF16 replay exceeded its fixed acceptance thresholds")
    return {"status": "pass", "source_dtype": "BF16", "uses_cuda": False,
            "uses_transformers_or_torch": False,
            "thresholds": {"shared_relative_l2_max": 0.001, "shared_cosine_min": 0.99999,
                           "expert_relative_l2_max": 0.01, "expert_cosine_min": 0.9999,
                           "norm_relative_l2_max": 0.005, "norm_cosine_min": 0.99999,
                           "router_probability_max_abs": 0.003,
                           "router_weight_max_abs": 0.005},
            "worst": {"shared_relative_l2": shared_relative, "shared_cosine": shared_cosine,
                      "expert_relative_l2": expert_relative, "expert_cosine": expert_cosine,
                      "norm_relative_l2": norm_relative, "norm_cosine": norm_cosine,
                      "router_probability_max_abs": probability_error,
                      "router_weight_max_abs": weight_error},
            "shared_boundaries": shared_records, "norm_boundaries": norm_records,
            "router_boundaries": router_records,
            "expert_boundaries": expert_records}


def validate_real_nvfp4_adapter(root: Path) -> dict[str, Any]:
    lock_path = root.parent / f"{root.name}.lock.json"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise OracleError("compiled artifact lock must be a regular non-symlink file")
    if lock_path.stat().st_size <= 0 or lock_path.stat().st_size > 16 * 1024 * 1024:
        raise OracleError("compiled artifact lock exceeds bound")
    lock_bytes = lock_path.read_bytes()
    lock = _json_loads(lock_bytes)
    expected_content = "471805f7dad8abb84300be78b2822a63dcb1d35bff5aa98426a162cc8532ee17"
    if (lock.get("schema_version") != 1 or
            lock.get("artifact_content_sha256") != expected_content or
            lock.get("source_lock_sha256") !=
            "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230"):
        raise OracleError("compiled artifact lock identity is invalid")
    reader = SafeTensorReader(root)
    samples = []
    expected_rows = {
        "model.language_model.layers.0.mlp.gate_proj":
            "2e090475c2662ad9e6cdf3b4240e93c1dafc21ab85d5f052b7a1e9ac18af93b1",
        "model.language_model.layers.0.experts.gate_up_proj":
            "234793f87f78bc443a8a99cac532cbc8643f432135a230ea55b9929d7a8fb766",
    }
    for stem in (
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.experts.gate_up_proj",
    ):
        _, packed_dtype, packed_shape, _, _ = reader.record(stem + ".weight_packed")
        _, scale_dtype, scale_shape, _, _ = reader.record(stem + ".weight_scale")
        global_dtype, global_shape, divisor_bytes = reader.prefix(stem + ".weight_global_scale", 4)
        if (packed_dtype != "U8" or scale_dtype != "F8_E4M3" or
                global_dtype != "F32" or global_shape != (1,)):
            raise OracleError("compiled NVFP4 component dtype contract mismatch")
        columns = packed_shape[-1] * 2
        scale_columns = scale_shape[-1]
        if columns // 16 != scale_columns:
            raise OracleError("compiled NVFP4 group-16 geometry mismatch")
        _, _, packed = reader.prefix(stem + ".weight_packed", packed_shape[-1])
        _, _, scales = reader.prefix(stem + ".weight_scale", scale_shape[-1])
        divisor = struct.unpack("<f", divisor_bytes)[0]
        first = dequantize_nvfp4_row(packed, scales, divisor, columns)
        payload = b"".join(struct.pack("<f", value) for value in first)
        row_sha256 = hashlib.sha256(payload).hexdigest()
        if row_sha256 != expected_rows[stem]:
            raise OracleError("compiled NVFP4 dequantized row identity is invalid")
        samples.append({"stem": stem, "logical_columns": columns,
                        "weight_divisor": divisor,
                        "first_row_f32_sha256": row_sha256,
                        "finite": all(math.isfinite(value) for value in first)})
    return {"artifact_content_sha256": expected_content,
            "artifact_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "sample_count": len(samples), "samples": samples}


def build_report(golden_path: Path, source_root: Path, compiled_root: Path,
                 acceptance: bool = False) -> dict[str, Any]:
    golden = _load_json_file(golden_path)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout)
    if acceptance and dirty:
        raise OracleError("M10 acceptance generation requires a clean implementation commit")
    limitations = [
        "the trusted PyTorch capture has one exact-probability tie with nonportable topk ordering",
        "router residual is reconstructed from a BF16 pre-norm capture, so router numeric checks use explicit tolerances",
        "M11 production CUDA code is not imported or executed",
    ]
    if not acceptance:
        limitations.insert(2, "the implementation worktree is dirty; clean commit-bound acceptance remains required")
    report = {"schema_version": 1, "milestone": "M10",
            "status": ("acceptance_pass" if acceptance else
                       "diagnostic_pass_acceptance_pending_clean_commit"),
            "acceptance": acceptance, "code_revision": revision + ("-dirty" if dirty else ""),
            "oracle": {"runtime": "python-numpy-cpu",
            "uses_production_cuda": False, "router_precision": "FP32 softmax and selected normalization",
            "tie_policy": "lower_expert_id", "expert_reduction": "top_k_slot_order_fp32",
            "activation": "gelu_pytorch_tanh", "capture_boundaries": ["router_probabilities",
            "top8_ids", "top8_weights", "shared_output", "expert_product", "expert_down",
            "weighted_contribution", "routed_sum", "combined_feed_forward", "layer_output"]},
            "trusted_bf16": validate_trusted_bf16_golden(golden),
            "real_bf16_replay": validate_real_bf16_replay(source_root, golden),
            "real_nvfp4_adapter": validate_real_nvfp4_adapter(compiled_root),
            "limitations": limitations}
    if acceptance:
        report["implementation_commit"] = revision
        report["owner_decision"] = {
            "date": "2026-08-14",
            "decision": "M10 accepted",
            "tie_policy": "lower_expert_id",
            "expert_reduction": "top_k_slot_order_fp32",
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--source-model", type=Path, default=SOURCE)
    parser.add_argument("--compiled-model", type=Path, default=COMPILED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output or (ACCEPTANCE_OUTPUT if args.acceptance else OUTPUT)
    payload = (json.dumps(build_report(args.golden, args.source_model, args.compiled_model,
                                       acceptance=args.acceptance), indent=2,
                          sort_keys=True) + "\n").encode("utf-8")
    if args.check:
        if not output.is_file() or output.read_bytes() != payload:
            raise SystemExit(f"generated output is stale: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
