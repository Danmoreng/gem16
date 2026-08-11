from __future__ import annotations

from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from tools.gem16_compile.common import (
    BoundedWorkspace,
    DataError,
    OutputError,
    peak_rss_bytes,
    write_all,
)
from tools.gem16_compile.encoders import (
    FP8RowwiseScaleEncoder,
    FP8RowwiseWeightEncoder,
    default_encoder_registry,
)
from tools.gem16_compile.plan import TensorCompilePlan
from tools.gem16_compile.reader import TensorDescriptor
from tools.gem16_compile.quantize_fp8 import (
    BF16_MIN_POSITIVE_BITS,
    BF16_ONE_BITS,
    E4M3FN_MAX,
    E4M3FN_NAN_CODE,
    FP8_CONTRACT_ID,
    FP8_CONTRACT_VERSION,
    decode_bf16,
    decode_e4m3fn,
    encode_e4m3fn,
    quantize_bf16_row,
    round_binary32_to_bf16_rne,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "tools/gem16_compile/specs/fp8-attention-rowwise-v1.json"


def bf16_row(*bits: int) -> bytes:
    return b"".join(struct.pack("<H", bit) for bit in bits)


class FP8CodecTest(unittest.TestCase):
    def test_contract_json_freezes_code_constants(self) -> None:
        document = json.loads(SPEC.read_text(encoding="utf-8"))
        self.assertEqual(document["contract_id"], FP8_CONTRACT_ID)
        self.assertEqual(document["version"], FP8_CONTRACT_VERSION)
        self.assertEqual(document["source"]["endianness"], "little")
        self.assertEqual(document["output"]["weights_dtype"], "F8_E4M3")
        self.assertEqual(document["output"]["codec_semantics"],
                         "E4M3FN (finite-only E4M3 with NaN encodings rejected)")
        telemetry = document["telemetry"]
        self.assertEqual(telemetry["element_order"],
                         "logical source row order k=0..K-1")
        self.assertEqual(telemetry["source_value_precision"],
                         "decoded BF16 represented as binary32")
        self.assertEqual(telemetry["reconstruction"],
                         "binary32_rne(decode_e4m3fn(code) * decode_bf16(stored_scale))")
        self.assertEqual(telemetry["accumulation_precision"], "binary64 Python float")
        self.assertEqual(telemetry["partial_row_accumulation"],
                         "each row contribution is computed before matrix aggregation")
        self.assertEqual(telemetry["accumulation_order"],
                         "left-to-right in logical source row order")
        aggregates = telemetry["matrix_aggregate_definitions"]
        self.assertIn("sqrt(error_sum_squares)", aggregates["relative_l2_error"])
        self.assertEqual(aggregates["cosine_clamp"],
                         "clamp cosine_similarity to [-1.0, 1.0] after division")
        self.assertIn("10 * log10(source_sum_squares / error_sum_squares)",
                      aggregates["sqnr_db"])
        self.assertIn("sqnr_db is null", aggregates["perfect_reconstruction"])
        self.assertIn("finite", aggregates["finite_json"])
        self.assertEqual(telemetry["row_metrics"], [
            "element_count", "source_min", "source_max", "source_sum_squares",
            "reconstruction_sum_squares", "source_reconstruction_dot",
            "error_sum_squares", "max_absolute_error",
        ])
        runtime = document["runtime_projection"]
        self.assertEqual(runtime["activation_scale"],
                         "dynamic per-token binary32_rne(max_abs / 448.0)")
        self.assertEqual(runtime["activation_all_zero_scale"], "1.0")
        self.assertEqual(runtime["activation_bytes"], "E4M3FN byte codes")
        self.assertEqual(runtime["accumulation"], "FP32")
        self.assertEqual(runtime["scale_order"],
                         "(accumulator * input_scale) * weight_scale")
        self.assertEqual(runtime["output_boundary"], "BF16 RNE closure")
        self.assertEqual(document["arithmetic"]["binary32_rounding"], "nearest_even")
        self.assertEqual(document["arithmetic"]["bf16_rounding"], "nearest_even")
        self.assertEqual(document["arithmetic"]["e4m3fn_rounding"], "nearest_even")
        self.assertIn("even least-significant bit", document["arithmetic"]["tie_rule"])
        self.assertEqual(document["arithmetic"]["e4m3fn_finite_max"], E4M3FN_MAX)
        self.assertIn("preserves each signed zero", document["arithmetic"]["zero_row"])
        self.assertEqual(document["scale"]["all_zero"], "BF16 1.0 (0x3f80)")
        self.assertEqual(document["scale"]["positive_underflow"],
                         "clamp to minimum positive BF16 subnormal (0x0001)")
        self.assertEqual(document["arithmetic"]["e4m3fn_nan_codes"], ["0x7f", "0xff"])
        self.assertEqual(E4M3FN_NAN_CODE, 0x7F)
        self.assertEqual(BF16_ONE_BITS, 0x3F80)
        self.assertEqual(BF16_MIN_POSITIVE_BITS, 0x0001)

    def test_exhaustive_finite_e4m3_round_trip(self) -> None:
        for unsigned_code in range(256):
            if (unsigned_code & 0x7F) == E4M3FN_NAN_CODE:
                continue
            decoded = decode_e4m3fn(unsigned_code)
            self.assertEqual(encode_e4m3fn(decoded), unsigned_code)

    def test_every_positive_midpoint_tie_is_even(self) -> None:
        values = [decode_e4m3fn(code) for code in range(0x7F)]
        for low in range(0x7E):
            midpoint = (values[low] + values[low + 1]) / 2.0
            midpoint = struct.unpack("<f", struct.pack("<f", midpoint))[0]
            # All E4M3 adjacent midpoints are exactly representable in f32.
            self.assertEqual(midpoint, (values[low] + values[low + 1]) / 2.0)
            expected = low if low % 2 == 0 else low + 1
            self.assertEqual(encode_e4m3fn(midpoint), expected)
            self.assertEqual(encode_e4m3fn(-midpoint), expected | 0x80)

    def test_e4m3_boundaries_saturation_zero_and_rejection(self) -> None:
        self.assertEqual(decode_e4m3fn(0x01), 1.0 / 512.0)
        self.assertEqual(decode_e4m3fn(0x08), 1.0 / 64.0)
        self.assertEqual(decode_e4m3fn(0x38), 1.0)
        self.assertEqual(decode_e4m3fn(0x61), 36.0)
        self.assertEqual(decode_e4m3fn(0x7E), 448.0)
        self.assertEqual(decode_e4m3fn(0xB8), -1.0)
        self.assertEqual(encode_e4m3fn(1000.0), 0x7E)
        self.assertEqual(encode_e4m3fn(-1000.0), 0xFE)
        self.assertEqual(encode_e4m3fn(0.0), 0x00)
        self.assertEqual(encode_e4m3fn(-0.0), 0x80)
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(DataError):
                encode_e4m3fn(value)
        for code in (0x7F, 0xFF):
            with self.assertRaises(DataError):
                decode_e4m3fn(code)

    def test_bf16_round_to_nearest_even_and_decode(self) -> None:
        even_tie = struct.unpack("<f", struct.pack("<I", 0x3F808000))[0]
        odd_tie = struct.unpack("<f", struct.pack("<I", 0x3F818000))[0]
        self.assertEqual(round_binary32_to_bf16_rne(even_tie), 0x3F80)
        self.assertEqual(round_binary32_to_bf16_rne(odd_tie), 0x3F82)
        self.assertEqual(round_binary32_to_bf16_rne(decode_bf16(0x0001)), 0x0001)
        self.assertEqual(decode_bf16(0x3F80), 1.0)
        with self.assertRaises(DataError):
            round_binary32_to_bf16_rne(-1.0)
        with self.assertRaises(DataError):
            round_binary32_to_bf16_rne(math.inf)

    def test_row_zero_tiny_mixed_and_statistics(self) -> None:
        zero = quantize_bf16_row(bf16_row(0x0000, 0x8000, 0x0000))
        self.assertTrue(zero.zero_row)
        self.assertFalse(zero.scale_underflow_clamped)
        self.assertEqual(zero.scale_bits, BF16_ONE_BITS)
        self.assertEqual(zero.scale_bytes, b"\x80\x3f")
        self.assertEqual(zero.weight_bytes, b"\x00\x80\x00")
        self.assertEqual(sum(zero.histogram), 3)
        self.assertLessEqual(zero.saturation_count, zero.metrics.element_count)
        self.assertEqual(zero.metrics.element_count, 3)
        self.assertEqual(zero.metrics.source_min, 0.0)
        self.assertEqual(zero.metrics.source_max, 0.0)
        self.assertEqual(zero.metrics.source_sum_squares, 0.0)
        self.assertEqual(zero.metrics.reconstruction_sum_squares, 0.0)
        self.assertEqual(zero.metrics.source_reconstruction_dot, 0.0)
        self.assertEqual(zero.metrics.error_sum_squares, 0.0)
        self.assertEqual(zero.metrics.max_absolute_error, 0.0)

        exact = quantize_bf16_row(bf16_row(0x43E0, 0xC3E0, 0x0000, 0x8000))
        self.assertEqual(exact.weight_bytes, bytes.fromhex("7efe0080"))
        self.assertEqual(exact.metrics.element_count, 4)
        self.assertEqual(exact.metrics.source_min, -448.0)
        self.assertEqual(exact.metrics.source_max, 448.0)
        self.assertEqual(exact.metrics.source_sum_squares, 401408.0)
        self.assertEqual(exact.metrics.reconstruction_sum_squares, 401408.0)
        self.assertEqual(exact.metrics.source_reconstruction_dot, 401408.0)
        self.assertEqual(exact.metrics.error_sum_squares, 0.0)
        self.assertEqual(exact.metrics.max_absolute_error, 0.0)
        self.assertEqual(sum(exact.histogram), exact.metrics.element_count)
        self.assertLessEqual(exact.saturation_count, exact.metrics.element_count)

        tiny = quantize_bf16_row(bf16_row(BF16_MIN_POSITIVE_BITS))
        self.assertFalse(tiny.zero_row)
        self.assertTrue(tiny.scale_underflow_clamped)
        self.assertEqual(tiny.scale_bits, BF16_MIN_POSITIVE_BITS)
        self.assertEqual(tiny.weight_bytes, b"\x38")

        mixed = quantize_bf16_row(bf16_row(0x3F80, 0xC000, 0x3F00, 0x8000))
        self.assertFalse(mixed.zero_row)
        self.assertFalse(mixed.scale_underflow_clamped)
        self.assertEqual(mixed.scale_bits, 0x3B92)
        self.assertEqual(mixed.weight_bytes, bytes.fromhex("76fe6e80"))
        self.assertEqual(mixed.saturation_count, 1)
        self.assertEqual(sum(mixed.histogram), len(mixed.weight_bytes))
        self.assertEqual(sum(mixed.histogram), mixed.metrics.element_count)
        self.assertLessEqual(mixed.saturation_count, mixed.metrics.element_count)

        rounded_scale = quantize_bf16_row(bf16_row(0x3F82, 0xBF82))
        self.assertEqual(rounded_scale.scale_bits, 0x3B15)
        self.assertEqual(rounded_scale.weight_bytes, bytes.fromhex("7efe"))
        self.assertFalse(rounded_scale.scale_underflow_clamped)
        self.assertEqual(rounded_scale.saturation_count, 0)
        self.assertEqual(rounded_scale.histogram[0x7E], 1)
        self.assertEqual(rounded_scale.histogram[0xFE], 1)
        self.assertEqual(sum(rounded_scale.histogram), rounded_scale.metrics.element_count)
        self.assertLessEqual(rounded_scale.saturation_count,
                             rounded_scale.metrics.element_count)

    def test_rejects_empty_odd_nan_inf(self) -> None:
        for source in (b"", b"\x00", bf16_row(0x7F80), bf16_row(0xFF80),
                       bf16_row(0x7FC1), bf16_row(0xFFC1)):
            with self.assertRaises(DataError):
                quantize_bf16_row(source)

    def test_determinism_and_bounded_width(self) -> None:
        import tracemalloc

        pattern = (0x3F80, 0xC000, 0x3F00, 0x8000,
                   0x3F82, 0xBF82, 0x0001, 0x8001)
        source = bf16_row(*(pattern * (2816 // len(pattern))))
        tracemalloc.start()
        try:
            first = quantize_bf16_row(source)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        second = quantize_bf16_row(source)
        self.assertEqual(first, second)
        self.assertEqual(len(first.weight_bytes), 2816)
        self.assertEqual(sum(first.histogram), 2816)
        self.assertEqual(sum(first.histogram), first.metrics.element_count)
        self.assertLessEqual(first.saturation_count, first.metrics.element_count)
        self.assertLess(peak, 4 * 1024 * 1024)


class _PartialWriter:
    def __init__(self, maximum: int = 3) -> None:
        self.maximum = maximum
        self.payload = bytearray()

    def write(self, value: bytes | bytearray | memoryview) -> int:
        chunk = bytes(value[:self.maximum])
        self.payload.extend(chunk)
        return len(chunk)


class _ZeroProgressWriter:
    def write(self, value: bytes | bytearray | memoryview) -> int:
        return 0


class FP8EncoderTest(unittest.TestCase):
    @staticmethod
    def _plan(encoder: str, output_name: str, dtype: str, shape: tuple[int, ...]) -> TensorCompilePlan:
        return TensorCompilePlan(
            output_name=output_name,
            operation_id="gem16.fp8.test",
            source_names=("source",),
            encoder=encoder,
            transformation="bf16-to-fp8-rowwise" if dtype == "F8_E4M3" else "bf16-to-row-scale",
            transformation_version=1,
            output_dtype=dtype,
            physical_shape=shape,
            logical_dtype=dtype,
            logical_shape=shape,
            axis_transformation="output-row",
            quantizer_parameters={"contract": FP8_CONTRACT_ID},
            dequantization_equation="decode(weight) * scale",
            role="attention_q_projection",
            residency_class="immutable_device_text",
            disk_layout="row-major",
            runtime_layout="source_nk_fp8" if dtype == "F8_E4M3" else "row_scale",
            aliased=False,
        )

    def _fixture(self, payload: bytes, shape: tuple[int, int], staging: int | None = None) -> tuple[Path, TensorDescriptor, BoundedWorkspace, tempfile.TemporaryDirectory[str]]:
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "source.safetensors"
        prefix = b"header-for-offset"
        path.write_bytes(prefix + payload + b"trailer")
        source = TensorDescriptor(
            name="source",
            dtype="BF16",
            shape=shape,
            shard="source.safetensors",
            path=path,
            absolute_offset=len(prefix),
            data_offset=0,
            byte_length=len(payload),
            shard_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        cap = max(256 * 1024 * 1024, peak_rss_bytes() + 64 * 1024 * 1024)
        workspace = BoundedWorkspace(
            cap, staging_bytes=staging if staging is not None else max(4096, shape[1] * 2)
        )
        return path, source, workspace, temporary

    def test_registry_and_bounded_rowwise_outputs(self) -> None:
        payload = bf16_row(
            0x3F80, 0xC000, 0x3F00, 0x8000,
            0x43E0, 0xC3E0, 0x0000, 0x8000,
        )
        _, source, workspace, temporary = self._fixture(payload, (2, 4))
        try:
            weight_plan = self._plan("fp8-rowwise-weight-v1", "q.weight", "F8_E4M3", (2, 4))
            scale_plan = self._plan("fp8-rowwise-scale-v1", "q.weight_scale", "BF16", (2, 1))
            weight_output = BytesIO()
            scale_output = BytesIO()
            weight = FP8RowwiseWeightEncoder().compile_tensor(
                weight_plan, (source,), weight_output, workspace
            )
            scale = FP8RowwiseScaleEncoder().compile_tensor(
                scale_plan, (source,), scale_output, workspace
            )
            expected_weights = quantize_bf16_row(payload[:8]).weight_bytes + quantize_bf16_row(payload[8:]).weight_bytes
            expected_scales = quantize_bf16_row(payload[:8]).scale_bytes + quantize_bf16_row(payload[8:]).scale_bytes
            self.assertEqual(weight_output.getvalue(), expected_weights)
            self.assertEqual(scale_output.getvalue(), expected_scales)
            expected_source_hash = hashlib.sha256(payload).hexdigest()
            self.assertEqual(weight.source_sha256, (expected_source_hash,))
            self.assertEqual(scale.source_sha256, (expected_source_hash,))
            self.assertEqual(weight.output_sha256, hashlib.sha256(expected_weights).hexdigest())
            self.assertEqual(scale.output_sha256, hashlib.sha256(expected_scales).hexdigest())
            stats = weight.statistics
            assert stats is not None
            self.assertEqual(stats["contract_id"], FP8_CONTRACT_ID)
            self.assertEqual(stats["contract_version"], FP8_CONTRACT_VERSION)
            self.assertEqual(stats["rows"], 2)
            self.assertEqual(stats["columns"], 4)
            self.assertEqual(stats["elements"], 8)
            self.assertEqual(sum(stats["histogram"]), 8)
            self.assertLessEqual(stats["saturation_count"], 8)
            self.assertEqual(stats["saturation_rate"], stats["saturation_count"] / 8)
            self.assertEqual(stats["zero_rows"], 0)
            self.assertEqual(stats["underflow_clamped_rows"], 0)
            self.assertTrue(math.isfinite(stats["relative_l2_error"]))
            self.assertTrue(math.isfinite(stats["cosine_similarity"]))
            row_metrics = [
                quantize_bf16_row(payload[offset:offset + 8]).metrics
                for offset in (0, 8)
            ]
            source_energy = sum(item.source_sum_squares for item in row_metrics)
            error_energy = sum(item.error_sum_squares for item in row_metrics)
            expected_sqnr = 10.0 * math.log10(source_energy / error_energy)
            self.assertAlmostEqual(stats["sqnr_db"], expected_sqnr, places=12)
            reconstruction_energy = sum(
                item.reconstruction_sum_squares for item in row_metrics
            )
            dot = sum(item.source_reconstruction_dot for item in row_metrics)
            expected_cosine = dot / math.sqrt(source_energy * reconstruction_energy)
            self.assertAlmostEqual(stats["cosine_similarity"], expected_cosine, places=12)
            scale_stats = scale.statistics
            assert scale_stats is not None
            self.assertEqual(scale_stats["component"], "scale")
            self.assertEqual(scale_stats["rows"], 2)
            self.assertEqual(scale_stats["columns"], 1)
            self.assertEqual(workspace.maximum_transform_row_bytes, 8)
            self.assertEqual(workspace.telemetry()["maximum_transform_row_bytes"], 8)
            self.assertEqual(default_encoder_registry().keys(), {"copy-v1"})
        finally:
            temporary.cleanup()

    def test_zero_vector_aggregate_uses_defined_sentinels(self) -> None:
        payload = bf16_row(0x0000, 0x8000, 0x0000, 0x8000)
        _, source, workspace, temporary = self._fixture(payload, (1, 4))
        try:
            output = BytesIO()
            result = FP8RowwiseWeightEncoder().compile_tensor(
                self._plan("fp8-rowwise-weight-v1", "q.weight", "F8_E4M3", (1, 4)),
                (source,), output, workspace,
            )
            assert result.statistics is not None
            self.assertEqual(result.statistics["relative_l2_error"], 0.0)
            self.assertEqual(result.statistics["cosine_similarity"], 1.0)
            self.assertIsNone(result.statistics["sqnr_db"])
            self.assertTrue(result.statistics["perfect_reconstruction"])
        finally:
            temporary.cleanup()

    def test_bounded_write_all_handles_partial_and_zero_progress(self) -> None:
        partial = _PartialWriter(maximum=2)
        write_all(partial, b"abcdef", "partial test")
        self.assertEqual(bytes(partial.payload), b"abcdef")
        with self.assertRaises(OutputError):
            write_all(_ZeroProgressWriter(), b"abcdef", "zero test")

    def test_rowwise_output_handles_partial_writes(self) -> None:
        payload = bf16_row(0x3F80, 0xC000, 0x3F00, 0x8000)
        _, source, workspace, temporary = self._fixture(payload, (1, 4))
        try:
            output = _PartialWriter(maximum=1)
            FP8RowwiseWeightEncoder().compile_tensor(
                self._plan("fp8-rowwise-weight-v1", "q.weight", "F8_E4M3", (1, 4)),
                (source,), output, workspace,
            )
            self.assertEqual(bytes(output.payload), bytes.fromhex("76fe6e80"))
        finally:
            temporary.cleanup()

    def test_rowwise_source_bounds_and_cleanup_after_quantizer_failure(self) -> None:
        payload = bf16_row(0x3F80, 0xC000, 0x3F00, 0x8000)
        path, source, workspace, temporary = self._fixture(payload, (1, 4))
        try:
            path.write_bytes(path.read_bytes()[:source.absolute_offset + source.byte_length - 1])
            with self.assertRaises(DataError):
                FP8RowwiseWeightEncoder().compile_tensor(
                    self._plan("fp8-rowwise-weight-v1", "q.weight", "F8_E4M3", (1, 4)),
                    (source,), BytesIO(), workspace,
                )
        finally:
            temporary.cleanup()

        bad_payload = bf16_row(0x3F80, 0xC000, 0x7FC1, 0x3F00)
        path, bad_source, workspace, temporary = self._fixture(bad_payload, (1, 4))
        try:
            with self.assertRaises(DataError):
                FP8RowwiseWeightEncoder().compile_tensor(
                    self._plan("fp8-rowwise-weight-v1", "q.weight", "F8_E4M3", (1, 4)),
                    (bad_source,), BytesIO(), workspace,
                )
            renamed = path.with_name("renamed-source.safetensors")
            path.rename(renamed)
            self.assertTrue(renamed.is_file())
        finally:
            temporary.cleanup()

    def test_largest_attention_row_stays_bounded(self) -> None:
        columns = 8192
        payload = bf16_row(*([0x3F80, 0xC000] * (columns // 2)))
        _, source, workspace, temporary = self._fixture(
            payload, (1, columns), staging=16 * 1024
        )
        try:
            output = BytesIO()
            result = FP8RowwiseWeightEncoder().compile_tensor(
                self._plan("fp8-rowwise-weight-v1", "q.weight", "F8_E4M3", (1, columns)),
                (source,), output, workspace,
            )
            self.assertEqual(len(output.getvalue()), columns)
            self.assertEqual(result.statistics["elements"], columns)
            self.assertEqual(workspace.maximum_transform_row_bytes, columns * 2)
            telemetry = workspace.telemetry()
            self.assertEqual(telemetry["maximum_transform_row_bytes"], columns * 2)
            self.assertLessEqual(telemetry["peak_rss_bytes"], workspace.host_memory_cap_bytes)
        finally:
            temporary.cleanup()

    def test_rowwise_encoders_are_deterministic(self) -> None:
        payload = bf16_row(0x3F80, 0xC000, 0x3F00, 0x8000) * 2
        results = []
        outputs = []
        for _ in range(2):
            _, source, workspace, temporary = self._fixture(payload, (2, 4))
            try:
                output = BytesIO()
                result = FP8RowwiseWeightEncoder().compile_tensor(
                    self._plan("fp8-rowwise-weight-v1", "q.weight", "F8_E4M3", (2, 4)),
                    (source,), output, workspace
                )
                results.append(result)
                outputs.append(output.getvalue())
            finally:
                temporary.cleanup()
        self.assertEqual(results[0], results[1])
        self.assertEqual(outputs[0], outputs[1])

    def test_rowwise_encoder_rejects_invalid_contracts_and_ranges(self) -> None:
        payload = bf16_row(0x3F80, 0xC000, 0x3F00, 0x8000)
        _, source, workspace, temporary = self._fixture(payload, (1, 4))
        try:
            encoder = FP8RowwiseWeightEncoder()
            plan = self._plan("fp8-rowwise-weight-v1", "q.weight", "F8_E4M3", (1, 4))
            bad_source = TensorDescriptor(**{**source.__dict__, "byte_length": len(payload) - 2})
            with self.assertRaises(DataError):
                encoder.compile_tensor(plan, (bad_source,), BytesIO(), workspace)
            bad_dtype = TensorDescriptor(**{**source.__dict__, "dtype": "F16"})
            with self.assertRaises(DataError):
                encoder.compile_tensor(plan, (bad_dtype,), BytesIO(), workspace)
            bad_shape = TensorDescriptor(**{**source.__dict__, "shape": (4,)})
            with self.assertRaises(DataError):
                encoder.compile_tensor(plan, (bad_shape,), BytesIO(), workspace)
            bad_plan = self._plan("fp8-rowwise-weight-v1", "q.weight", "BF16", (1, 4))
            with self.assertRaises(DataError):
                encoder.compile_tensor(bad_plan, (source,), BytesIO(), workspace)
            short = TensorDescriptor(**{**source.__dict__, "byte_length": len(payload) - 2})
            with self.assertRaises(DataError):
                encoder.compile_tensor(plan, (short,), BytesIO(), workspace)
        finally:
            temporary.cleanup()

        large_payload = bf16_row(*([0x3F80] * 2049))
        _, large_source, small_workspace, large_temporary = self._fixture(
            large_payload, (1, 2049), staging=4096
        )
        try:
            with self.assertRaises(DataError):
                FP8RowwiseWeightEncoder().compile_tensor(
                    self._plan("fp8-rowwise-weight-v1", "q.weight", "F8_E4M3", (1, 2049)),
                    (large_source,), BytesIO(), small_workspace
                )
        finally:
            large_temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
