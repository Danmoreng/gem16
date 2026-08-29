from __future__ import annotations

import os
from pathlib import Path
import tempfile
import struct
import unittest

import numpy as np

from tools.gem16_compile.common import InvalidPlanError
from tools.gem16_compile.trellis35 import (
    deterministic_signs,
    calibration_hessian,
    finalize_hessian,
    proxy_error,
    quantize_matrix_ldlq,
    read_calibration_capture,
    read_source_expert,
    reconstruct_regularized,
    regularize_weight,
)
from tools.gem16_compile.trellis35_artifact import (
    decode_payload_matrix,
    pack_encoded_tiles,
    sample_scale_tiles,
)
from tools.gem16_compile.reader import TensorDescriptor
from tools.gem16_compile.trellis35_quant import (
    CODEBOOK_MUL1,
    TENSOR_CORE_INVERSE_PERMUTATION,
    TENSOR_CORE_PERMUTATION,
    blockwise_hadamard_right,
    decode_codebook,
    gelu_tanh_product,
    inverse_gate_up_output,
    encoded_tile_from_branches,
    pack_trellis_tile,
)
from tools.verify_gemma4_26b_trellis35_layer import (
    REGION_ORDER,
    expected_region_bytes,
    region_layout,
)


class Trellis35CompilerOracleTest(unittest.TestCase):
    def test_calibration_capture_parser_and_per_expert_hessian(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.bin"
            gate = np.linspace(-1.0, 1.0, 2816, dtype="<f4")
            down = np.arange(8 * 704, dtype="<f4").reshape(8, 704) / 1024.0
            ids = np.arange(8, dtype="<u4")
            with path.open("wb") as output:
                output.write(b"G16T35C1")
                output.write(struct.pack("<III", 1, 0, 1))
                output.write(struct.pack("<I", 0))
                output.write(ids.tobytes())
                output.write(gate.tobytes())
                output.write(down.tobytes())
            capture = read_calibration_capture(path)
            gate_hessian, gate_count = calibration_hessian(capture, "gate_up")
            down_hessian, down_count = calibration_hessian(capture, "down", 3)
            self.assertEqual((gate_count, down_count), (1, 1))
            np.testing.assert_allclose(gate_hessian, np.outer(gate, gate))
            np.testing.assert_allclose(down_hessian, np.outer(down[3], down[3]))

    def test_verified_bf16_expert_reader_preserves_fused_shape_and_pads_down(self) -> None:
        shapes = {
            "gate_up": (128, 1408, 2816),
            "down": (128, 2816, 704),
        }
        with tempfile.TemporaryDirectory() as directory:
            for family, shape in shapes.items():
                path = Path(directory) / f"{family}.bin"
                expert_bytes = shape[1] * shape[2] * 2
                with path.open("wb") as stream:
                    stream.truncate(expert_bytes)
                descriptor = os.open(path, os.O_WRONLY)
                try:
                    os.pwrite(descriptor, bytes.fromhex("803f0040"), 0)
                finally:
                    os.close(descriptor)
                tensor = TensorDescriptor(
                    name=family,
                    dtype="BF16",
                    shape=shape,
                    shard=path.name,
                    path=path,
                    absolute_offset=0,
                    data_offset=0,
                    byte_length=128 * expert_bytes,
                    shard_sha256="0" * 64,
                )
                expert = read_source_expert(tensor, 0, family)
                expected_shape = (2816, 1408) if family == "gate_up" else (768, 2816)
                self.assertEqual(expert.shape, expected_shape)
                self.assertEqual(float(expert[0, 0]), 1.0)
                self.assertEqual(float(expert[1, 0]), 2.0)
                if family == "down":
                    self.assertTrue(np.all(expert[704:] == 0.0))

    def test_hessian_finalization_and_signs_are_deterministic(self) -> None:
        rng = np.random.default_rng(11)
        activations = rng.normal(size=(192, 128))
        capture = activations.T @ activations
        first = finalize_hessian(capture, 192, seed=29, domain="layer0:gate_up")
        second = finalize_hessian(capture, 192, seed=29, domain="layer0:gate_up")
        np.testing.assert_array_equal(first.input_signs, second.input_signs)
        np.testing.assert_allclose(first.transformed, second.transformed, rtol=0, atol=0)
        np.testing.assert_allclose(first.ldl, second.ldl, rtol=0, atol=0)
        self.assertTrue(np.all(np.diag(first.ldl) == 0.0))
        self.assertTrue(np.isfinite(first.ldl).all())

    def test_regularization_reconstructs_full_128_aligned_matrix(self) -> None:
        rng = np.random.default_rng(17)
        source = rng.normal(0.0, 0.2, size=(128, 256))
        signs = deterministic_signs(128, 41, "fixture:input")
        regularized = regularize_weight(
            source,
            signs,
            seed=41,
            domain="fixture:weight",
            global_scale=1.125,
        )
        reconstructed = reconstruct_regularized(regularized)
        np.testing.assert_allclose(reconstructed, source, rtol=1e-12, atol=1e-12)

    def test_single_tile_ldlq_emits_valid_k3_k4_payloads_and_proxy(self) -> None:
        source = np.sin(np.arange(256, dtype=np.float64).reshape(16, 16) * 0.071)
        hessian = np.eye(16, dtype=np.float64)
        factor = np.zeros((16, 16), dtype=np.float64)
        for rate, expected_bytes in ((3, 96), (4, 128)):
            result = quantize_matrix_ldlq(source, factor, rate)
            self.assertEqual(len(result.encoded_tiles), 1)
            self.assertEqual(len(result.packed_payload), expected_bytes)
            self.assertTrue(np.isfinite(result.reconstructed).all())
            self.assertGreater(result.squared_error, 0.0)
            measured = proxy_error(source, result.reconstructed, hessian)
            self.assertAlmostEqual(measured, result.squared_error / np.sum(source * source))

    def test_native_state_batch_packer_matches_scalar_oracle(self) -> None:
        for rate in (3, 4):
            tiles = []
            for tile in range(3):
                branches = tuple(
                    (index * 5 + index // 7 + tile) & ((1 << rate) - 1)
                    for index in range(256)
                )
                tiles.append(encoded_tile_from_branches(branches, rate))
            states = np.asarray(tiles, dtype=np.uint16)
            expected = b"".join(pack_trellis_tile(tile, rate) for tile in tiles)
            self.assertEqual(pack_encoded_tiles(states, rate), expected)
            decoded = decode_payload_matrix(expected, rate, 16, 48)
            expected_decoded = np.asarray(
                [decode_codebook(value, CODEBOOK_MUL1) for tile in tiles for value in tile],
                dtype=np.float32,
            ).reshape(3, 256)
            expected_matrix = expected_decoded[:, np.asarray(TENSOR_CORE_INVERSE_PERMUTATION)].reshape(
                1, 3, 16, 16
            ).transpose(0, 2, 1, 3).reshape(16, 48)
            np.testing.assert_array_equal(decoded, expected_matrix)
            states[1, 17] ^= np.uint16(1 << 8)
            with self.assertRaises(InvalidPlanError):
                pack_encoded_tiles(states, rate)

    def test_scale_tile_sampler_is_deterministic_and_tensor_core_ordered(self) -> None:
        source = np.arange(32 * 48, dtype=np.float32).reshape(32, 48) / 1024.0
        first = sample_scale_tiles(source)
        second = sample_scale_tiles(source)
        self.assertEqual(first.shape, (15, 256))
        np.testing.assert_array_equal(first, second)
        expected_first = source[:16, :16].reshape(256)[
            np.asarray(TENSOR_CORE_PERMUTATION)
        ]
        np.testing.assert_array_equal(first[0], expected_first)

    def test_final_layer_region_contract_is_exact_and_bounded(self) -> None:
        regions = {}
        position = 0
        for name in REGION_ORDER:
            position = (position + 255) & -256
            size = expected_region_bytes(name)
            regions[name] = {"offset": position, "bytes": size}
            position += size
        artifact_bytes = (position + 255) & -256
        self.assertEqual(artifact_bytes, 345_147_392)
        self.assertEqual(region_layout({"regions": regions}, artifact_bytes), regions)
        regions["down_descriptor"]["offset"] += 256
        with self.assertRaises(InvalidPlanError):
            region_layout({"regions": regions}, artifact_bytes)

    def test_fused_gate_up_inverse_before_split_matches_high_precision_reference(self) -> None:
        rng = np.random.default_rng(23)
        source = rng.normal(0.0, 0.05, size=(128, 1408))
        activation = rng.normal(0.0, 0.2, size=128)
        signs = deterministic_signs(128, 53, "fixture:gate_up:input")
        regularized = regularize_weight(
            source,
            signs,
            seed=53,
            domain="fixture:gate_up",
        )
        transformed_activation = blockwise_hadamard_right(
            (activation * regularized.suh).reshape(1, 128)
        ).reshape(128)
        transformed_output = transformed_activation @ regularized.transformed
        gate, up = inverse_gate_up_output(transformed_output, regularized.svh)
        reference = activation @ source
        np.testing.assert_allclose(gate, reference[:704], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(up, reference[704:], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            gelu_tanh_product(gate, up),
            gelu_tanh_product(reference[:704], reference[704:]),
            rtol=1e-11,
            atol=1e-11,
        )

    def test_hessian_and_regularization_fail_closed(self) -> None:
        with self.assertRaises(InvalidPlanError):
            finalize_hessian(np.eye(127), 1, seed=0, domain="bad")
        with self.assertRaises(InvalidPlanError):
            finalize_hessian(np.zeros((128, 128)), 1, seed=0, domain="bad")
        with self.assertRaises(InvalidPlanError):
            regularize_weight(np.zeros((128, 127)), np.ones(128), seed=0, domain="bad")
        with self.assertRaises(InvalidPlanError):
            quantize_matrix_ldlq(np.zeros((16, 16)), np.zeros((15, 15)), 3)


if __name__ == "__main__":
    unittest.main()
