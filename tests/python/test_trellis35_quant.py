from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import numpy as np

from tools.gem16_compile.common import InvalidPlanError
from tools.gem16_compile.trellis35_quant import (
    CODEBOOK_3INST,
    CODEBOOK_MCG,
    CODEBOOK_MUL1,
    TENSOR_CORE_INVERSE_PERMUTATION,
    TENSOR_CORE_PERMUTATION,
    decode_codebook_half_bits,
    encoded_tile_from_branches,
    gelu_tanh_product,
    hadamard_128,
    inverse_gate_up_output,
    inverse_permutation,
    pack_trellis_tile,
    quantize_trellis_tile,
    reconstruct_matrix,
    regularize_matrix,
    unpack_trellis_tile,
    validate_encoded_tile,
)


ROOT = Path(__file__).resolve().parents[2]


class Trellis35QuantOracleTest(unittest.TestCase):
    def test_tensor_core_permutation_is_exact_and_bijective(self) -> None:
        self.assertEqual(len(TENSOR_CORE_PERMUTATION), 256)
        self.assertEqual(TENSOR_CORE_PERMUTATION[:8], (0, 16, 128, 144, 8, 24, 136, 152))
        self.assertEqual(TENSOR_CORE_PERMUTATION[-8:], (103, 119, 231, 247, 111, 127, 239, 255))
        self.assertEqual(
            tuple(TENSOR_CORE_PERMUTATION[index] for index in TENSOR_CORE_INVERSE_PERMUTATION),
            tuple(range(256)),
        )

    def test_k3_k4_packing_round_trips_and_has_fixed_hashes(self) -> None:
        expected = {
            3: (96, "bdc93f20bc6959aced975d522cdec35be573fd0fdad15581bf256de045188640"),
            4: (128, "8ed6ecf035a536fd677f9dd6d9b34d7ca2db2ab38c02d0e91aefaa177e5db6e2"),
        }
        for rate, (size, digest) in expected.items():
            branches = tuple((index * 5 + index // 7 + 3) & ((1 << rate) - 1) for index in range(256))
            encoded = encoded_tile_from_branches(branches, rate)
            payload = pack_trellis_tile(encoded, rate)
            self.assertEqual(len(payload), size)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
            self.assertEqual(unpack_trellis_tile(payload, rate), encoded)

    def test_tail_biting_and_malformed_payloads_fail_closed(self) -> None:
        encoded = list(encoded_tile_from_branches([0] * 256, 3))
        encoded[17] ^= 1 << 8
        with self.assertRaises(InvalidPlanError):
            validate_encoded_tile(encoded, 3)
        with self.assertRaises(InvalidPlanError):
            pack_trellis_tile([0] * 255, 3)
        with self.assertRaises(InvalidPlanError):
            unpack_trellis_tile(bytes(95), 3)
        with self.assertRaises(InvalidPlanError):
            encoded_tile_from_branches([8] * 256, 3)
        with self.assertRaises(InvalidPlanError):
            encoded_tile_from_branches([0] * 256, 2)

    def test_codebook_decode_matches_pinned_cuda_golden_half_bits(self) -> None:
        states = (0, 1, 2, 3, 0x1234, 0x7FFF, 0xFFFF)
        expected = {
            CODEBOOK_3INST: (0x3A25, 0xBB5B, 0x3B74, 0x34B6, 0xC066, 0xBA96, 0xB110),
            CODEBOOK_MCG: (0x3F60, 0x304E, 0xBA13, 0x3AB8, 0x3E0F, 0x4172, 0x3ACD),
            CODEBOOK_MUL1: (0xC2E8, 0x3921, 0xB72D, 0x3239, 0xB89E, 0x3FCA, 0xB936),
        }
        for codebook_id, half_bits in expected.items():
            self.assertEqual(
                tuple(decode_codebook_half_bits(state, codebook_id) for state in states),
                half_bits,
            )

    def test_license_and_provenance_are_pinned(self) -> None:
        license_path = ROOT / "third_party/exllamav3_quant/LICENSE"
        quantizer_path = (
            ROOT / "third_party/exllamav3_quant/quant/quantize_tiles_kernel.cuh"
        )
        provenance = (ROOT / "third_party/exllamav3_quant/PROVENANCE.md").read_text(encoding="utf-8")
        self.assertEqual(
            hashlib.sha256(license_path.read_bytes().rstrip(b"\n")).hexdigest(),
            "27a32b6263fcd96c79d3beeecf221c4366780bdf15ad51986f48650bd7369bff",
        )
        self.assertIn("0c49587a7c235e6303a6bbedc8b665272ad3a2ea", provenance)
        self.assertIn("MIT", provenance)
        self.assertEqual(
            hashlib.sha256(quantizer_path.read_bytes()).hexdigest(),
            "e687fa4b9cb6905bafe11d36431129152f7428c22fa2089410396a7683b75561",
        )
        self.assertIn("only lane 0 publishes", provenance)

    def test_inverse_permutation_rejects_duplicates(self) -> None:
        with self.assertRaises(InvalidPlanError):
            inverse_permutation([0] * 256)

    def test_cpu_viterbi_is_deterministic_tail_biting_and_improves_error(self) -> None:
        values = tuple(np.sin(np.arange(256, dtype=np.float64) * 0.173) * 1.25)
        for rate in (3, 4):
            first = quantize_trellis_tile(values, rate)
            second = quantize_trellis_tile(values, rate)
            self.assertEqual(first.encoded, second.encoded)
            self.assertEqual(first.reconstructed, second.reconstructed)
            self.assertEqual(first.squared_error, second.squared_error)
            self.assertEqual(validate_encoded_tile(first.encoded, rate), first.encoded)
            self.assertLess(first.squared_error, float(np.dot(values, values)))
            self.assertEqual(unpack_trellis_tile(pack_trellis_tile(first.encoded, rate), rate), first.encoded)

    def test_cpu_viterbi_matches_pinned_cuda_k3_k4_golden_states(self) -> None:
        values = tuple((((index * 37) % 61) - 30) / 16.0 for index in range(256))
        expected = {
            3: "36cc860b34245853d67f55f24fbd0bb1281529c2c5310866d33b56f58b5a0597",
            4: "0b3303bc9bb2d677a0a65faa835a85d06fc08da03ce35e56bfdfde7546a7e7b3",
        }
        for rate, digest in expected.items():
            encoded = quantize_trellis_tile(values, rate).encoded
            payload = b"".join(value.to_bytes(2, "little") for value in encoded)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)

    def test_hadamard_fixture_reconstruction_round_trip(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(0.0, 0.2, size=(256, 256))
        suh = rng.uniform(0.5, 1.5, size=256)
        svh = rng.uniform(0.5, 1.5, size=256)
        transformed = regularize_matrix(source, suh, svh)
        reconstructed = reconstruct_matrix(transformed, suh, svh)
        np.testing.assert_allclose(reconstructed, source, rtol=1e-12, atol=1e-12)
        identity = hadamard_128() @ hadamard_128().T
        np.testing.assert_allclose(identity, np.eye(128), rtol=1e-14, atol=1e-14)

    def test_gate_up_inverse_crosses_boundary_before_split(self) -> None:
        transformed = np.zeros(1408, dtype=np.float64)
        transformed[700] = 1.0
        gate, up = inverse_gate_up_output(transformed, np.ones(1408, dtype=np.float64))
        self.assertTrue(np.any(gate != 0.0))
        self.assertTrue(np.any(up != 0.0))
        product = gelu_tanh_product(gate, up)
        self.assertEqual(product.shape, (704,))
        self.assertTrue(np.isfinite(product).all())

    def test_transform_and_gate_up_shape_errors_fail_closed(self) -> None:
        with self.assertRaises(InvalidPlanError):
            reconstruct_matrix(np.zeros((128, 127)), np.ones(128), np.ones(127))
        with self.assertRaises(InvalidPlanError):
            inverse_gate_up_output(np.zeros(1407), np.ones(1408))


if __name__ == "__main__":
    unittest.main()
