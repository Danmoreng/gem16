from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest

from tools.gem16_compile.common import InvalidPlanError, canonical_json_bytes
from tools.generate_gemma4_26b_trellis35_checkpoint import cleanup_layer_intermediates
from tools.package_gemma4_26b_trellis35_checkpoint import align, tensor_plans
from tools.verify_gemma4_26b_trellis35_checkpoint import (
    _safe_file,
    _verify_content_hash,
    _verify_zero_extent,
)


class Trellis35CheckpointTest(unittest.TestCase):
    def test_packager_excludes_routed_experts_and_compacts_alignment(self) -> None:
        compilation = {
            "tensors": [
                {"output_name": "a", "role": "attention", "byte_length": 17},
                {"output_name": "b", "role": "routed_expert_gate_up", "byte_length": 33},
                {"output_name": "c", "role": "final_norm", "byte_length": 65},
            ]
        }
        plans, extent = tensor_plans(
            compilation,
            expected_tensor_count=3,
            expected_source_bytes=768,
            expected_selected_count=2,
            expected_non_routed_bytes=512,
        )
        self.assertEqual([item["name"] for item in plans], ["a", "c"])
        self.assertEqual([item["source_image_offset"] for item in plans], [0, 512])
        self.assertEqual([item["destination_offset"] for item in plans], [0, 256])
        self.assertEqual(extent, 512)

    def test_checkpoint_content_hash_is_canonical_and_fail_closed(self) -> None:
        import hashlib
        content = {"schema_version": 1, "value": [3, 4]}
        content["checkpoint_content_sha256"] = hashlib.sha256(
            canonical_json_bytes(content)
        ).hexdigest()
        self.assertEqual(_verify_content_hash(content, "fixture"), content["checkpoint_content_sha256"])
        content["value"] = [4, 3]
        with self.assertRaises(InvalidPlanError):
            _verify_content_hash(content, "fixture")

    def test_safe_checkpoint_path_rejects_symlink_and_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "payload.bin").write_bytes(b"payload")
            self.assertEqual(_safe_file(root, "payload.bin", "fixture"), root / "payload.bin")
            (root / "link.bin").symlink_to(root / "payload.bin")
            with self.assertRaises(InvalidPlanError):
                _safe_file(root, "link.bin", "fixture")
            with self.assertRaises(InvalidPlanError):
                _safe_file(root, "../payload.bin", "fixture")

    def test_cleanup_is_bounded_to_layer_intermediates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layer = Path(directory)
            (layer / "candidates").mkdir()
            (layer / "candidates" / "candidate.bin").write_bytes(b"x")
            for name in ("calibration.bin", "gate-inputs.f32", "down-inputs.f32"):
                (layer / name).write_bytes(b"x")
            (layer / "verification.json").write_bytes(b"keep")
            cleanup_layer_intermediates(layer)
            self.assertFalse((layer / "candidates").exists())
            self.assertEqual((layer / "verification.json").read_bytes(), b"keep")

    def test_zero_extent_verification(self) -> None:
        stream = io.BytesIO(b"a\0\0b")
        _verify_zero_extent(stream, 1, 2)
        with self.assertRaises(InvalidPlanError):
            _verify_zero_extent(stream, 0, 1)

    def test_alignment_is_exact(self) -> None:
        self.assertEqual([align(value) for value in (0, 1, 255, 256, 257)], [0, 256, 256, 256, 512])


if __name__ == "__main__":
    unittest.main()
