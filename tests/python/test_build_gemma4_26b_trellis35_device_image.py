from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.build_gemma4_26b_trellis35_device_image import (
    InvalidPlanError,
    add_content_hash,
    canonical_content_sha256,
    checked_shape,
    parse_sha256s,
    safe_relative,
    stream_concat,
    validate_tensor_layout,
    verify_sha256_inventory,
)


class Trellis35DeviceImageTest(unittest.TestCase):
    def test_stream_concat_is_exact_and_independently_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.bin"
            second = root / "second.bin"
            partial = root / "model.gem16.partial"
            first.write_bytes(b"alpha")
            second.write_bytes(b"beta-gamma")
            expected = first.read_bytes() + second.read_bytes()
            digest = stream_concat((first, second), partial, len(expected), 3)
            self.assertEqual(partial.read_bytes(), expected)
            self.assertEqual(digest, hashlib.sha256(expected).hexdigest())

    def test_stream_concat_never_publishes_final_on_short_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            partial = root / "model.gem16.partial"
            final = root / "model.gem16"
            source.write_bytes(b"short")
            with self.assertRaises(InvalidPlanError):
                stream_concat((source,), partial, 8, 2)
            self.assertFalse(final.exists())
            self.assertTrue(partial.exists())

    def test_stream_concat_refuses_existing_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            partial = root / "model.gem16.partial"
            source.write_bytes(b"source")
            partial.write_bytes(b"prior")
            with self.assertRaises(InvalidPlanError):
                stream_concat((source,), partial, source.stat().st_size)

    def test_second_pass_digest_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            partial = root / "model.gem16.partial"
            source.write_bytes(b"payload")
            with mock.patch(
                "tools.build_gemma4_26b_trellis35_device_image.sha256_file",
                return_value="0" * 64,
            ):
                with self.assertRaises(InvalidPlanError):
                    stream_concat((source,), partial, source.stat().st_size)

    def test_sha256_inventory_rejects_duplicate_and_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = "a" * 64
            (root / "SHA256SUMS").write_text(
                f"{digest}  payload.bin\n{digest}  payload.bin\n",
                encoding="ascii",
            )
            with self.assertRaises(InvalidPlanError):
                parse_sha256s(root, expected_entries=2)
            (root / "SHA256SUMS").write_text(
                f"{digest}  ../payload.bin\n", encoding="ascii"
            )
            with self.assertRaises(InvalidPlanError):
                parse_sha256s(root, expected_entries=1)

    def test_sha256_inventory_rejects_wrong_payload_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.bin").write_bytes(b"payload")
            for name in ("SHA256SUMS", "README.md", "LICENSE", "NOTICE"):
                (root / name).write_bytes(b"fixture")
            with self.assertRaisesRegex(InvalidPlanError, "SHA-256 mismatch"):
                verify_sha256_inventory(root, {"payload.bin": "0" * 64})

    def test_sha256_inventory_rejects_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"payload"
            (root / "payload.bin").write_bytes(payload)
            for name in ("SHA256SUMS", "README.md", "LICENSE", "NOTICE"):
                (root / name).write_bytes(b"fixture")
            (root / "unexpected.bin").write_bytes(b"unexpected")
            with self.assertRaisesRegex(InvalidPlanError, "inventory changed"):
                verify_sha256_inventory(
                    root, {"payload.bin": hashlib.sha256(payload).hexdigest()}
                )

    def test_safe_relative_rejects_windows_and_parent_escape(self) -> None:
        for value in ("../model.gem16", "/model.gem16", "a\\b", ""):
            with self.subTest(value=value):
                with self.assertRaises(InvalidPlanError):
                    safe_relative(value, "fixture")

    def test_tensor_shape_and_layout_validation(self) -> None:
        logical = checked_shape([8, 32], "logical shape")
        physical = checked_shape([8, 16], "physical shape")
        validate_tensor_layout(
            "U8", "sm120_row8_k64", logical, physical, 128
        )
        with self.assertRaises(InvalidPlanError):
            checked_shape([8, 0], "logical shape")
        with self.assertRaises(InvalidPlanError):
            validate_tensor_layout(
                "BF16", "sm120_row8_k64", logical, physical, 256
            )
        with self.assertRaises(InvalidPlanError):
            validate_tensor_layout(
                "U8", "sm120_row8_k64", logical, [8, 15], 120
            )

    def test_canonical_content_hash_detects_metadata_change(self) -> None:
        document = add_content_hash(
            {"schema_version": 1, "value": [1, 2, 3]},
            "checkpoint_content_sha256",
        )
        expected = document["checkpoint_content_sha256"]
        self.assertEqual(
            canonical_content_sha256(document, "checkpoint_content_sha256"),
            expected,
        )
        document["value"] = [3, 2, 1]
        with self.assertRaises(InvalidPlanError):
            canonical_content_sha256(document, "checkpoint_content_sha256")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_sha256_inventory_path_cannot_be_symlinked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.bin"
            target.write_bytes(b"payload")
            (root / "link.bin").symlink_to(target)
            with self.assertRaises(InvalidPlanError):
                from tools.build_gemma4_26b_trellis35_device_image import safe_file

                safe_file(root, "link.bin", "fixture")


if __name__ == "__main__":
    unittest.main()
