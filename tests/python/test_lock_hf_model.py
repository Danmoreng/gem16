from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gem16_lock_hf_model", ROOT / "tools/lock_hf_model.py"
)
assert SPEC is not None and SPEC.loader is not None
lock_hf_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lock_hf_model)


class LockHuggingFaceModelTest(unittest.TestCase):
    def test_git_blob_oid_uses_canonical_blob_header(self) -> None:
        payload = b"model metadata\n"
        expected = hashlib.sha1(
            f"blob {len(payload)}\0".encode("ascii") + payload
        ).hexdigest()
        self.assertEqual(lock_hf_model.git_blob_oid(payload), expected)

    def test_lfs_entry_records_git_lfs_and_xet_identities(self) -> None:
        sibling = {
            "rfilename": "nested/model.safetensors",
            "size": 123,
            "blobId": "1" * 40,
            "lfs": {"sha256": "2" * 64, "size": 123, "pointerSize": 130},
        }
        with mock.patch.object(
            lock_hf_model, "fetch_xet_hash", return_value="3" * 64
        ):
            entry = lock_hf_model.create_entry(
                "owner/model", "4" * 40, sibling, 1024
            )
        self.assertEqual(
            entry,
            {
                "path": "nested/model.safetensors",
                "size": 123,
                "sha256": "2" * 64,
                "git_oid": "1" * 40,
                "lfs_oid": "2" * 64,
                "xet_hash": "3" * 64,
            },
        )

    def test_non_lfs_entry_requires_content_hash(self) -> None:
        sibling = {
            "rfilename": "config.json",
            "size": 12,
            "blobId": "1" * 40,
            "lfs": None,
        }
        with mock.patch.object(
            lock_hf_model, "fetch_inline_identity", return_value="2" * 64
        ) as fetch:
            entry = lock_hf_model.create_entry(
                "owner/model", "4" * 40, sibling, 1024
            )
        self.assertEqual(entry["sha256"], "2" * 64)
        self.assertEqual(entry["git_oid"], "1" * 40)
        fetch.assert_called_once()

    def test_build_lock_rejects_unsafe_repository_before_request(self) -> None:
        with mock.patch.object(lock_hf_model, "fetch_model_document") as fetch:
            with self.assertRaisesRegex(ValueError, "invalid Hugging Face repository"):
                lock_hf_model.build_lock(
                    "owner/model?download=1", "1" * 40, "https://example.com/terms"
                )
        fetch.assert_not_called()

    def test_build_lock_rejects_resolved_revision_drift(self) -> None:
        with mock.patch.object(
            lock_hf_model,
            "fetch_model_document",
            return_value={"sha": "2" * 40, "siblings": [{}]},
        ):
            with self.assertRaisesRegex(RuntimeError, "resolved revision differs"):
                lock_hf_model.build_lock(
                    "owner/model", "1" * 40, "https://example.com/terms"
                )

    def test_build_lock_rejects_private_repository(self) -> None:
        with mock.patch.object(
            lock_hf_model,
            "fetch_model_document",
            return_value={"sha": "1" * 40, "private": True, "siblings": [{}]},
        ):
            with self.assertRaisesRegex(RuntimeError, "private repository"):
                lock_hf_model.build_lock(
                    "owner/model", "1" * 40, "https://example.com/terms"
                )


if __name__ == "__main__":
    unittest.main()
