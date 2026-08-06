from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gem16_fetch_model", ROOT / "tools/fetch_model.py"
)
assert SPEC is not None and SPEC.loader is not None
fetch_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_model)


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, status: int) -> None:
        super().__init__(payload)
        self.status = status


class FetchModelTest(unittest.TestCase):
    def test_file_source_can_override_repository_and_revision(self) -> None:
        lock = {
            "repository": "unsloth/model",
            "revision": "1" * 40,
        }
        entry = {
            "path": "tokenizer_config.json",
            "source": {
                "repository": "google/gemma-4-12B-it",
                "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
                "path": "tokenizer_config.json",
            },
        }
        self.assertEqual(
            fetch_model.resolve_source(lock, entry),
            (
                "google/gemma-4-12B-it",
                "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
                "tokenizer_config.json",
            ),
        )

    def test_source_change_replaces_existing_file_without_resuming_it(self) -> None:
        payload = b"official-google-tokenizer-config"
        expected_hash = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "tokenizer_config.json"
            destination.write_bytes(b"older-unsloth-file")
            response = FakeResponse(payload, 200)
            with mock.patch.object(
                fetch_model.urllib.request, "urlopen", return_value=response
            ) as urlopen:
                fetch_model.download(
                    "https://huggingface.co/google/model/resolve/" + "7" * 40 +
                    "/tokenizer_config.json",
                    destination,
                    len(payload),
                    expected_hash,
                )

            self.assertEqual(destination.read_bytes(), payload)
            request = urlopen.call_args.args[0]
            self.assertNotIn("Range", request.headers)
            self.assertFalse(
                destination.with_name(destination.name + ".incomplete").exists()
            )
            self.assertFalse(
                destination.with_name(destination.name + ".incomplete.json").exists()
            )

    def test_lock_validation_rejects_mutable_revision_and_bad_hash(self) -> None:
        base = {
            "schema_version": 2,
            "repository": "owner/model",
            "revision": "1" * 40,
            "files": [
                {
                    "path": "nested/model.safetensors",
                    "size": 4,
                    "sha256": "2" * 64,
                    "git_oid": "3" * 40,
                }
            ],
        }
        self.assertEqual(fetch_model.validate_lock(base), base["files"])

        mutable = dict(base, revision="main")
        with self.assertRaisesRegex(ValueError, "full immutable commit SHA"):
            fetch_model.validate_lock(mutable)

        bad_hash = dict(base, files=[dict(base["files"][0], sha256="short")])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            fetch_model.validate_lock(bad_hash)

        bad_source = dict(
            base,
            files=[
                dict(
                    base["files"][0],
                    source={"repository": "owner/model?download=1"},
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "source repository"):
            fetch_model.validate_lock(bad_source)

    def test_nested_paths_are_allowed_but_escape_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            self.assertEqual(
                fetch_model.safe_target(root, ".eval_results/result.json"),
                root / ".eval_results" / "result.json",
            )
            for unsafe in ("../escape", "/absolute", "a\\escape", "a/../../escape"):
                with self.subTest(unsafe=unsafe):
                    with self.assertRaisesRegex(ValueError, "unsafe"):
                        fetch_model.safe_target(root, unsafe)

            outside = Path(temporary) / "outside"
            outside.mkdir()
            (root / "linked").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes through a symlink"):
                fetch_model.safe_target(root, "linked/file")

    def test_capacity_preflight_rejects_budget_before_download(self) -> None:
        lock = {
            "schema_version": 2,
            "repository": "owner/model",
            "revision": "1" * 40,
            "files": [
                {
                    "path": "model-1.safetensors",
                    "size": 7,
                    "sha256": "2" * 64,
                    "lfs_oid": "2" * 64,
                },
                {
                    "path": "model-2.safetensors",
                    "size": 11,
                    "sha256": "3" * 64,
                    "lfs_oid": "3" * 64,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = fetch_model.validate_lock(lock)
            with self.assertRaisesRegex(RuntimeError, "configured maximum"):
                fetch_model.capacity_preflight(
                    lock, entries, root, max_new_bytes=17, min_free_bytes=0,
                    available_bytes=100,
                )
            with self.assertRaisesRegex(RuntimeError, "only 20 bytes are free"):
                fetch_model.capacity_preflight(
                    lock, entries, root, max_new_bytes=18, min_free_bytes=3,
                    available_bytes=20,
                )
            self.assertEqual(
                fetch_model.capacity_preflight(
                    lock, entries, root, max_new_bytes=18, min_free_bytes=2,
                    available_bytes=20,
                ),
                18,
            )

    def test_multi_shard_view_links_do_not_duplicate_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "blob"
            payload.write_bytes(b"model-shard")
            first = root / "view" / "nested" / "model-1.safetensors"
            second = root / "other-view" / "model-1.safetensors"
            fetch_model.link_file(payload, first)
            fetch_model.link_file(payload, second)
            self.assertTrue(payload.samefile(first))
            self.assertTrue(payload.samefile(second))
            self.assertEqual(payload.stat().st_ino, first.stat().st_ino)
            self.assertEqual(payload.stat().st_ino, second.stat().st_ino)

    def test_partial_resume_requires_matching_download_identity(self) -> None:
        payload = b"resume-this-download"
        split = 7
        expected_hash = hashlib.sha256(payload).hexdigest()
        url = "https://huggingface.co/repository/resolve/" + "a" * 40 + "/file"
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "file"
            partial = destination.with_name(destination.name + ".incomplete")
            metadata = destination.with_name(destination.name + ".incomplete.json")
            partial.write_bytes(payload[:split])
            metadata.write_text(
                json.dumps(
                    {"url": url, "size": len(payload), "sha256": expected_hash},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            response = FakeResponse(payload[split:], 206)
            with mock.patch.object(
                fetch_model.urllib.request, "urlopen", return_value=response
            ) as urlopen:
                fetch_model.download(url, destination, len(payload), expected_hash)

            self.assertEqual(destination.read_bytes(), payload)
            request = urlopen.call_args.args[0]
            self.assertEqual(request.headers["Range"], f"bytes={split}-")


if __name__ == "__main__":
    unittest.main()
