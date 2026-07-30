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
