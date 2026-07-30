from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.hf_cache import hub_cache_root, locked_snapshot_path


class HuggingFaceCacheTest(unittest.TestCase):
    def test_hugging_face_environment_precedence(self) -> None:
        home = Path("/home/tester")
        self.assertEqual(
            Path("/explicit/hub"),
            hub_cache_root({"HF_HUB_CACHE": "/explicit/hub", "HF_HOME": "/hf"}, home),
        )
        self.assertEqual(Path("/hf/hub"), hub_cache_root({"HF_HOME": "/hf"}, home))
        self.assertEqual(
            Path("/xdg/huggingface/hub"),
            hub_cache_root({"XDG_CACHE_HOME": "/xdg"}, home),
        )
        self.assertEqual(home / ".cache/huggingface/hub", hub_cache_root({}, home))

    def test_external_locked_file_uses_composed_cache_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "model.lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "repository": "owner/model",
                        "revision": "1" * 40,
                        "files": [
                            {
                                "path": "tokenizer_config.json",
                                "source": {
                                    "repository": "google/model",
                                    "revision": "2" * 40,
                                    "path": "tokenizer_config.json",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                root / ".gem16" / "snapshots" / f"owner--model--{'1' * 40}",
                locked_snapshot_path(lock_path, root),
            )


if __name__ == "__main__":
    unittest.main()
