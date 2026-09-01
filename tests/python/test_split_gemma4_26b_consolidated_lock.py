from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/split_gemma4_26b_consolidated_lock.py"
SPEC = importlib.util.spec_from_file_location("split_26b_lock", SCRIPT)
assert SPEC and SPEC.loader
SPLITTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPLITTER)


def entry(path: str) -> dict[str, object]:
    return {
        "path": path,
        "size": 1,
        "sha256": "a" * 64,
        "git_oid": "b" * 40,
    }


def full_lock() -> dict[str, object]:
    paths: list[str] = []
    for component, (prefix, _filename) in SPLITTER.COMPONENTS.items():
        paths.extend(
            prefix + relative
            for relative in SPLITTER.REQUIRED_COMPONENT_FILES[component]
        )
    return {
        "schema_version": 2,
        "repository": SPLITTER.REPOSITORY,
        "revision": "c" * 40,
        "resolved_at_utc": "2026-09-01T00:00:00Z",
        "source_url": f"https://huggingface.co/{SPLITTER.REPOSITORY}/tree/{'c' * 40}",
        "terms_url": "https://ai.google.dev/gemma/terms",
        "files": [entry(path) for path in sorted(paths)],
    }


class ConsolidatedLockSplitTest(unittest.TestCase):
    def test_subdirectory_component_becomes_flat_composed_view(self) -> None:
        lock = full_lock()
        component = SPLITTER.component_lock(lock, "vision", "vision/")
        self.assertEqual(component["repository"], SPLITTER.REPOSITORY)
        self.assertEqual(component["component_path"], "vision")
        self.assertEqual(
            {item["path"] for item in component["files"]},
            SPLITTER.REQUIRED_COMPONENT_FILES["vision"],
        )
        for item in component["files"]:
            self.assertEqual(item["source"]["repository"], SPLITTER.REPOSITORY)
            self.assertEqual(item["source"]["revision"], "c" * 40)
            self.assertEqual(item["source"]["path"], "vision/" + item["path"])

    def test_root_target_uses_native_snapshot_paths(self) -> None:
        component = SPLITTER.component_lock(full_lock(), "target", "")
        self.assertEqual(component["component_path"], ".")
        self.assertTrue(all("source" not in item for item in component["files"]))

    def test_missing_required_file_is_rejected(self) -> None:
        lock = full_lock()
        lock["files"] = [
            item for item in lock["files"] if item["path"] != "vision/vision.gem16"
        ]
        with self.assertRaisesRegex(SPLITTER.SplitError, "missing required files"):
            SPLITTER.component_lock(lock, "vision", "vision/")


if __name__ == "__main__":
    unittest.main()
