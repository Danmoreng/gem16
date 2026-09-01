import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "tools/generate_native_studio_model_catalog.py"
GENERATED_PATH = ROOT / "nativeStudio/src/model_catalog.generated.h"
SPEC = importlib.util.spec_from_file_location("generate_model_catalog", GENERATOR_PATH)
assert SPEC and SPEC.loader
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class NativeStudioModelCatalogTest(unittest.TestCase):
    def test_generated_catalog_is_current(self):
        self.assertEqual(GENERATOR.render(), GENERATED_PATH.read_text(encoding="utf-8"))

    def test_generated_catalog_contains_all_six_locked_components(self):
        generated = GENERATED_PATH.read_text(encoding="utf-8")
        locks = (
            "gemma4-12b-nvfp4.lock.json",
            "gemma4-12b-mtp-assistant.lock.json",
            "gemma4-26b-gem16-target.lock.json",
            "gemma4-26b-gem16-assistant.lock.json",
            "gemma4-26b-trellis35-target.lock.json",
            "gemma4-26b-vision-fp8.lock.json",
        )
        for lock_name in locks:
            lock = json.loads((ROOT / "models" / lock_name).read_text())
            self.assertIn(lock["repository"], generated)
            self.assertIn(lock["revision"], generated)
            for item in lock["files"]:
                self.assertIn(item["path"], generated)
                self.assertIn(item["sha256"], generated)
                self.assertIn(item.get("lfs_oid") or item["git_oid"], generated)

    def test_12b_external_tokenizer_stays_in_google_repository(self):
        generated = GENERATED_PATH.read_text(encoding="utf-8")
        self.assertIn('"google/gemma-4-12B-it"', generated)
        catalog_source = ROOT / "nativeStudio/src/model_catalog.cpp"
        self.assertIn('".gem16/snapshots"', catalog_source.read_text())

    def test_generator_rejects_unsafe_lock_paths(self):
        lock = {
            "schema_version": 1,
            "repository": "owner/model",
            "revision": "a" * 40,
            "files": [
                {
                    "path": "../escape",
                    "size": 1,
                    "sha256": "b" * 64,
                    "git_oid": "c" * 40,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.lock.json"
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe relative path"):
                GENERATOR.load_lock(path)


if __name__ == "__main__":
    unittest.main()
