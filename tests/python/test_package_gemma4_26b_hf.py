import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/package_gemma4_26b_hf.py"
SPEC = importlib.util.spec_from_file_location("package_gemma4_26b_hf", MODULE_PATH)
assert SPEC and SPEC.loader
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


UPSTREAM_CARD = """---
license: apache-2.0
pipeline_tag: image-text-to-text
---

# Google upstream card

Exact upstream body.
"""


class PackageGemma426BHuggingFaceTest(unittest.TestCase):
    def test_target_card_links_quantized_base_and_retains_upstream_body(self):
        card = PACKAGE.model_card("target", UPSTREAM_CARD)
        self.assertIn("license: apache-2.0", card)
        self.assertIn(
            "base_model: google/gemma-4-26B-A4B-it-qat-q4_0-unquantized", card,
        )
        self.assertIn("base_model_relation: quantized", card)
        self.assertIn("https://github.com/Danmoreng/gem16", card)
        self.assertIn("# Google upstream card\n\nExact upstream body.", card)
        self.assertEqual(1, card.count("license: apache-2.0"))
        self.assertNotIn("pipeline_tag: image-text-to-text", card)

    def test_assistant_card_is_not_described_as_standalone(self):
        card = PACKAGE.model_card("assistant", UPSTREAM_CARD)
        self.assertIn("base_model_relation: quantized", card)
        self.assertIn("not a standalone chat model", card)
        self.assertIn(PACKAGE.TARGET_REPOSITORY, card)

    def test_generated_metadata_uses_current_qualified_context(self):
        metadata = PACKAGE.model_metadata(
            "target", {"artifact_content_sha256": "a" * 64},
        )
        self.assertEqual(86_016, metadata["maximum_context_tokens"])
        self.assertEqual(98_304, metadata["target_only_maximum_context_tokens"])

    def test_publication_metadata_includes_card_license_and_notice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_card = root / "upstream.md"
            source_card.write_text(UPSTREAM_CARD, encoding="utf-8")
            output = root / "package"
            output.mkdir()
            PACKAGE.write_publication_metadata(output, "target", source_card)

            self.assertEqual((ROOT / "LICENSE").read_bytes(), (output / "LICENSE").read_bytes())
            self.assertIn(PACKAGE.TARGET_SOURCE_REVISION, (output / "NOTICE").read_text())
            self.assertIn("Upstream model card", (output / "README.md").read_text())

    def test_current_hub_locks_still_pin_qualified_weight_hashes(self):
        target = json.loads(
            (ROOT / "models/gemma4-26b-gem16-target.lock.json").read_text()
        )
        assistant = json.loads(
            (ROOT / "models/gemma4-26b-gem16-assistant.lock.json").read_text()
        )
        target_model = next(item for item in target["files"] if item["path"] == "model.gem16")
        assistant_model = next(
            item for item in assistant["files"]
            if item["path"] == "model-00001-of-00001.safetensors"
        )
        self.assertEqual(PACKAGE.TARGET_IMAGE_BYTES, target_model["size"])
        self.assertEqual(PACKAGE.TARGET_IMAGE_SHA256, target_model["sha256"])
        self.assertEqual(
            "4d3ce2102ad0631d9e7e0586be0b108d5789cbc5b90d21b4c50613979228d927",
            assistant_model["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
