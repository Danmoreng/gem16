from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.fetch_model import exact_hex, validate_lock


ROOT = Path(__file__).resolve().parents[2]
MODEL_LOCKS = {
    "gemma4-26b-qat-bf16.lock.json": (
        "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
        "f1e06dc520982d9b9edd76859fdb7ab209449949",
        51_644_341_801,
    ),
    "gemma4-26b-base-bf16.lock.json": (
        "google/gemma-4-26B-A4B-it",
        "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        51_644_340_088,
    ),
    "gemma4-26b-qat-q4_0.lock.json": (
        "google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
        "d1c082be9cf3c8a514acf63b8761f4b41935842e",
        15_634_222_799,
    ),
    "gemma4-26b-unsloth-nvfp4.lock.json": (
        "unsloth/gemma-4-26B-A4B-it-NVFP4",
        "20df0542b1a86ce19f495ac2eca2c7c12bce82f9",
        16_942_099_391,
    ),
}


class Gemma426BLocksTest(unittest.TestCase):
    def test_product_components_share_one_immutable_repository_revision(self) -> None:
        repository = "danmoreng/gemma-4-26B-A4B-it-GEM16"
        revision = "6de2a057f11332420819f8e6efd08e42d7a03bc7"
        components = {
            "gemma4-26b-gem16-target.lock.json": ("target", ""),
            "gemma4-26b-trellis35-target.lock.json": ("trellis35", "trellis35/"),
            "gemma4-26b-gem16-assistant.lock.json": ("assistant", "assistant/"),
            "gemma4-26b-vision-fp8.lock.json": ("vision", "vision/"),
        }
        for filename, (component, prefix) in components.items():
            with self.subTest(filename=filename):
                lock = json.loads((ROOT / "models" / filename).read_text())
                validate_lock(lock)
                self.assertEqual(lock["repository"], repository)
                self.assertEqual(lock["revision"], revision)
                self.assertEqual(lock["component"], component)
                for entry in lock["files"]:
                    if not prefix:
                        self.assertNotIn("source", entry)
                        continue
                    self.assertEqual(entry["source"]["repository"], repository)
                    self.assertEqual(entry["source"]["revision"], revision)
                    self.assertEqual(
                        entry["source"]["path"], prefix + entry["path"]
                    )

    def test_model_locks_pin_complete_expected_snapshots(self) -> None:
        for filename, (repository, revision, total_bytes) in MODEL_LOCKS.items():
            with self.subTest(filename=filename):
                document = json.loads(
                    (ROOT / "models" / filename).read_text(encoding="utf-8")
                )
                entries = validate_lock(document)
                self.assertEqual(document["repository"], repository)
                self.assertEqual(document["revision"], revision)
                self.assertEqual(sum(int(entry["size"]) for entry in entries), total_bytes)
                self.assertTrue(document["terms_url"].startswith("https://"))
                self.assertTrue(all("git_oid" in entry for entry in entries))
                self.assertTrue(
                    all(
                        "lfs_oid" in entry and "xet_hash" in entry
                        for entry in entries
                        if int(entry["size"]) > 64 * 1024 * 1024
                    )
                )

    def test_reference_sources_have_full_commits_and_required_oracles(self) -> None:
        document = json.loads(
            (ROOT / "models/gemma4-26b-reference-sources.lock.json").read_text(
                encoding="utf-8"
            )
        )
        references = {entry["name"]: entry for entry in document["references"]}
        required = {
            "transformers",
            "vllm",
            "flashinfer",
            "compressed-tensors",
            "cutlass",
            "llama.cpp",
            "imp",
            "accelerate",
            "safetensors",
            "pytorch",
            "huggingface_hub",
        }
        self.assertEqual(set(references), required)
        for name, entry in references.items():
            with self.subTest(name=name):
                exact_hex(entry["revision"], 40, f"revision for {name}")
                self.assertTrue(entry["repository"].startswith("https://github.com/"))
                self.assertTrue(entry["license"])
                self.assertTrue(entry["intended_use"])


if __name__ == "__main__":
    unittest.main()
