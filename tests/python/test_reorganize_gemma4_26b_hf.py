import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/reorganize_gemma4_26b_hf.py"
SPEC = importlib.util.spec_from_file_location(
    "reorganize_gemma4_26b_hf", MODULE_PATH,
)
assert SPEC and SPEC.loader
REORGANIZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REORGANIZE
SPEC.loader.exec_module(REORGANIZE)


class ReorganizeGemma426BHuggingFaceTest(unittest.TestCase):
    def test_all_payload_names_include_model_and_real_format(self):
        names = [component.payload_name for component in REORGANIZE.COMPONENTS]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertTrue(name.startswith("gemma-4-26b-a4b-it-"), name)
            self.assertNotIn("model.gem16", name)
        self.assertIn("trellis35-w4a8", names[0])
        self.assertIn("vision-fp8-e4m3fn", names[1])
        self.assertIn("assistant-hybrid-nvfp4-fp8-bf16", names[2])
        self.assertIn("target-hybrid-nvfp4-fp8-bf16", names[3])

    def test_plan_is_non_destructive_to_source_revision(self):
        plan = REORGANIZE.build_plan()
        self.assertEqual(REORGANIZE.SOURCE_REVISION, plan["source_revision"])
        self.assertEqual(4, len([
            item for item in plan["copies"]
            if item["destination"].endswith((".gem16", ".safetensors"))
        ]))
        self.assertIn("model.gem16", plan["delete_files"])
        self.assertTrue(all(
            item["destination"].startswith(("components/", "internal/"))
            for item in plan["copies"]
        ))

    def test_profiles_publish_only_compact_vision(self):
        profiles = REORGANIZE.profiles_document()
        self.assertEqual(1, len(profiles["public_profiles"]))
        self.assertEqual(
            "gemma4-26b-a4b-compact-vision",
            profiles["public_profiles"][0]["profile_id"],
        )
        self.assertEqual("regression and rollback",
                         profiles["internal_profiles"][0]["purpose"])
        self.assertIn("profiles.json", REORGANIZE.notice())
        self.assertNotIn("gem16_components.json", REORGANIZE.notice())

    def test_assistant_index_is_rewritten_to_descriptive_payload(self):
        component = REORGANIZE.COMPONENTS[2]
        source = {
            "metadata": {"total_size": component.payload_bytes},
            "weight_map": {"a": component.source_payload, "b": component.source_payload},
        }
        result = REORGANIZE.normalized_index(source, component)
        self.assertEqual(
            {component.payload_name}, set(result["weight_map"].values())
        )
        self.assertEqual({component.source_payload}, set(source["weight_map"].values()))

    def test_remote_preflight_uses_lfs_metadata_not_payload_hashing(self):
        siblings = []
        for item in REORGANIZE.build_plan()["copies"]:
            component = next(
                (candidate for candidate in REORGANIZE.COMPONENTS
                 if item["source"] == REORGANIZE.source_path(
                     candidate, candidate.source_payload
                 )),
                None,
            )
            siblings.append(SimpleNamespace(
                rfilename=item["source"],
                size=component.payload_bytes if component else 1,
                lfs=(SimpleNamespace(sha256=component.payload_sha256)
                     if component else None),
            ))
        REORGANIZE.validate_remote_files(siblings)


if __name__ == "__main__":
    unittest.main()
