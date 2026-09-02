import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/package_gemma4_26b_vision_hf.py"
SPEC = importlib.util.spec_from_file_location(
    "package_gemma4_26b_vision_hf", MODULE_PATH,
)
assert SPEC and SPEC.loader
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


class PackageGemma426BVisionHuggingFaceTest(unittest.TestCase):
    def test_profile_identity_is_stable_across_decode_modes(self):
        manifest = PACKAGE.compatibility_manifest(
            {"path": "trellis35"},
            {"path": "assistant"},
            {"path": "vision"},
        )
        self.assertEqual(2, manifest["schema_version"])
        profiles = manifest["compatible_profiles"]
        vision = [
            profile for profile in profiles
            if profile["profile_id"] ==
            "gemma4-26b-a4b-trellis35-vision-fp8"
        ]
        self.assertEqual(2, len(vision))
        self.assertEqual(
            {"ordinary", "fixed-d2"},
            {profile["decode_mode"] for profile in vision},
        )
        self.assertEqual(
            {"production_candidate"},
            {profile["qualification_state"] for profile in vision},
        )
        fixed_d2 = next(
            profile for profile in vision
            if profile["decode_mode"] == "fixed-d2"
        )
        self.assertEqual(
            ["trellis35_target", "fp8_vision", "fixed_d2_assistant"],
            fixed_d2["components"],
        )
        self.assertEqual(64, len(manifest["content_sha256"]))


if __name__ == "__main__":
    unittest.main()
