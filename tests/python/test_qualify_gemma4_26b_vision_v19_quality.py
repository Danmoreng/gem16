#!/usr/bin/env python3
"""Host tests for the V19 bounded Vision quality qualifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/qualify_gemma4_26b_vision_v19_quality.py"
SPEC = importlib.util.spec_from_file_location("qualify_vision_v19_quality", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VisionV19QualityTest(unittest.TestCase):
    def test_repository_suite_is_valid_and_complete(self) -> None:
        document, assets = MODULE.load_suite(ROOT / "benchmarks/vision-v19/suite.json")
        self.assertEqual(tuple(document["budgets"]), MODULE.EXPECTED_BUDGETS)
        self.assertEqual(
            {asset["geometry"] for asset in assets}, {"square", "wide", "tall"}
        )
        categories = {
            category for asset in assets for category in asset["categories"]
        }
        self.assertEqual(
            categories,
            {
                "image_description",
                "ocr",
                "chart",
                "document_page",
                "counting",
                "spatial_relations",
                "colors",
                "small_details",
            },
        )

    def test_checks_are_casefolded_and_support_number_words(self) -> None:
        exact = {"terms": ["ns-731"]}
        count = {"terms": ["5"], "alternatives": ["five"]}
        grouped = {"terms": ["cedar", "7"], "alternatives": [["cedar", "seven"]]}
        self.assertTrue(MODULE.check_output("Code: NS–731", exact))
        self.assertTrue(MODULE.check_output("There are FIVE stars.", count))
        self.assertFalse(MODULE.check_output("There are four stars.", count))
        self.assertTrue(MODULE.check_output("Cedar is seven.", grouped))
        self.assertFalse(MODULE.check_output("Birch is seven.", grouped))


if __name__ == "__main__":
    unittest.main()
