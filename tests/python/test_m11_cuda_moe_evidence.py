import unittest
import json
from pathlib import Path

from tools.summarize_gemma4_26b_m11 import THRESHOLDS, metrics


class M11CudaMoeEvidenceTest(unittest.TestCase):
    def test_metrics_identity(self) -> None:
        result = metrics([1.0, -2.0], [1.0, -2.0])
        self.assertEqual(result["max_abs"], 0.0)
        self.assertEqual(result["relative_l2"], 0.0)
        self.assertAlmostEqual(result["cosine"], 1.0)

    def test_thresholds_are_fixed_and_specific(self) -> None:
        self.assertEqual(THRESHOLDS["router_probability_max_abs"], 0.003)
        self.assertEqual(THRESHOLDS["expert_relative_l2_max"], 0.30)
        self.assertGreater(THRESHOLDS["expert_cosine_min"], 0.95)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metrics([1.0], [1.0, 2.0])

    def test_clean_acceptance_binds_implementation(self) -> None:
        root = Path(__file__).resolve().parents[2]
        acceptance = json.loads((root / "artifacts/m11/acceptance.json").read_text())
        implementation = "91ee47586cc426c051dee247ddfcf4a6b765ecfd"
        self.assertTrue(acceptance["acceptance"])
        self.assertEqual(acceptance["status"], "acceptance_pass")
        self.assertEqual(acceptance["implementation_commit"], implementation)
        self.assertEqual(acceptance["code_revision"], implementation)
        self.assertTrue(acceptance["lifecycle"]["forward_allocation_free"])
        self.assertTrue(acceptance["lifecycle"]["repeated_bitwise_identical"])
        self.assertEqual(len(acceptance["expert_metrics"]), 8)
        self.assertTrue(all(item["status"] == "pass"
                            for item in acceptance["sanitizers"]))


if __name__ == "__main__":
    unittest.main()
