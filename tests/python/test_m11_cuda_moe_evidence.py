import unittest

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


if __name__ == "__main__":
    unittest.main()
