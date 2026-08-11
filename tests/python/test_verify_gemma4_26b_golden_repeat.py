from __future__ import annotations

import unittest

from tools.verify_gemma4_26b_golden_repeat import compare


class VerifyGemma426BGoldenRepeatTest(unittest.TestCase):
    def fixture(self) -> dict[str, object]:
        return {
            "checkpoint": {"revision": "1" * 40},
            "software": {"transformers": "5.14.1"},
            "execution": {
                "gpu_memory_limit": "12GiB",
                "max_rss_kib": 100,
                "performance_eligible": False,
            },
            "prompt": {"input_token_ids": [1, 2]},
            "captures": {"layer_0.output": {"hash": "abc"}},
            "final_logits": {"sha256": "def"},
            "generated_token_ids": [3],
        }

    def test_repeat_ignores_only_nondeterministic_rss_peak(self) -> None:
        first = self.fixture()
        second = self.fixture()
        second["execution"] = dict(second["execution"], max_rss_kib=200)
        report = compare(first, second)
        self.assertEqual(report["status"], "pass")
        self.assertIn("execution", report["compared_fields"])
        self.assertEqual(report["execution_metadata_excluded"], ["max_rss_kib"])

    def test_repeat_rejects_other_execution_drift(self) -> None:
        first = self.fixture()
        second = self.fixture()
        second["execution"] = dict(second["execution"], gpu_memory_limit="11GiB")
        report = compare(first, second)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["mismatched_fields"], ["execution"])


if __name__ == "__main__":
    unittest.main()
