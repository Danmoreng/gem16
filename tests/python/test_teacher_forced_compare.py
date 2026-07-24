from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
MODULE_PATH = ROOT / "tools" / "teacher_forced_compare.py"
SPEC = importlib.util.spec_from_file_location("teacher_forced_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
teacher_forced_compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teacher_forced_compare)


class TeacherForcedCompareTest(unittest.TestCase):
    def test_aggregate_reports_position_metrics(self) -> None:
        prompts = [
            {
                "comparison": {
                    "steps": [
                        {
                            "top1_agreement": True,
                            "reference_top1_token_id": 1,
                            "reference_top1_engine_rank": 1,
                            "reference_top20_overlap": 19,
                            "reference_top_logprob_comparisons": [
                                {"token_id": 1, "absolute_delta": 0.1},
                                {"token_id": 2, "absolute_delta": 0.2},
                            ],
                        },
                        {
                            "top1_agreement": False,
                            "reference_top1_token_id": 2,
                            "reference_top1_engine_rank": 3,
                            "reference_top20_overlap": 17,
                            "reference_top_logprob_comparisons": [
                                {"token_id": 2, "absolute_delta": 0.3},
                            ],
                        },
                    ]
                }
            }
        ]
        result = teacher_forced_compare.aggregate_comparisons(prompts)
        self.assertEqual(result["positions_compared"], 2)
        self.assertEqual(result["top1_agreements"], 1)
        self.assertEqual(result["top1_agreement_rate"], 0.5)
        self.assertEqual(result["reference_top1_in_engine_top5"], 2)
        self.assertEqual(result["prompts_with_all_top1_agree"], 0)
        self.assertAlmostEqual(
            result["mean_reference_top20_logprob_absolute_delta"], 0.2
        )
        self.assertAlmostEqual(
            result["mean_selected_logprob_absolute_delta"], 0.2
        )

    def test_reference_cache_mismatch_is_rejected(self) -> None:
        with self.assertRaises(teacher_forced_compare.TeacherForcedError):
            teacher_forced_compare.validate_reference_cache(
                {"execution": {"kv_cache_dtype": "bfloat16"}}, "fp8"
            )


if __name__ == "__main__":
    unittest.main()
