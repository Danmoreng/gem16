from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "teacher_forced_llama.py"
SPEC = importlib.util.spec_from_file_location("teacher_forced_llama", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
teacher_forced_llama = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teacher_forced_llama)


class TeacherForcedLlamaTest(unittest.TestCase):
    def test_position_compares_selected_distributions(self) -> None:
        response = {
            "completion_probabilities": [
                {
                    "id": 7,
                    "top_logprobs": [
                        {"id": 7, "logprob": -0.2},
                        {"id": 8, "logprob": -1.7},
                    ],
                }
            ]
        }
        reference = [
            {"token_id": 7, "logprob": -0.1},
            {"token_id": 9, "logprob": -2.0},
        ]
        result = teacher_forced_llama.compare_position(response, reference, 7, 0)
        self.assertTrue(result["top1_agreement"])
        self.assertEqual(result["reference_top1_llama_cpp_top20_rank"], 1)
        self.assertEqual(result["top20_overlap_count"], 1)
        self.assertAlmostEqual(result["selected_logprob_delta"], -0.1)


if __name__ == "__main__":
    unittest.main()
