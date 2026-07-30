import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "compare_quality_benchmarks.py"
)
SPEC = importlib.util.spec_from_file_location("compare_quality_benchmarks", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compare_quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_quality)


def result(scores, backend):
    return {
        "schema_version": 1,
        "status": "complete",
        "benchmark": "gsm8k",
        "endpoint": {"backend": backend, "model": backend},
        "protocol": {
            "reasoning": "none",
            "generation": "greedy",
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": None,
            "seed": 42,
            "max_tokens": 512,
            "repeats": 1,
        },
        "examples": [
            {"id": f"q{index}", "samples": [{"score": score}]}
            for index, score in enumerate(scores)
        ],
    }


class CompareQualityBenchmarksTest(unittest.TestCase):
    def test_paired_comparison(self):
        compared = compare_quality.compare(
            result([1, 1, 0, 0], "vllm"), result([1, 0, 1, 0], "gem16")
        )
        self.assertEqual(compared["reference"]["accuracy"], 0.5)
        self.assertEqual(compared["candidate"]["accuracy"], 0.5)
        self.assertEqual(compared["paired"]["both_correct"], 1)
        self.assertEqual(compared["paired"]["reference_only_correct"], 1)
        self.assertEqual(compared["paired"]["candidate_only_correct"], 1)
        self.assertEqual(compared["paired"]["both_wrong"], 1)
        self.assertEqual(compared["paired"]["mcnemar_exact_two_sided_p"], 1.0)

    def test_protocol_mismatch_is_rejected(self):
        candidate = result([1], "gem16")
        candidate["protocol"]["reasoning"] = "high"
        with self.assertRaises(compare_quality.ComparisonError):
            compare_quality.compare(result([1], "vllm"), candidate)


if __name__ == "__main__":
    unittest.main()
