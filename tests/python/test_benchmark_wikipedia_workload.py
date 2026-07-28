import importlib.util
from pathlib import Path
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "benchmark_wikipedia_workload.py"
SPEC = importlib.util.spec_from_file_location("benchmark_wikipedia_workload", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BenchmarkWikipediaWorkloadTest(unittest.TestCase):
    def test_llama_mtp_counters_are_normalized(self) -> None:
        response = {
            "tokens": list(range(16)),
            "tokens_evaluated": 3,
            "truncated": False,
            "stop_type": "limit",
            "tokens_cached": 0,
            "timings": {
                "prompt_ms": 2.0,
                "predicted_ms": 4.0,
                "predicted_n": 16,
                "predicted_per_second": 4000.0,
                "draft_n": 12,
                "draft_n_accepted": 9,
            },
        }
        generation = {
            "max_new_tokens": 16,
            "seed": 0,
            "ignore_eos": True,
            "suppress_token_ids": [],
        }
        with mock.patch.object(MODULE, "http_json", return_value=response), mock.patch.object(
            MODULE.time, "perf_counter", side_effect=(1.0, 1.01)
        ):
            run, tokens = MODULE.run_llama_request(
                "http://127.0.0.1:1", [1, 2, 3], generation, 2
            )
        self.assertEqual(tokens, list(range(16)))
        self.assertEqual(run["mtp"]["verification_groups"], 7)
        self.assertEqual(run["mtp"]["accepted_tokens"], 9)
        self.assertEqual(run["mtp"]["rejected_tokens"], 3)
        self.assertAlmostEqual(run["mtp"]["mean_accepted_length"], 9 / 7)

    def test_llama_mtp_counters_require_active_mode(self) -> None:
        response = {
            "tokens": [1, 2],
            "tokens_evaluated": 1,
            "truncated": False,
            "stop_type": "limit",
            "timings": {
                "prompt_ms": 1.0,
                "predicted_ms": 1.0,
                "predicted_n": 2,
                "predicted_per_second": 2000.0,
                "draft_n": 1,
                "draft_n_accepted": 1,
            },
        }
        generation = {
            "max_new_tokens": 2,
            "seed": 0,
            "suppress_token_ids": [],
        }
        with mock.patch.object(MODULE, "http_json", return_value=response), mock.patch.object(
            MODULE.time, "perf_counter", side_effect=(1.0, 1.01)
        ):
            with self.assertRaises(MODULE.BenchmarkError):
                MODULE.run_llama_request(
                    "http://127.0.0.1:1", [1], generation, 0
                )


if __name__ == "__main__":
    unittest.main()
