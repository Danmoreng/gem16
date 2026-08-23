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
    def test_llama_tensor_overrides_are_repeatable(self) -> None:
        argv = [
            "benchmark_wikipedia_workload.py",
            "--engine",
            "llama-cpp",
            "--workload",
            "workload.json",
            "--output",
            "output.json",
            "--llama-override-tensor",
            "token_embd.weight=CUDA0",
            "--llama-override-tensor",
            "output.weight=CUDA0",
            "--llama-log-verbosity",
            "4",
            "--llama-batch-size",
            "1024",
            "--llama-ubatch-size",
            "512",
            "--llama-draft-override-tensor",
            "token_embd.weight=CUDA0",
        ]
        with mock.patch.object(MODULE.sys, "argv", argv):
            args = MODULE.parse_args()
        self.assertEqual(
            args.llama_override_tensor,
            ["token_embd.weight=CUDA0", "output.weight=CUDA0"],
        )
        self.assertEqual(args.llama_log_verbosity, 4)
        self.assertEqual(args.llama_batch_size, 1024)
        self.assertEqual(args.llama_ubatch_size, 512)
        self.assertEqual(
            args.llama_draft_override_tensor,
            ["token_embd.weight=CUDA0"],
        )

    def test_llama_ubatch_must_not_exceed_batch(self) -> None:
        argv = [
            "benchmark_wikipedia_workload.py",
            "--engine",
            "llama-cpp",
            "--workload",
            "workload.json",
            "--output",
            "output.json",
            "--llama-batch-size",
            "256",
            "--llama-ubatch-size",
            "512",
        ]
        with mock.patch.object(MODULE.sys, "argv", argv):
            with self.assertRaises(SystemExit):
                MODULE.parse_args()

    def test_zero_warmups_are_accepted(self) -> None:
        self.assertEqual(MODULE.nonnegative_int("0"), 0)
        with self.assertRaises(MODULE.argparse.ArgumentTypeError):
            MODULE.nonnegative_int("-1")

    def test_vllm_model_weights_are_separate_from_config(self) -> None:
        argv = [
            "benchmark_wikipedia_workload.py",
            "--engine",
            "vllm",
            "--workload",
            "workload.json",
            "--output",
            "output.json",
            "--model",
            "hf-config",
            "--vllm-model-weights",
            "weights.gguf",
        ]
        with mock.patch.object(MODULE.sys, "argv", argv):
            args = MODULE.parse_args()
        self.assertEqual(args.model, Path("hf-config"))
        self.assertEqual(args.vllm_model_weights, Path("weights.gguf"))

    def test_single_repetition_summary_is_a_characterization(self) -> None:
        summary = MODULE.summarize([46.422])
        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["mean"], 46.422)
        self.assertEqual(summary["median"], 46.422)
        self.assertEqual(summary["standard_deviation"], 0.0)
        self.assertEqual(summary["confidence_interval_95"], [46.422, 46.422])

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

    def test_llama_ngram_counters_are_normalized(self) -> None:
        response = {
            "tokens": list(range(12)),
            "tokens_evaluated": 2,
            "truncated": False,
            "stop_type": "limit",
            "tokens_cached": 0,
            "timings": {
                "prompt_ms": 2.0,
                "predicted_ms": 4.0,
                "predicted_n": 12,
                "predicted_per_second": 3000.0,
                "draft_n": 20,
                "draft_n_accepted": 8,
            },
        }
        generation = {
            "max_new_tokens": 12,
            "seed": 0,
            "ignore_eos": True,
            "suppress_token_ids": [],
        }
        with mock.patch.object(MODULE, "http_json", return_value=response), mock.patch.object(
            MODULE.time, "perf_counter", side_effect=(1.0, 1.01)
        ):
            run, _ = MODULE.run_llama_request(
                "http://127.0.0.1:1", [1, 2], generation, 8, ("ngram-mod",)
            )
        self.assertNotIn("mtp", run)
        self.assertEqual(run["speculative"]["verification_groups"], 4)
        self.assertEqual(run["speculative"]["accepted_tokens"], 8)
        self.assertAlmostEqual(run["speculative"]["mean_accepted_length"], 2.0)

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
