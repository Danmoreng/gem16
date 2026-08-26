import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "benchmark_quality.py"
SPEC = importlib.util.spec_from_file_location("benchmark_quality", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark_quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_quality
SPEC.loader.exec_module(benchmark_quality)


class BenchmarkQualityTest(unittest.TestCase):
    def test_gem16_payload_uses_supported_controls_only(self):
        payload = benchmark_quality.build_request_payload(
            backend="gem16",
            model="gem16",
            messages=[{"role": "user", "content": "test"}],
            reasoning="high",
            generation="greedy",
            max_tokens=8192,
            seed=42,
        )
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertNotIn("seed", payload)
        self.assertNotIn("chat_template_kwargs", payload)

    def test_generic_payload_materializes_checkpoint_profile(self):
        payload = benchmark_quality.build_request_payload(
            backend="openai",
            model="reference",
            messages=[],
            reasoning="none",
            generation="checkpoint",
            max_tokens=512,
            seed=7,
        )
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(payload["top_p"], 0.95)
        self.assertEqual(payload["top_k"], 64)
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(
            payload["chat_template_kwargs"], {"enable_thinking": False}
        )

    def test_runtime_estimate(self):
        seconds = benchmark_quality.estimate_seconds(10, 2, 350, 35.0, 0.5)
        self.assertEqual(seconds, 210.0)
        with self.assertRaises(benchmark_quality.BenchmarkError):
            benchmark_quality.estimate_seconds(0, 1, 1, 1.0, 0.0)

    def test_numeric_answer(self):
        self.assertEqual(benchmark_quality._numeric_answer(" 1,234 "), 1234)
        self.assertEqual(benchmark_quality._numeric_answer("2.5"), 2.5)

    def test_resumable_prediction_journal_round_trip(self):
        example = SimpleNamespace(
            id="gsm8k-0",
            inputs={"problem": "2 + 2?"},
            target=4,
            meta={},
        )
        sample = SimpleNamespace(
            text="4",
            completion_tokens=1,
            prompt_tokens=7,
            reasoning_tokens=0,
            finish_reason="stop",
            generation_start_time=10.0,
            generation_end_time=10.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output-rs0.jsonl"
            writer = benchmark_quality.ResumablePredictionsWriter(path, [example])
            writer(example, 0, sample, 1.0, "4")
            writer.close()
            rows = benchmark_quality.load_prediction_rows(path, [example])
            self.assertEqual(set(rows), {"gsm8k-0"})
            self.assertEqual(rows["gsm8k-0"]["num_prompt_tokens"], 7)
            self.assertTrue(rows["gsm8k-0"]["symbolic_correct"])

            writer = benchmark_quality.ResumablePredictionsWriter(path, [example])
            with self.assertRaises(benchmark_quality.BenchmarkError):
                writer(example, 0, sample, 1.0, "4")
            writer.close()

    def test_resume_state_locks_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume-state.json"
            identity = {"benchmark": "gsm8k", "planned_example_ids": ["gsm8k-0"]}
            created = benchmark_quality.initialize_resume_state(path, identity)
            loaded = benchmark_quality.initialize_resume_state(path, identity)
            self.assertEqual(created["identity"], loaded["identity"])
            with self.assertRaises(benchmark_quality.BenchmarkError):
                benchmark_quality.initialize_resume_state(
                    path,
                    {"benchmark": "gsm8k", "planned_example_ids": ["gsm8k-1"]},
                )

    def test_prediction_journal_rejects_changed_problem(self):
        example = SimpleNamespace(
            id="gsm8k-0",
            inputs={"problem": "original"},
            target=4,
            meta={},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output-rs0.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "gsm8k-0",
                        "problem": "changed",
                        "expected_answer": "4",
                        "symbolic_correct": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(benchmark_quality.BenchmarkError):
                benchmark_quality.load_prediction_rows(path, [example])


if __name__ == "__main__":
    unittest.main()
