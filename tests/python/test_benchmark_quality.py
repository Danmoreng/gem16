import importlib.util
from pathlib import Path
import sys
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


if __name__ == "__main__":
    unittest.main()
