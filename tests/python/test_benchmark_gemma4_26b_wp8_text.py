from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "benchmark_gemma4_26b_wp8_text.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_wp8_text", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
wp8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wp8)


class BenchmarkGemma4Moe26BWp8TextTest(unittest.TestCase):
    def test_largest_repeat_count(self) -> None:
        repeats, tokens = wp8.largest_repeat_count(101, lambda value: 5 + value * 3)
        self.assertEqual(repeats, 32)
        self.assertEqual(tokens, 101)

    def test_retrieval_placement_preserves_repeat_count(self) -> None:
        contract = {
            "filler": "f ",
            "needle": "N-1",
            "needle_sentence": "needle {needle} ",
            "query": "value?",
        }
        early = wp8.retrieval_message(contract, 10, 0.1)
        late = wp8.retrieval_message(contract, 10, 0.9)
        self.assertEqual(early.count("f "), 10)
        self.assertEqual(late.count("f "), 10)
        self.assertIn("N-1", early)
        self.assertIn("N-1", late)
        self.assertNotEqual(early, late)

    def test_invalid_target_is_rejected(self) -> None:
        with self.assertRaises(wp8.BenchmarkError):
            wp8.largest_repeat_count(0, lambda _: 0)


if __name__ == "__main__":
    unittest.main()
