from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "compare_gemma4_26b_wp8.py"
)
SPEC = importlib.util.spec_from_file_location("compare_gemma4_26b_wp8", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
wp8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wp8)


def capture(position, layer, output, probabilities, ids):
    return {
        "position": position,
        "layer": layer,
        "output": output,
        "router_probabilities": probabilities,
        "router_top_ids": ids,
    }


class CompareGemma4Moe26BWp8Test(unittest.TestCase):
    def test_vector_metrics_identical(self) -> None:
        metrics = wp8.vector_metrics([1.0, -2.0, 3.0], [1.0, -2.0, 3.0])
        self.assertEqual(metrics["relative_l2"], 0.0)
        self.assertAlmostEqual(metrics["cosine"], 1.0)
        self.assertEqual(metrics["max_abs_error"], 0.0)

    def test_probability_metrics_normalize_inputs(self) -> None:
        metrics = wp8.probability_metrics([2.0, 2.0], [0.5, 0.5])
        self.assertAlmostEqual(metrics["l1"], 0.0)
        self.assertAlmostEqual(metrics["kl_control_to_candidate"], 0.0)

    def test_logit_metrics_report_rank_and_kl(self) -> None:
        metrics = wp8.logit_metrics([0.0, 2.0, 3.0], [0.0, 3.0, 2.0])
        self.assertFalse(metrics["top1_agreement"])
        self.assertEqual(metrics["control_top1_candidate_rank"], 2)
        self.assertGreater(metrics["kl_control_to_candidate"], 0.0)

    def test_capture_contract_rejects_missing_layer(self) -> None:
        report = {
            "captures": [capture(0, 0, [1.0], [1.0], [0])]
        }
        with self.assertRaises(wp8.ComparisonError):
            wp8.require_exact_capture_contract(report, [0, 1], [0], "fixture")

    def test_compare_passes_identical_frozen_fixture(self) -> None:
        layers = list(range(30))
        positions = [0, 17]
        captures = [
            capture(position, layer, [1.0, 2.0], [0.75, 0.25], [0, 1])
            for position in positions
            for layer in layers
        ]
        report = {
            "capture_layers": "all",
            "prompt_token_ids": [1, 2],
            "continuation_token_ids": [3],
            "deterministic": True,
            "full_logits_repeat_equal": True,
            "all_logits_finite": True,
            "first_generated": [4],
            "continuation": {
                "start_position": 3,
                "end_position": 4,
                "first_prediction": 4,
                "second_prediction": 4,
            },
            "memory": {
                "free_after_first_run_bytes": 100,
                "free_after_runs_bytes": 100,
            },
            "captures": captures,
        }
        golden = {
            "generated_token_ids": [1],
            "captures": {},
        }
        for layer in (0, 5, 6, 29):
            golden["captures"][f"layer_{layer}.output"] = {
                "rows": [
                    {"position": position, "values_f32": [1.0, 2.0]}
                    for position in positions
                ]
            }
            golden["captures"][f"layer_{layer}.router_top_ids"] = {
                "rows": [
                    {"position": position, "values_i64": [0, 1]}
                    for position in positions
                ]
            }
        suite = {
            "numerical_differential": {
                "capture_layers": layers,
                "capture_positions": positions,
                "thresholds": {
                    "layer_relative_l2_max": 0.25,
                    "layer_cosine_min": 0.98,
                    "router_top8_set_overlap_min": 2,
                    "full_logit_kl_control_to_candidate_max": 0.02,
                    "control_top1_candidate_rank_max": 5,
                },
            }
        }
        result = wp8.compare(
            suite, report, [0.0, 2.0], report, [0.0, 2.0], golden, [0.0, 2.0]
        )
        self.assertEqual(result["decision"], "proceed")
        self.assertEqual(result["quality_decision"], "proceed")
        self.assertEqual(result["summary"]["cases"], 60)
        self.assertTrue(result["gates"]["full_logit_envelope"])


if __name__ == "__main__":
    unittest.main()
