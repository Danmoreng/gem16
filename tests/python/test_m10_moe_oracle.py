from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.gemma4_26b_moe_oracle import (
    OracleError, bf16_round, decode_e2m1, decode_e4m3fn,
    dequantize_nvfp4_row, fused_expert_mlp_bf16, mlp_bf16, moe_layer_bf16, router_bf16,
    SafeTensorReader, select_topk, softmax_fp32,
)


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "artifacts/m10/diagnostic-summary.json"
ACCEPTANCE = ROOT / "artifacts/m10/acceptance.json"
IMPLEMENTATION_COMMIT = "eac6b443b239d5e04c5be5daef3dd659d57d5de9"


class M10MoeOracleTest(unittest.TestCase):
    def test_softmax_topk_and_lower_id_tie_policy(self) -> None:
        probabilities = softmax_fp32([2.0, 2.0, 1.0, 0.0])
        selected = select_topk(probabilities, [1.0, 2.0, 3.0, 4.0], 2)
        self.assertEqual(selected["ids"], [0, 1])
        self.assertAlmostEqual(sum(selected["normalized_probabilities"]), 1.0)
        self.assertAlmostEqual(selected["weights"][1], 1.0)

    def test_router_rejects_nonfinite_and_invalid_scale(self) -> None:
        with self.assertRaises(OracleError):
            softmax_fp32([0.0, math.nan])
        with self.assertRaises(OracleError):
            select_topk([0.5, 0.5], [1.0, 0.0], 1)

    def test_all_equal_and_dominant_router_cases(self) -> None:
        equal = select_topk([0.25] * 4, [1.0] * 4, 2)
        self.assertEqual(equal["ids"], [0, 1])
        self.assertEqual(equal["normalized_probabilities"], [0.5, 0.5])
        dominant = select_topk(softmax_fp32([20.0, 0.0, -1.0, -2.0]), [1.0] * 4, 2)
        self.assertEqual(dominant["ids"][0], 0)
        self.assertGreater(dominant["normalized_probabilities"][0], 0.999999)

    def test_bf16_mlp_and_full_branch_captures_are_transparent(self) -> None:
        identity = [[1.0, 0.0], [0.0, 1.0]]
        swap = [[0.0, 1.0], [1.0, 0.0]]
        shared = {"gate": identity, "up": identity, "down": identity}
        experts = {
            0: {"gate": identity, "up": identity, "down": identity},
            1: {"gate": swap, "up": swap, "down": identity},
        }
        norms = {name: [1.0, 1.0] for name in
                 ("pre_shared", "post_shared", "pre_expert", "post_expert", "post_combined")}
        router = {"scale": [1.0, 1.0], "projection": identity,
                  "per_expert_scale": [1.0, 1.0], "top_k": 2}
        trace = moe_layer_bf16([1.0, 0.5], norms, shared, experts, router)
        self.assertEqual(len(trace["expert_contributions"]), 2)
        self.assertEqual([item["rank"] for item in trace["expert_contributions"]], [0, 1])
        self.assertEqual(len(trace["output"]), 2)
        self.assertTrue(all(math.isfinite(value) for value in trace["output"]))
        self.assertEqual(mlp_bf16([1.0, 0.0], identity, identity, identity)["up"], [1.0, 0.0])

    def test_fused_axis_deterministic_reduction_and_layer_scalar(self) -> None:
        identity = [[1.0, 0.0], [0.0, 1.0]]
        zeros = [[0.0, 0.0], [0.0, 0.0]]
        fused = fused_expert_mlp_bf16([1.0, 0.5], identity + identity, identity)
        separate = mlp_bf16([1.0, 0.5], identity, identity, identity)
        self.assertEqual(fused, separate)
        with self.assertRaises(OracleError):
            fused_expert_mlp_bf16([1.0, 0.5], identity + [identity[0]], identity)

        norms = {name: [1.0, 1.0] for name in
                 ("pre_shared", "post_shared", "pre_expert", "post_expert", "post_combined")}
        router = {"scale": [1.0, 1.0], "projection": identity,
                  "per_expert_scale": [1.0, 1.0], "top_k": 2}
        experts = {expert: {"gate": identity, "up": identity, "down": identity}
                   for expert in (0, 1)}
        shared = {"gate": zeros, "up": zeros, "down": zeros}
        first = moe_layer_bf16([1.0, 0.5], norms, shared, experts, router)
        second = moe_layer_bf16([1.0, 0.5], norms, shared, experts, router)
        half = moe_layer_bf16([1.0, 0.5], norms, shared, experts, router, layer_scalar=0.5)
        self.assertEqual(first, second)
        self.assertTrue(all(value == 0.0 for value in first["shared"]["output"]))
        self.assertTrue(any(value != 0.0 for value in first["routed_sum_fp32"]))
        self.assertEqual(half["output"], [bf16_round(value * 0.5) for value in first["output"]])

    def test_independent_nvfp4_dequantizer(self) -> None:
        self.assertEqual([decode_e2m1(code) for code in range(8)],
                         [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        self.assertEqual(decode_e4m3fn(0x38), 1.0)
        packed = bytes([0x21, 0x43, 0x65, 0x07] * 2)
        row = dequantize_nvfp4_row(packed, bytes([0x38]), 2.0, 16)
        self.assertEqual(row[:8], [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 0.0])
        with self.assertRaises(OracleError):
            dequantize_nvfp4_row(packed, b"", 2.0, 16)

    def test_safetensors_reader_rejects_duplicate_index_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "model.safetensors.index.json"
            index.write_text('{"weight_map":{},"weight_map":{}}', encoding="utf-8")
            with self.assertRaises(OracleError):
                SafeTensorReader(root)
            index.write_text('{"weight_map":{"tensor":"../escape.safetensors"}}', encoding="utf-8")
            with self.assertRaises(OracleError):
                SafeTensorReader(root).record("tensor")
            index.write_text('{"weight_map":{"a":"model.safetensors","b":"model.safetensors"}}',
                             encoding="utf-8")
            header = json.dumps({"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
                                 "b": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}},
                                separators=(",", ":")).encode("utf-8")
            (root / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0")
            with self.assertRaises(OracleError):
                SafeTensorReader(root).record("a")

    def test_compact_real_diagnostic(self) -> None:
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "diagnostic_pass_acceptance_pending_clean_commit")
        self.assertFalse(summary["acceptance"])
        self.assertFalse(summary["oracle"]["uses_production_cuda"])
        self.assertEqual(summary["trusted_bf16"]["router_ordered_exact"], 7)
        self.assertEqual(summary["trusted_bf16"]["router_tie_equivalent"], 1)
        self.assertEqual(summary["trusted_bf16"]["expert_contribution_count"], 64)
        self.assertEqual(summary["real_bf16_replay"]["status"], "pass")
        self.assertEqual(len(summary["real_bf16_replay"]["shared_boundaries"]), 8)
        self.assertEqual(len(summary["real_bf16_replay"]["norm_boundaries"]), 24)
        self.assertEqual(len(summary["real_bf16_replay"]["router_boundaries"]), 8)
        self.assertEqual(len(summary["real_bf16_replay"]["expert_boundaries"]), 64)
        self.assertEqual(summary["real_bf16_replay"]["thresholds"], {
            "expert_cosine_min": 0.9999,
            "expert_relative_l2_max": 0.01,
            "norm_cosine_min": 0.99999,
            "norm_relative_l2_max": 0.005,
            "router_probability_max_abs": 0.003,
            "router_weight_max_abs": 0.005,
            "shared_cosine_min": 0.99999,
            "shared_relative_l2_max": 0.001,
        })
        self.assertEqual(summary["trusted_bf16"]["routed_sum_thresholds"], {
            "cosine_min": 0.99999, "max_abs": 0.03125, "relative_l2": 0.005,
        })
        self.assertEqual(summary["real_nvfp4_adapter"]["sample_count"], 2)
        self.assertTrue(all(item["finite"] for item in summary["real_nvfp4_adapter"]["samples"]))

    def test_clean_acceptance_binds_implementation_commit(self) -> None:
        acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
        self.assertEqual(acceptance["status"], "acceptance_pass")
        self.assertTrue(acceptance["acceptance"])
        self.assertEqual(acceptance["implementation_commit"], IMPLEMENTATION_COMMIT)
        self.assertEqual(acceptance["code_revision"], IMPLEMENTATION_COMMIT)
        self.assertEqual(acceptance["owner_decision"], {
            "date": "2026-08-14",
            "decision": "M10 accepted",
            "expert_reduction": "top_k_slot_order_fp32",
            "tie_policy": "lower_expert_id",
        })
        self.assertEqual(len(acceptance["real_bf16_replay"]["expert_boundaries"]), 64)
        self.assertNotIn("dirty", " ".join(acceptance["limitations"]))


if __name__ == "__main__":
    unittest.main()
