from __future__ import annotations

import unittest

from tools.audit_gemma4_26b_sources import (
    compare_contracts,
    differing_paths,
    expected_q4_metadata,
    tensor_contract,
    unsloth_layout_summary,
)


class AuditGemma426BSourcesTest(unittest.TestCase):
    def test_differing_paths_are_deterministic_and_nested(self) -> None:
        first = {"same": 1, "nested": {"value": 2}, "only_first": True}
        second = {"same": 1, "nested": {"value": 3}, "only_second": True}
        self.assertEqual(
            differing_paths(first, second),
            ["nested.value", "only_first", "only_second"],
        )

    def test_tensor_contract_rejects_duplicate_names(self) -> None:
        inventory = {
            "tensors": [
                {"name": "weight", "dtype": "BF16", "shape": [2], "bytes": 4},
                {"name": "weight", "dtype": "BF16", "shape": [2], "bytes": 4},
            ]
        }
        with self.assertRaisesRegex(ValueError, "duplicate tensor name"):
            tensor_contract(inventory)

    def test_contract_comparison_reports_structural_differences(self) -> None:
        first = {"a": ("BF16", (2,), 4), "b": ("BF16", (4,), 8)}
        second = {"a": ("BF16", (3,), 6), "c": ("BF16", (4,), 8)}
        report = compare_contracts(first, second)
        self.assertFalse(report["exact"])
        self.assertEqual(report["common_tensor_count"], 1)
        self.assertEqual(report["first_only_count"], 1)
        self.assertEqual(report["second_only_count"], 1)
        self.assertEqual(report["structural_mismatch_examples"], ["a"])

    def test_q4_metadata_derives_global_and_local_kv_heads(self) -> None:
        config = {
            "num_hidden_layers": 2,
            "max_position_embeddings": 4096,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 128,
            "final_logit_softcapping": 30.0,
        }
        metadata = expected_q4_metadata(config)
        self.assertEqual(metadata["gemma4.block_count"], 2)
        self.assertEqual(metadata["gemma4.attention.head_count_kv"], [2, 1])

    def test_unsloth_layout_summarizes_role_and_dtype(self) -> None:
        inventory = {
            "tensors": [
                {
                    "name": "layer.expert.weight_packed",
                    "dtype": "U8",
                    "bytes": 4,
                },
                {
                    "name": "layer.expert.weight_scale",
                    "dtype": "F8_E4M3",
                    "bytes": 1,
                },
            ]
        }
        self.assertEqual(
            unsloth_layout_summary(inventory)["tensor_count_by_role_and_dtype"],
            {"weight_packed:U8": 1, "weight_scale:F8_E4M3": 1},
        )


if __name__ == "__main__":
    unittest.main()
