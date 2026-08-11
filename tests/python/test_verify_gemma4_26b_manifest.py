from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from tools.verify_gemma4_26b_manifest import load_json, summarize


def tensor(name: str, role: str, byte_length: int) -> dict[str, object]:
    return {
        "name": name,
        "tensor_role": role,
        "residency_class": "compiler_source_text",
        "byte_length": byte_length,
        "logical_dtype": "BF16",
        "source_family": "google_gemma4_26b_bf16_source",
        "quantization_component": "weight",
        "quantization_producer": "none",
        "local_scale_dtype": "none",
        "global_scale_role": "none",
        "activation_scale_role": "none",
        "final_gpu_layout": "compiler_source_order_only",
        "layer_index": -1,
        "expert_index": -1,
        "expert_axis": -1,
        "logical_axis_order": "",
        "aliased": False,
    }


class VerifyGemma426BManifestTest(unittest.TestCase):
    def source_fixture(self) -> tuple[dict[str, object], dict[str, object]]:
        embedding = tensor(
            "model.language_model.embed_tokens.weight",
            "tied_embedding_and_output",
            10,
        )
        embedding["aliased"] = True
        gate_up = tensor(
            "model.language_model.layers.0.experts.gate_up_proj",
            "routed_expert_gate_up",
            20,
        )
        gate_up["expert_axis"] = 0
        gate_up["logical_axis_order"] = "expert,gate_then_up,input"
        manifest = {
            "schema_version": 3,
            "model_variant": "gemma4_moe_26b_a4b",
            "checkpoint_profile": "source_bf16",
            "validation_contract": "gemma4_26b_m03_exact_inventory_v1",
            "runtime_supported": False,
            "tensor_contract_validated": True,
            "total_tensor_bytes": 30,
            "text_only_tensor_bytes": 30,
            "skipped_tensor_bytes": 0,
            "tensors": [embedding, gate_up],
            "totals_by_role": [
                {
                    "role": "routed_expert_gate_up",
                    "tensor_count": 1,
                    "bytes": 20,
                },
                {
                    "role": "tied_embedding_and_output",
                    "tensor_count": 1,
                    "bytes": 10,
                },
            ],
            "totals_by_residency": [
                {
                    "residency_class": "compiler_source_text",
                    "tensor_count": 2,
                    "bytes": 30,
                }
            ],
        }
        fixture = {
            "source_bf16": {
                "payload_bytes": 30,
                "text_bytes": 30,
                "tensor_count": 2,
                "compile_excluded_vision_tensor_count": 0,
                "compile_excluded_vision_bytes": 0,
            }
        }
        return manifest, fixture

    def test_minimal_source_contract_passes(self) -> None:
        manifest, fixture = self.source_fixture()
        report = summarize(manifest, fixture, "source_bf16")
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["unknown_semantic_tensor_count"], 0)

    def test_duplicate_json_keys_and_non_object_tensors_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(
                '{"schema_version":3,"schema_version":2}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_json(path, 1024)

        manifest, fixture = self.source_fixture()
        manifest["tensors"].append("not-an-object")
        report = summarize(manifest, fixture, "source_bf16")
        self.assertEqual(report["status"], "fail")
        self.assertIn("tensor entry must be an object", report["errors"])

    def test_wrong_gate_up_order_and_totals_fail(self) -> None:
        manifest, fixture = self.source_fixture()
        bad = copy.deepcopy(manifest)
        bad["tensors"][1]["logical_axis_order"] = "expert,up_then_gate,input"
        bad["totals_by_role"][0]["bytes"] = 19
        report = summarize(bad, fixture, "source_bf16")
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("Gate/Up" in error for error in report["errors"]))
        self.assertTrue(any("totals_by_role" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
