from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.generate_gemma4_26b_manifests import (
    EXPERTS,
    FIXTURE,
    OUTPUT_ROOT,
    ROOT,
    build_outputs,
    classify_external,
    classify_source,
    external_schema,
    require_exact_schema,
    source_schema,
)


class GenerateGemma426BManifestsTest(unittest.TestCase):
    def test_generated_outputs_are_current_and_compact(self) -> None:
        outputs = build_outputs()
        self.assertEqual(len(outputs), 8)
        for path, expected in outputs.items():
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.read_bytes(), expected, path)
            self.assertLess(len(expected), 64 * 1024, path)

    def test_fixture_freezes_source_external_and_compiled_totals(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(fixture["contract"], "gemma4_26b_m03_exact_inventory_v1")
        self.assertEqual(fixture["source_bf16"]["tensor_count"], 1013)
        self.assertEqual(fixture["source_bf16"]["payload_bytes"], 51_611_872_412)
        self.assertEqual(fixture["source_bf16"]["text_tensor_count"], 657)
        self.assertEqual(
            fixture["source_bf16"]["compile_excluded_vision_bytes"],
            1_145_588_832,
        )
        external = fixture["external_unsloth_nvfp4"]
        self.assertEqual(external["tensor_count"], 47_478)
        self.assertEqual(external["serialized_experts_per_layer"], EXPERTS)
        self.assertFalse(external["is_project_compiled_artifact"])
        profiles = fixture["compiled_hybrid"]["profiles"]
        self.assertEqual(profiles["q4_0_head"]["tensor_count"], 1282)
        self.assertEqual(profiles["nvfp4_head"]["tensor_count"], 1285)
        self.assertEqual(
            profiles["q4_0_head"]["aligned_weight_arena_bytes"],
            14_696_667_648,
        )
        self.assertEqual(
            profiles["nvfp4_head"]["aligned_weight_arena_bytes"],
            14_696_668_160,
        )
        self.assertEqual(profiles["q4_0_head"]["alignment_bytes"], 256)
        self.assertEqual(
            profiles["q4_0_head"]["alignment_padding_bytes"], 98_460
        )
        self.assertEqual(
            profiles["nvfp4_head"]["alignment_padding_bytes"], 98_964
        )
        self.assertEqual(
            fixture["compiled_hybrid"]["fp8_kv_32k_bytes"], 440_401_920
        )

    def test_exact_schemas_reject_missing_duplicate_and_wrong_shapes(self) -> None:
        source = source_schema()
        external = external_schema()
        self.assertEqual(len(source), 1013)
        self.assertEqual(len(external), 47_478)

        missing = dict(source)
        del missing["model.language_model.layers.0.experts.down_proj"]
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            require_exact_schema("missing", missing, source)

        wrong_shape = dict(source)
        name = "model.language_model.layers.0.experts.gate_up_proj"
        dtype, _, byte_count = wrong_shape[name]
        wrong_shape[name] = (dtype, (1408, 128, 2816), byte_count)
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            require_exact_schema("wrong", wrong_shape, source)

        duplicate_name = name + ".duplicate"
        duplicate = dict(source)
        duplicate[duplicate_name] = duplicate[name]
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            require_exact_schema("duplicate", duplicate, source)

    def test_classifiers_cover_router_experts_and_vision_without_fallback(self) -> None:
        role, layer, expert = classify_source(
            "model.language_model.layers.29.router.per_expert_scale"
        )
        self.assertEqual((role, layer, expert), ("router_per_expert_scale", 29, None))
        role, layer, expert = classify_external(
            "model.language_model.layers.29.experts.127.down_proj.weight_scale"
        )
        self.assertEqual((role, layer, expert), ("routed_expert_down", 29, 127))
        role, layer, expert = classify_source(
            "model.vision_tower.patch_embedder.position_embedding_table"
        )
        self.assertEqual((role, layer, expert), ("vision_embedding", None, None))
        with self.assertRaisesRegex(ValueError, "unknown source tensor"):
            classify_source("model.mtp.layers.0.weight")

    def test_layer_table_freezes_local_global_v_ownership(self) -> None:
        table = json.loads(
            (OUTPUT_ROOT / "layer-table.json").read_text(encoding="utf-8")
        )["layers"]
        self.assertEqual(len(table), 30)
        for row in table:
            expected_global = row["layer"] % 6 == 5
            self.assertEqual(row["attention_type"] == "full_attention", expected_global)
            self.assertEqual(row["owns_v_projection"], not expected_global)
            self.assertEqual(
                row["external_unsloth_nvfp4"]["serialized_expert_count"], 128
            )
            self.assertTrue(
                row["external_unsloth_nvfp4"]["expert_indices_complete"]
            )

    def test_synthetic_admission_matches_the_frozen_memory_contract(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        report_path = (
            ROOT
            / "docs/evidence/gemma4_26b/m03-synthetic-32k-admission.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["not_model_execution"])
        contract = report["contract"]
        frozen = fixture["compiled_hybrid"]
        self.assertEqual(
            contract["selected_conservative_weight_arena_bytes"],
            frozen["selected_conservative_weight_arena_bytes"],
        )
        self.assertEqual(contract["fp8_kv_32k_bytes"], frozen["fp8_kv_32k_bytes"])
        measurement = report["measurement"]
        self.assertGreaterEqual(
            measurement["final_direct_free_bytes"],
            measurement["required_free_margin_bytes"],
        )
        self.assertEqual(
            measurement["free_after_release_bytes"],
            measurement["free_after_context_bytes"],
        )
        self.assertTrue(all(report["gates"].values()))

    def test_all_fixture_input_hashes_name_repository_files(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(
            set(fixture["inputs"]),
            {"qat_bf16", "ordinary_bf16", "unsloth_nvfp4", "official_q4_0"},
        )
        for source in fixture["inputs"].values():
            self.assertRegex(source["lock_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(source["raw_inventory_sha256"], r"^[0-9a-f]{64}$")
        transformers = fixture["semantic_references"]["transformers"]
        self.assertEqual(
            transformers["revision"],
            "a08ace4bbd97e721c98751deec37d87b026acadc",
        )
        self.assertEqual(transformers["class"], "Gemma4TextExperts")
        self.assertRegex(transformers["reference_lock_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue((ROOT / transformers["reference_lock"]).is_file())
        self.assertEqual(ROOT, Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    unittest.main()
