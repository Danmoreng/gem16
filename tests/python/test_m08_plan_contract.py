"""M08 complete hybrid plan, config, and external-lock tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.gem16_compile.common import BoundedWorkspace, DataError, InvalidPlanError, canonical_json_bytes
from tools.gem16_compile.compiler import (
    _external_lock_document,
    _verify_external_artifact_lock,
    compare_reproducibility,
)
from tools.gem16_compile.plan import load_quantization_plan
from tools.gem16_compile.profiles import M05_SOURCE_LOCK_SHA256, M08_PROFILE
from tools.gem16_compile.reader import LockedFile, TensorDescriptor, VerifiedSource
from tools.gem16_compile.writer import m08_config_bytes
from tools.generate_gemma4_26b_m08_plan import make_plan, make_summary

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "benchmarks/goldens/gemma4_26b/hybrid/qat-compiler-plan.json"
INVENTORY = ROOT / "benchmarks/goldens/gemma4_26b/source-inventories/qat-bf16.json"
DIAGNOSTIC_SUMMARY = ROOT / "artifacts/m08/diagnostic-summary.json"
ACCEPTANCE_SUMMARY = ROOT / "artifacts/m08/acceptance.json"


class M08PlanContractTest(unittest.TestCase):
    def descriptors(self) -> dict[str, TensorDescriptor]:
        inventory = json.loads(INVENTORY.read_text())
        return {
            item["name"]: TensorDescriptor(
                item["name"], item["dtype"], tuple(item["shape"]),
                "source.safetensors", Path("/tmp/source.safetensors"),
                0, 0, item["bytes"], "0" * 64,
            )
            for item in inventory["tensors"]
        }

    def load(self, document: dict | None = None):
        metadata = {
            name: LockedFile(name, Path("/tmp") / name, 1, "0" * 64)
            for name in (
                "chat_template.jinja", "config.json", "generation_config.json",
                "tokenizer.json", "tokenizer_config.json",
            )
        }
        source = VerifiedSource(
            Path("/tmp"), Path("/tmp/source.lock.json"),
            M05_SOURCE_LOCK_SHA256["qat_bf16"], "repo", "0" * 40, "now", metadata,
        )
        with tempfile.TemporaryDirectory(prefix="gem16-m08-plan-") as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(document or json.loads(PLAN.read_text())))
            return load_quantization_plan(
                path, source, self.descriptors(), M08_PROFILE.name,
                M08_PROFILE.head_format,
            )

    def test_generated_complete_partition_and_bytes(self) -> None:
        plan = make_plan()
        summary = make_summary(plan)
        self.assertEqual(len(plan["tensors"]), 1285)
        self.assertEqual(len(plan["excluded_tensors"]), 356)
        self.assertEqual(summary["output_tensor_bytes"], 14_696_569_196)
        self.assertEqual(summary["outputs_by_encoder"]["copy-v1"], 391)
        self.assertEqual(summary["outputs_by_encoder"]["constant-bf16-one-v1"], 60)
        self.assertEqual(summary["outputs_by_encoder"]["nvfp4-packed-v1"], 151)
        self.assertEqual(summary["outputs_by_encoder"]["fp8-rowwise-weight-v1"], 115)
        tied = [item for item in plan["tensors"] if item["role"] == "tied_embedding_and_output"]
        self.assertEqual(len(tied), 4)
        self.assertTrue(all(item["aliased"] for item in tied))
        self.assertFalse(any(item["output_name"].startswith("lm_head") for item in plan["tensors"]))
        self.assertEqual({item["family"] for item in plan["excluded_tensors"]}, {"vision"})
        self.load(plan)

    def test_missing_copy_bad_constant_and_extra_modality_are_rejected(self) -> None:
        plan = json.loads(PLAN.read_text())
        plan["tensors"] = plan["tensors"][1:]
        with self.assertRaises(InvalidPlanError):
            self.load(plan)
        plan = json.loads(PLAN.read_text())
        constant = next(
            item for item in plan["tensors"]
            if item["encoder"] == "constant-bf16-one-v1"
        )
        constant["quantizer_parameters"]["value"] = 0.5
        with self.assertRaises(InvalidPlanError):
            self.load(plan)
        plan = json.loads(PLAN.read_text())
        plan["excluded_tensors"][0]["family"] = "audio"
        with self.assertRaises(InvalidPlanError):
            self.load(plan)

    def test_generated_config_is_explicit_text_only(self) -> None:
        source = json.loads(
            (ROOT / "tests/fixtures/gemma4_26b_config.json").read_text()
        )
        compiled = json.loads(m08_config_bytes(canonical_json_bytes(source)))
        self.assertEqual(compiled["architectures"], source["architectures"])
        self.assertEqual(compiled["gem16"]["profile"], M08_PROFILE.name)
        self.assertTrue(compiled["gem16"]["text_only"])
        for capability in ("vision", "audio", "video", "mtp"):
            self.assertFalse(compiled["gem16"][f"supports_{capability}"])

    def test_external_lock_binds_exact_files_and_detects_corruption(self) -> None:
        compilation = {
            "artifact_profile": M08_PROFILE.name,
            "artifact_status": M08_PROFILE.artifact_status,
            "source": {"lock_sha256": "1" * 64},
            "compiler": {"commit": "2" * 40},
        }
        with tempfile.TemporaryDirectory(prefix="gem16-m08-lock-") as directory:
            root = Path(directory) / "artifact"
            root.mkdir()
            (root / "config.json").write_bytes(b"{}\n")
            (root / "gem16_compilation.json").write_bytes(b"{}\n")
            workspace = BoundedWorkspace(512 * 1024 * 1024, 4096)
            document = _external_lock_document(root, compilation, workspace)
            lock = Path(directory) / "artifact.lock.json"
            lock.write_bytes(canonical_json_bytes(document))
            digest = _verify_external_artifact_lock(
                root, lock, compilation, workspace
            )
            self.assertEqual(len(digest), 64)
            (root / "config.json").write_bytes(b"{ }\n")
            with self.assertRaises(DataError):
                _verify_external_artifact_lock(root, lock, compilation, workspace)

    def test_schemas_and_generated_outputs_are_current(self) -> None:
        self.assertEqual(PLAN.read_bytes(), canonical_json_bytes(make_plan()))
        plan_schema = json.loads(
            (ROOT / "tools/gem16_compile/schemas/compiler-plan.schema.json").read_text()
        )
        compilation_schema = json.loads(
            (ROOT / "tools/gem16_compile/schemas/gem16-compilation.schema.json").read_text()
        )
        self.assertIn(M08_PROFILE.name, plan_schema["properties"]["artifact_profile"]["enum"])
        self.assertIn("compilerM08", compilation_schema["$defs"])
        self.assertEqual(
            compilation_schema["$defs"]["quantizationM08"]["const"]["profile"],
            M08_PROFILE.name,
        )

    def test_reproducibility_report_is_attributed_to_m08(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gem16-m08-repro-") as directory:
            root = Path(directory)
            for name in ("left", "right"):
                artifact = root / name
                artifact.mkdir()
                (artifact / "gem16_compilation.json").write_bytes(
                    canonical_json_bytes({"artifact_profile": M08_PROFILE.name})
                )
            report = compare_reproducibility(root / "left", root / "right")
            self.assertEqual(report["milestone"], "M08")
            self.assertEqual(report["status"], "pass")

    def test_compact_diagnostic_is_explicitly_not_acceptance(self) -> None:
        summary = json.loads(DIAGNOSTIC_SUMMARY.read_text())
        self.assertEqual(summary["status"], "diagnostic_pass_not_acceptance")
        self.assertFalse(summary["acceptance"])
        self.assertTrue(summary["compiler"]["dirty"])
        self.assertTrue(summary["diagnostic_reproducibility"]["byte_identical"])
        self.assertEqual(summary["artifact"]["output_tensor_count"], 1285)
        self.assertEqual(summary["artifact"]["output_tensor_bytes"], 14_696_569_196)
        self.assertEqual(
            summary["verification"]["protected_12b_inspect"]["tensor_count"],
            1389,
        )

    def test_compact_acceptance_binds_clean_reproducible_evidence(self) -> None:
        summary = json.loads(ACCEPTANCE_SUMMARY.read_text())
        self.assertEqual(summary["status"], "acceptance_pass")
        self.assertTrue(summary["acceptance"])
        self.assertFalse(summary["compiler"]["dirty"])
        self.assertEqual(
            summary["compiler"]["commit"],
            "f433358b8e2c1250b95801fc898faee4fcedcbe5",
        )
        self.assertEqual(summary["reproducibility"]["complete_build_count"], 2)
        self.assertTrue(summary["reproducibility"]["byte_identical"])
        self.assertTrue(summary["reproducibility"]["external_locks_byte_identical"])
        self.assertEqual(summary["reproducibility"]["mismatch_count"], 0)
        self.assertEqual(
            summary["memory"]["nvfp4_aligned_weight_arena_bytes"],
            14_696_668_160,
        )
        self.assertTrue(all(summary["memory"]["gates"].values()))
        self.assertTrue(summary["memory"]["not_model_execution"])
        self.assertEqual(
            summary["verification"]["protected_12b_inspect"]["tensor_count"],
            1389,
        )


if __name__ == "__main__":
    unittest.main()
