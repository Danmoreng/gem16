from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.generate_gemma4_26b_m05_reports import (
    DEFAULT_QAT,
    DEFAULT_QAT_PLAN,
    DEFAULT_ORDINARY,
    DEFAULT_ORDINARY_PLAN,
    DEFAULT_CONFIG,
    DEFAULT_SPEC,
    ReportError,
    generate,
)
from tools.gem16_compile.common import canonical_json_bytes


class M05RetainedReportsTest(unittest.TestCase):
    def test_checked_inputs_generate_current_reports_and_reconcile(self) -> None:
        tensor, summary = generate()
        self.assertEqual(tensor["evidence_class"], "clean_revision_evidence")
        self.assertEqual(tensor["totals"]["matrix_count"], 115)
        self.assertEqual(tensor["totals"]["output_tensor_count"], 230)
        self.assertEqual(summary["aggregate_qat"]["histogram_sum"], 1_110_179_840)
        expected_provenance = {
            "source_lock_sha256", "compiler_plan_sha256", "resolved_plan_sha256",
            "compilation_manifest_sha256", "compile_report_sha256",
            "compiler_commit", "compiler_dirty",
        }
        for record in (tensor["sources"]["ordinary_bf16"],
                       tensor["sources"]["qat_bf16"], summary["source"]):
            self.assertEqual(set(record), expected_provenance)
            self.assertFalse(record["compiler_dirty"])
            self.assertRegex(record["compiler_commit"], r"^[0-9a-f]{40}$")
            for key, value in record.items():
                if key.endswith("sha256"):
                    self.assertRegex(value, r"^[0-9a-f]{64}$")
        self.assertEqual(
            summary["limitations"], [
                "Telemetry summarizes stored weights and row scales only.",
                "No model-quality or QAT attribution claim is made.",
                "Compiler provenance binds both lanes to one clean implementation commit.",
            ]
        )
        self.assertEqual(sum(summary["aggregate_qat"]["histogram"]),
                         summary["totals"]["weight_elements"])
        self.assertEqual(len(summary["aggregate_qat"]["histogram"]), 256)
        self.assertEqual(
            Path("artifacts/m05/fp8-tensor-report.json").read_bytes(),
            canonical_json_bytes(tensor),
        )
        self.assertEqual(
            Path("artifacts/m05/qat-fp8-summary.json").read_bytes(),
            canonical_json_bytes(summary),
        )

    def test_mutated_histogram_is_rejected_by_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ordinary = json.loads(DEFAULT_ORDINARY.read_text(encoding="utf-8"))
            ordinary["fp8_tensor_statistics"][0]["statistics"]["histogram"][0] += 1
            ordinary_path = root / "ordinary.json"
            ordinary_path.write_bytes(canonical_json_bytes(ordinary))
            with self.assertRaises(ReportError):
                generate(ordinary_path, DEFAULT_QAT, DEFAULT_CONFIG,
                         DEFAULT_ORDINARY_PLAN, DEFAULT_QAT_PLAN, DEFAULT_SPEC)

    def test_invalid_compiler_commit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ordinary = json.loads(DEFAULT_ORDINARY.read_text(encoding="utf-8"))
            ordinary["compiler_commit"] = "not-a-commit"
            ordinary_path = root / "ordinary.json"
            ordinary_path.write_bytes(canonical_json_bytes(ordinary))
            with self.assertRaises(ReportError):
                generate(ordinary_path, DEFAULT_QAT, DEFAULT_CONFIG,
                         DEFAULT_ORDINARY_PLAN, DEFAULT_QAT_PLAN, DEFAULT_SPEC)

    def test_mismatched_native_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qat = json.loads(DEFAULT_QAT.read_text(encoding="utf-8"))
            qat["native_encoder"]["sha256"] = "a" * 64
            qat_path = root / "qat.json"
            qat_path.write_bytes(canonical_json_bytes(qat))
            with self.assertRaises(ReportError):
                generate(DEFAULT_ORDINARY, qat_path, DEFAULT_CONFIG,
                         DEFAULT_ORDINARY_PLAN, DEFAULT_QAT_PLAN, DEFAULT_SPEC)

    def test_global_v_is_absent_and_all_matrix_records_have_two_lanes(self) -> None:
        tensor, _summary = generate()
        self.assertEqual(tensor["global_layers_without_v"], [5, 11, 17, 23, 29])
        self.assertEqual(len(tensor["matrices"]), 115)
        self.assertTrue(all(set(matrix["ordinary"]) == {"weight", "scale"} and
                            set(matrix["qat"]) == {"weight", "scale"}
                            for matrix in tensor["matrices"]))
        self.assertFalse(any("layers.5.self_attn.v_proj" in matrix["source_name"]
                             for matrix in tensor["matrices"]))


if __name__ == "__main__":
    unittest.main()
