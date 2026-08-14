"""Validate compact M05 evidence without requiring retained raw reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
M05 = ROOT / "artifacts/m05"
ACCEPTANCE = M05 / "acceptance.json"
SUMMARY = M05 / "qat-fp8-summary.json"
RAW_INDEX = ROOT / "artifacts/raw-evidence-index.json"


def load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M05CompactEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.acceptance = load(ACCEPTANCE)
        self.summary = load(SUMMARY)
        self.raw_index = load(RAW_INDEX)

    def test_compact_summary_reconciles(self) -> None:
        self.assertEqual(self.summary["evidence_class"], "clean_revision_evidence")
        self.assertEqual(self.summary["totals"]["matrix_count"], 115)
        self.assertEqual(self.summary["totals"]["output_tensor_count"], 230)
        self.assertEqual(self.summary["totals"]["output_tensor_bytes"], 1_110_850_560)
        self.assertEqual(self.summary["aggregate_qat"]["histogram_sum"], 1_110_179_840)
        self.assertEqual(
            sum(self.summary["aggregate_qat"]["histogram"]),
            self.summary["totals"]["weight_elements"],
        )
        self.assertEqual(len(self.summary["aggregate_qat"]["histogram"]), 256)
        self.assertEqual(self.summary["global_layers_without_v"], [5, 11, 17, 23, 29])

    def test_compact_summary_is_bound_by_acceptance(self) -> None:
        reports = self.acceptance["semantic_reports"]
        self.assertEqual(reports["qat_summary"]["path"], "artifacts/m05/qat-fp8-summary.json")
        self.assertEqual(reports["qat_summary"]["sha256"], sha256(SUMMARY))
        expected_provenance = {
            "source_lock_sha256", "compiler_plan_sha256", "resolved_plan_sha256",
            "compilation_manifest_sha256", "compile_report_sha256",
            "compiler_commit", "compiler_dirty",
        }
        self.assertEqual(set(self.summary["source"]), expected_provenance)
        self.assertFalse(self.summary["source"]["compiler_dirty"])

    def test_raw_index_is_complete_and_hash_only(self) -> None:
        self.assertEqual(self.raw_index["retention_policy"], "hash_only_not_tracked")
        self.assertEqual(self.raw_index["ignored_root"], "artifacts/raw")
        reports = self.raw_index["raw_reports"]
        self.assertEqual(len(reports), 14)
        self.assertEqual(sum(item["size"] for item in reports), 15_181_271)
        self.assertEqual(
            {milestone: sum(item["milestone"] == milestone for item in reports)
             for milestone in ("M05", "M06", "M07")},
            {"M05": 7, "M06": 4, "M07": 3},
        )
        originals = {item["original_path"] for item in reports}
        ignored = {item["ignored_path"] for item in reports}
        self.assertEqual(len(originals), len(reports))
        self.assertEqual(len(ignored), len(reports))
        for item in reports:
            self.assertRegex(item["sha256"], re.compile(r"^[0-9a-f]{64}$"))
            self.assertTrue(item["ignored_path"].startswith("artifacts/raw/"))
            self.assertFalse((ROOT / item["original_path"]).exists())
            ignored_path = ROOT / item["ignored_path"]
            if ignored_path.exists():
                self.assertEqual(ignored_path.stat().st_size, item["size"])
                self.assertEqual(sha256(ignored_path), item["sha256"])

    def test_acceptance_points_to_hash_only_index(self) -> None:
        retention = self.acceptance["evidence_retention"]
        self.assertEqual(retention["policy"], "compact_acceptance_and_hashes")
        self.assertEqual(retention["raw_reports"], "hash_only_not_tracked")
        self.assertEqual(retention["raw_evidence_index"], "artifacts/raw-evidence-index.json")
        by_original = {
            item["original_path"]: item for item in self.raw_index["raw_reports"]
            if item["milestone"] == "M05"
        }
        self.assertEqual(
            by_original["artifacts/m05/fp8-tensor-report.json"]["sha256"],
            self.acceptance["semantic_reports"]["fp8_tensor_report"]["sha256"],
        )
        self.assertEqual(
            by_original["artifacts/m05/ordinary-vs-unsloth-fp8.json"]["sha256"],
            self.acceptance["ordinary_vs_unsloth"]["sha256"],
        )

    def test_ignore_rule_is_explicit(self) -> None:
        lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("/artifacts/raw/", lines)


if __name__ == "__main__":
    unittest.main()
