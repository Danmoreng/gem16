"""Bounded M06 diagnostic contract tests (no model/framework imports)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from tools.run_gemma4_26b_nvfp4_diagnostic import (
    DataError, InvalidPlanError, _Cursor, _build_parent_geometries,
    _diagnostic_exit_code, _finish_metrics, _native_job, _record_geometry,
    _validate_config, run_diagnostic,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "artifacts/m06/nvfp4-compiler-config.json"
ORDINARY = ROOT / "benchmarks/goldens/gemma4_26b/source-inventories/ordinary-bf16.json"
UNSLOTH = ROOT / "benchmarks/goldens/gemma4_26b/source-inventories/unsloth-nvfp4.json"
DEBUG_NATIVE = ROOT / "build/Linux/host-debug/bin/gem16-checkpoint-compiler"


class M06DiagnosticContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text())
        self.ordinary = json.loads(ORDINARY.read_text())
        self.unsloth = json.loads(UNSLOTH.read_text())
        self.ordinary_map = {x["name"]: x for x in self.ordinary["tensors"]}

    def test_frozen_sample_has_48_gate_up_down_and_split_experts(self) -> None:
        records = _validate_config(self.config)
        self.assertEqual(48, len(records))
        self.assertEqual({"gate", "up", "down"}, {x["projection"] for x in records})
        routed = [x for x in records if x["kind"] == "routed"]
        self.assertEqual({0, 63, 127}, {x["expert"] for x in routed})
        gate = next(x for x in routed if x["projection"] == "gate")
        up = next(x for x in routed if x["projection"] == "up")
        self.assertEqual([0], gate["source_slice"]["start"][1:2])
        self.assertEqual([704], up["source_slice"]["start"][1:2])

    def test_parent_jobs_preserve_full_source_divisors_and_gate_up_offsets(self) -> None:
        records = _validate_config(self.config)
        from tools.gem16_compile.common import BoundedWorkspace
        workspace = BoundedWorkspace(512 * 1024 * 1024, 4096)
        with patch.object(workspace, "hash_range", return_value="a" * 64):
            parents, geometries = _build_parent_geometries(records, self._descriptors(), self.unsloth, workspace)
        self.assertEqual(20, len(parents))
        self.assertEqual(48, len(geometries))
        gate = next(x for x in geometries if x["record"]["kind"] == "routed" and x["record"]["projection"] == "gate" and x["record"]["layer"] == 0 and x["record"]["expert"] == 63)
        up = next(x for x in geometries if x["record"]["kind"] == "routed" and x["record"]["projection"] == "up" and x["record"]["layer"] == 0 and x["record"]["expert"] == 63)
        self.assertEqual(gate["parent_name"], up["parent_name"])
        self.assertEqual(63 * 1408, gate["parent_row_start"])
        self.assertEqual(63 * 1408 + 704, up["parent_row_start"])
        with tempfile.TemporaryDirectory() as directory:
            with patch("tools.run_gemma4_26b_nvfp4_diagnostic._reserve"):
                job = _native_job(parents, Path(directory), 4)
            self.assertEqual(20, len(job["operations"]))
            self.assertEqual(48, sum(len(parent["records"]) for parent in parents))
            self.assertEqual(gate["parent_row_start"] * (gate["parent_columns"] // 2), gate["output"]["packed"]["offset"])
            self.assertEqual(up["parent_row_start"] * (up["parent_columns"] // 2), up["output"]["packed"]["offset"])
            self.assertNotEqual(gate["output"]["packed"]["offset"], up["output"]["packed"]["offset"])

    def test_thresholds_cover_all_three_relationships_and_cannot_weaken(self) -> None:
        self.assertEqual(set(self.config["diagnostic_sample"]["acceptance"]["thresholds"]),
                         {"ordinary_compiled", "unsloth_reference", "compiled_vs_unsloth"})
        bad = copy.deepcopy(self.config)
        bad["diagnostic_sample"]["acceptance"]["thresholds"]["compiled_vs_unsloth"]["cosine_min"] = 0.5
        with self.assertRaises(DataError):
            _validate_config(bad)

    def test_failed_threshold_report_returns_failure_exit(self) -> None:
        self.assertEqual(_diagnostic_exit_code({"status": "pass", "pass": True}), 0)
        self.assertEqual(_diagnostic_exit_code({"status": "fail", "pass": False}), 4)
        self.assertEqual(_diagnostic_exit_code({"status": "pass", "pass": False}), 4)

    def test_shape_mismatch_and_ambiguous_divisor_fail_closed(self) -> None:
        bad = copy.deepcopy(self.config)
        bad["diagnostic_sample"]["records"][0]["source_slice"]["stop"] = [2111, 2816]
        with self.assertRaises(DataError):
            _record_geometry(bad["diagnostic_sample"]["records"][0], self._descriptors(), self.unsloth)
        bad = copy.deepcopy(self.config)
        bad["quantizer"]["contract"]["global_scale_role"] = "multiplier"
        with self.assertRaises(DataError):
            _validate_config(bad)

    def test_unsloth_packed_and_scale_shapes_include_down(self) -> None:
        records = _validate_config(self.config)
        for record in records:
            geometry = _record_geometry(record, self._descriptors(), self.unsloth)
            rows, columns = geometry["rows"], geometry["columns"]
            self.assertEqual([rows, columns // 2], geometry["components"]["packed"]["shape"])
            self.assertEqual([rows, columns // 16], geometry["components"]["local"]["shape"])
            self.assertEqual([1], geometry["components"]["weight_global"]["shape"])

    def test_bounded_cursor_never_reads_more_than_buffer(self) -> None:
        class Guarded:
            def __init__(self, raw: bytes):
                self.raw, self.position, self.maximum = raw, 0, 0
            def readinto(self, view):
                self.maximum = max(self.maximum, len(view))
                count = min(len(view), len(self.raw) - self.position)
                view[:count] = self.raw[self.position:self.position + count]
                self.position += count
                return count
            def seek(self, offset): self.position = offset
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.bin"
            path.write_bytes(bytes(range(31)))
            cursor = _Cursor(path, 2, 27, 5)
            self.assertEqual(bytes(range(2, 29)), bytes(cursor._byte() for _ in range(27)))
            self.assertLessEqual(len(cursor.buffer), 5)
            cursor.close()

    @unittest.skipUnless(DEBUG_NATIVE.is_file(), "host Debug M06 compiler is not built")
    def test_debug_diagnostic_fails_before_checkpoint_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with self.assertRaisesRegex(InvalidPlanError, "Release build"):
                run_diagnostic(
                    ordinary_root=Path("/definitely/not/read/ordinary"),
                    ordinary_inventory=Path("/definitely/not/read/ordinary.json"),
                    unsloth_root=Path("/definitely/not/read/unsloth"),
                    unsloth_inventory=Path("/definitely/not/read/unsloth.json"),
                    config_path=CONFIG,
                    native_encoder=DEBUG_NATIVE,
                    output=output,
                    max_host_memory=512 * 1024 * 1024,
                    chunk_bytes=4096,
                )
            self.assertFalse(output.exists())

    def test_zero_norm_and_nonfinite_metrics_are_rejected_by_report_path(self) -> None:
        with self.assertRaises(DataError):
            _finish_metrics({"elements": 1, "source_sq": 0.0, "ref_sq": 0.0, "diff_sq": 0.0,
                             "dot": 0.0, "max_abs": 0.0, "codes": [0] * 16, "scales": [0] * 256}, "zero")
        with self.assertRaises(DataError):
            _finish_metrics({"elements": 1, "source_sq": float("nan"), "ref_sq": 1.0, "diff_sq": 1.0,
                             "dot": 0.0, "max_abs": 1.0, "codes": [0] * 16, "scales": [0] * 256}, "nonfinite")
        # The public diagnostic must not emit NaN/Infinity for degenerate
        # selected ranges; shape validation also fails before native work.
        bad = copy.deepcopy(self.config)
        bad["diagnostic_sample"]["records"][0]["source_slice"]["start"] = [0, 0]
        bad["diagnostic_sample"]["records"][0]["source_slice"]["stop"] = [0, 0]
        with self.assertRaises(DataError):
            _record_geometry(bad["diagnostic_sample"]["records"][0], self._descriptors(), self.unsloth)

    def _descriptors(self):
        # Geometry only needs the strict reader descriptor fields for this
        # unit-level mapping test; use a real descriptor-like object.
        from tools.gem16_compile.reader import TensorDescriptor
        return {
            name: TensorDescriptor(name, item["dtype"], tuple(item["shape"]), item["shard"], Path("/locked"),
                                   item["absolute_offset"], item["absolute_offset"], item["bytes"], "0" * 64)
            for name, item in self.ordinary_map.items()
        }


if __name__ == "__main__":
    unittest.main()
