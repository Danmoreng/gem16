from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.generate_gemma4_26b_fp8_plans import (
    DEFAULT_CONFIG,
    DEFAULT_ORDINARY_INVENTORY,
    DEFAULT_OUTPUT,
    DEFAULT_QAT_INVENTORY,
    generate,
    load_json,
    make_plan,
    plan_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
ORDINARY_LOCK = ROOT / "models/gemma4-26b-base-bf16.lock.json"
QAT_LOCK = ROOT / "models/gemma4-26b-qat-bf16.lock.json"
ORDINARY_PLAN = DEFAULT_OUTPUT / "ordinary-compiler-plan.json"
QAT_PLAN = DEFAULT_OUTPUT / "qat-compiler-plan.json"


class Gemma426BFP8PlanTest(unittest.TestCase):
    def test_checked_outputs_are_current(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated = generate(Path(temporary) / "fp8", Path(temporary) / "config.json")
            self.assertEqual(generated[Path(temporary) / "fp8" / "ordinary-compiler-plan.json"], ORDINARY_PLAN.read_bytes())
            self.assertEqual(generated[Path(temporary) / "fp8" / "qat-compiler-plan.json"], QAT_PLAN.read_bytes())
            self.assertEqual(generated[Path(temporary) / "config.json"], DEFAULT_CONFIG.read_bytes())

    def test_ordinary_and_qat_structures_differ_only_by_lock_hash(self) -> None:
        ordinary = json.loads(ORDINARY_PLAN.read_text(encoding="utf-8"))
        qat = json.loads(QAT_PLAN.read_text(encoding="utf-8"))
        ordinary_hash = ordinary.pop("source_lock_sha256")
        qat_hash = qat.pop("source_lock_sha256")
        self.assertNotEqual(ordinary_hash, qat_hash)
        self.assertEqual(ordinary, qat)

    def test_exact_attention_totals_and_global_v_omission(self) -> None:
        plan = json.loads(ORDINARY_PLAN.read_text(encoding="utf-8"))
        weights = [item for item in plan["tensors"] if item["encoder"] == "fp8-rowwise-weight-v1"]
        scales = [item for item in plan["tensors"] if item["encoder"] == "fp8-rowwise-scale-v1"]
        self.assertEqual(len(weights), 115)
        self.assertEqual(len(scales), 115)
        self.assertEqual(len(plan["tensors"]), 230)
        self.assertEqual(len(plan["excluded_tensors"]), 898)
        self.assertEqual(sum(item["family"] == "vision" for item in plan["excluded_tensors"]), 356)
        self.assertEqual(sum(item["family"] == "deferred_non_attention" for item in plan["excluded_tensors"]), 542)
        self.assertEqual(sum(item["output_dtype"] == "F8_E4M3" and item["physical_shape"][0] * item["physical_shape"][1] for item in plan["tensors"]), 1_110_179_840)
        self.assertEqual(sum(
            item["physical_shape"][0] * item["physical_shape"][1] * 2
            for item in scales
        ), 670_720)
        self.assertEqual(
            sum(
                (item["physical_shape"][0] * item["physical_shape"][1])
                * (1 if item["output_dtype"] == "F8_E4M3" else 2)
                for item in plan["tensors"]
            ),
            1_110_850_560,
        )

    def test_source_coverage_and_role_shape_totals(self) -> None:
        inventory = load_json(DEFAULT_ORDINARY_INVENTORY)
        source_names = {str(item["name"]) for item in inventory["tensors"]}
        plan = json.loads(ORDINARY_PLAN.read_text(encoding="utf-8"))
        covered = {name for item in plan["tensors"] for name in item["source_names"]}
        covered.update(item["source_name"] for item in plan["excluded_tensors"])
        self.assertEqual(covered, source_names)
        roles = {role: 0 for role in ("attention_q_projection", "attention_k_projection", "attention_v_projection", "attention_o_projection")}
        for item in plan["tensors"]:
            if item["encoder"] == "fp8-rowwise-weight-v1":
                roles[item["role"]] += 1
                self.assertEqual(item["logical_shape"], item["physical_shape"])
            else:
                self.assertEqual(item["physical_shape"][1], 1)
                self.assertEqual(item["logical_dtype"], "BF16")
        self.assertEqual(roles, {
            "attention_q_projection": 30,
            "attention_k_projection": 30,
            "attention_v_projection": 25,
            "attention_o_projection": 30,
        })
        for layer in (5, 11, 17, 23, 29):
            self.assertFalse(any(f"layers.{layer}.self_attn.v_proj" in item["output_name"] for item in plan["tensors"]))

    def test_lock_and_inventory_identity_and_config(self) -> None:
        config = load_json(DEFAULT_CONFIG)
        for inventory_path, lock_path, key in (
            (DEFAULT_ORDINARY_INVENTORY, ORDINARY_LOCK, "ordinary_bf16"),
            (DEFAULT_QAT_INVENTORY, QAT_LOCK, "qat_bf16"),
        ):
            inventory = load_json(inventory_path)
            lock = load_json(lock_path)
            lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
            self.assertEqual(inventory["source"]["lock_sha256"], lock_hash)
            self.assertEqual(inventory["source"]["revision"], lock["revision"])
            self.assertEqual(config["plans"][key]["source_lock_sha256"], lock_hash)
            self.assertEqual(config["plans"][key]["source_revision"], lock["revision"])
        self.assertEqual(config["attention"]["matrix_count"], 115)
        self.assertEqual(config["attention"]["output_tensor_count"], 230)
        self.assertEqual(config["attention"]["output_tensor_bytes"], 1_110_850_560)
        self.assertEqual(config["exclusions"], {
            "deferred_non_attention_count": 542,
            "deferred_reason": "deferred to M06-M08; absent from M05 attention-only partial artifact",
            "total_count": 898,
            "vision_count": 356,
            "vision_reason": "text-only Gemma 4 26B profile excludes vision tensors",
        })
        self.assertFalse(Path(config["plans"]["ordinary_bf16"]["path"]).is_absolute())
        self.assertNotIn("timestamp", json.dumps(config).lower())


if __name__ == "__main__":
    unittest.main()
