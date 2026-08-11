from __future__ import annotations

from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.gem16_compile.common import InvalidPlanError
from tools.gem16_compile.plan import load_quantization_plan
from tools.gem16_compile.profiles import (
    M05_APPROVED_SOURCE_LOCKS,
    M05_ATTENTION_TABLE,
    M05_PROFILE,
    M05_SOURCE_LOCK_SHA256,
)
from tools.gem16_compile.reader import TensorDescriptor, VerifiedSource
from tools.generate_gemma4_26b_fp8_plans import (
    DEFAULT_ORDINARY_INVENTORY,
    DEFAULT_OUTPUT,
    load_json,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = DEFAULT_OUTPUT / "ordinary-compiler-plan.json"
INVENTORY_PATH = DEFAULT_ORDINARY_INVENTORY


def _fake_sources() -> dict[str, TensorDescriptor]:
    inventory = load_json(INVENTORY_PATH)
    result: dict[str, TensorDescriptor] = {}
    for item in inventory["tensors"]:
        result[item["name"]] = TensorDescriptor(
            name=item["name"],
            dtype=item["dtype"],
            shape=tuple(item["shape"]),
            shard=item["shard"],
            path=ROOT / "does-not-exist.safetensors",
            absolute_offset=item["absolute_offset"],
            data_offset=item["data_offsets"][0] if "data_offsets" in item else 0,
            byte_length=item["bytes"],
            shard_sha256="0" * 64,
        )
    return result


def _verified(lock_sha: str, tensors: dict[str, TensorDescriptor]) -> VerifiedSource:
    return VerifiedSource(
        root=ROOT,
        lock_path=ROOT / "synthetic.lock.json",
        lock_sha256=lock_sha,
        repository="google/gemma-4-26b",
        revision="0" * 40,
        resolved_at_utc="2026-08-11T00:00:00Z",
        files={},
    )


class M05PlanContractTest(unittest.TestCase):
    def _load(self, document: dict, tensors=None, lock_sha=None):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return load_quantization_plan(
                path,
                _verified(lock_sha or M05_SOURCE_LOCK_SHA256["ordinary_bf16"], tensors or _fake_sources()),
                tensors or _fake_sources(),
                M05_PROFILE.name,
                M05_PROFILE.head_format,
            )

    def test_exact_production_table_and_approved_locks(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        sources = _fake_sources()
        loaded = self._load(plan, sources)
        self.assertEqual(len(M05_ATTENTION_TABLE), 115)
        self.assertEqual(len(loaded.tensors), 230)
        self.assertEqual(set(M05_APPROVED_SOURCE_LOCKS), set(M05_SOURCE_LOCK_SHA256.values()))

    def test_tiny_lock_and_wrong_source_contract_are_rejected(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        with self.assertRaises(InvalidPlanError):
            self._load(plan, lock_sha="0" * 64)
        wrong_contract = copy.deepcopy(plan)
        wrong_contract["source_contract"] = "synthetic-source"
        with self.assertRaises(InvalidPlanError):
            self._load(wrong_contract)

    def test_missing_extra_and_wrong_shape_attention_are_rejected(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        sources = _fake_sources()
        missing = dict(sources)
        missing.pop(next(iter(M05_ATTENTION_TABLE)))
        with self.assertRaises(InvalidPlanError):
            self._load(plan, missing)

        extra = dict(sources)
        name = "model.language_model.layers.5.self_attn.v_proj.weight"
        extra[name] = TensorDescriptor(
            name=name, dtype="BF16", shape=(1024, 2816), shard="x",
            path=ROOT / "missing", absolute_offset=0, data_offset=0,
            byte_length=1024 * 2816 * 2, shard_sha256="0" * 64,
        )
        with self.assertRaises(InvalidPlanError):
            self._load(plan, extra)

        wrong = dict(sources)
        name = "model.language_model.layers.0.self_attn.q_proj.weight"
        wrong[name] = replace(wrong[name], shape=(1, 1), byte_length=2)
        with self.assertRaises(InvalidPlanError):
            self._load(plan, wrong)

    def test_deferred_attention_and_false_exclusion_role_are_rejected(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        deferred = copy.deepcopy(plan)
        source_name = "model.language_model.layers.0.self_attn.q_proj.weight"
        deferred["tensors"] = [
            item for item in deferred["tensors"]
            if source_name not in item["source_names"]
        ]
        deferred["excluded_tensors"].append({
            "source_name": source_name,
            "family": "deferred_non_attention",
            "role": "attention_q_projection",
            "residency_class": "m05_deferred_non_attention",
            "reason": "deferred to M06-M08; absent from M05 attention-only partial artifact",
        })
        with self.assertRaises(InvalidPlanError):
            self._load(deferred)

        false_role = copy.deepcopy(plan)
        false_role["excluded_tensors"][0]["role"] = "wrong_role"
        with self.assertRaises(InvalidPlanError):
            self._load(false_role)

    def test_fabricated_global_v_is_rejected_even_if_plan_is_extended(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        sources = _fake_sources()
        name = "model.language_model.layers.5.self_attn.v_proj.weight"
        sources[name] = TensorDescriptor(
            name=name, dtype="BF16", shape=(1024, 2816), shard="x",
            path=ROOT / "missing", absolute_offset=0, data_offset=0,
            byte_length=1024 * 2816 * 2, shard_sha256="0" * 64,
        )
        with self.assertRaises(InvalidPlanError):
            self._load(plan, sources)


if __name__ == "__main__":
    unittest.main()
