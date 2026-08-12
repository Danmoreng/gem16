from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.gem16_compile.common import InvalidPlanError
from tools.gem16_compile.plan import load_quantization_plan
from tools.gem16_compile.profiles import M05_SOURCE_LOCK_SHA256, M06_PROFILE
from tools.gem16_compile.reader import TensorDescriptor, VerifiedSource
from tools.generate_gemma4_26b_nvfp4_plan import (
    OUTPUT_CONFIG,
    OUTPUT_PLAN,
    ROOT,
    load_json,
)


PLAN = load_json(OUTPUT_PLAN)
INVENTORY = load_json(
    ROOT / "benchmarks/goldens/gemma4_26b/source-inventories/qat-bf16.json"
)


def fake_sources() -> dict[str, TensorDescriptor]:
    return {
        item["name"]: TensorDescriptor(
            name=item["name"],
            dtype=item["dtype"],
            shape=tuple(item["shape"]),
            shard=item["shard"],
            path=ROOT / "does-not-exist.safetensors",
            absolute_offset=item["absolute_offset"],
            data_offset=0,
            byte_length=item["bytes"],
            shard_sha256="0" * 64,
        )
        for item in INVENTORY["tensors"]
    }


def verified(lock_sha: str) -> VerifiedSource:
    return VerifiedSource(
        root=ROOT,
        lock_path=ROOT / "models/gemma4-26b-qat-bf16.lock.json",
        lock_sha256=lock_sha,
        repository="google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
        revision="0" * 40,
        resolved_at_utc="2026-08-12T00:00:00Z",
        files={},
    )


class M06PlanContractTest(unittest.TestCase):
    def load(self, document=None, tensors=None, lock_sha=None):
        document = document or PLAN
        tensors = tensors or fake_sources()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return load_quantization_plan(
                path,
                verified(lock_sha or M05_SOURCE_LOCK_SHA256["qat_bf16"]),
                tensors,
                M06_PROFILE.name,
                M06_PROFILE.head_format,
            )

    def test_exact_counts_bytes_and_component_contract(self) -> None:
        loaded = self.load()
        self.assertEqual(len(loaded.tensors), 600)
        self.assertEqual(len(loaded.excluded_tensors), 863)
        self.assertEqual(loaded.output_tensor_bytes, 13_147_454_640)
        self.assertEqual(
            {item.output_name.rsplit(".", 1)[1] for item in loaded.tensors},
            {"weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"},
        )
        routed = [item for item in loaded.tensors if item.role == "routed_expert_gate_up"]
        self.assertTrue(routed)
        self.assertEqual(routed[0].logical_shape, (128, 1408, 2816))
        self.assertEqual(routed[0].axis_transformation, "expert,gate_then_up,input")
        packed = next(item for item in routed if item.encoder == "nvfp4-packed-v1")
        scale = next(item for item in routed if item.encoder == "nvfp4-local-scale-v1")
        scalar = next(item for item in routed if item.encoder == "nvfp4-weight-divisor-v1")
        self.assertEqual(packed.runtime_layout, "expert_major_sm120_row8_k64")
        self.assertEqual(scale.disk_layout, "canonical_row_major_group16_e4m3")
        self.assertEqual(scalar.disk_layout, "scalar_f32")

    def test_wrong_lock_and_component_mutations_fail_closed(self) -> None:
        with self.assertRaises(InvalidPlanError):
            self.load(lock_sha=M05_SOURCE_LOCK_SHA256["ordinary_bf16"])
        mutation = copy.deepcopy(PLAN)
        mutation["tensors"][0]["quantizer_parameters"]["global_scale_role"] = "multiplier"
        with self.assertRaises(InvalidPlanError):
            self.load(mutation)
        mutation = copy.deepcopy(PLAN)
        mutation["tensors"][0]["physical_shape"][-1] += 1
        with self.assertRaises(InvalidPlanError):
            self.load(mutation)

    def test_missing_component_and_fused_axis_fail(self) -> None:
        mutation = copy.deepcopy(PLAN)
        mutation["tensors"] = mutation["tensors"][1:]
        with self.assertRaises(InvalidPlanError):
            self.load(mutation)
        mutation = copy.deepcopy(PLAN)
        for item in mutation["tensors"]:
            if item["role"] == "routed_expert_gate_up":
                item["axis_transformation"] = "expert,up_then_gate,input"
                break
        with self.assertRaises(InvalidPlanError):
            self.load(mutation)

    def test_inventory_mutations_fail_closed(self) -> None:
        missing = fake_sources()
        missing.pop(next(iter(missing)))
        with self.assertRaises(InvalidPlanError):
            self.load(tensors=missing)
        extra = fake_sources()
        extra["model.language_model.layers.0.not_a_tensor"] = TensorDescriptor(
            name="model.language_model.layers.0.not_a_tensor", dtype="BF16", shape=(1,),
            shard="source.bin", path=ROOT / "does-not-exist.safetensors", absolute_offset=0,
            data_offset=0, byte_length=2, shard_sha256="0" * 64,
        )
        with self.assertRaises(InvalidPlanError):
            self.load(tensors=extra)
        malformed = fake_sources()
        malformed["model.language_model.layers.0.mlp.gate_proj.weight"] = TensorDescriptor(
            name="model.language_model.layers.0.mlp.gate_proj.weight", dtype="F16",
            shape=(2112, 2816), shard="source.bin", path=ROOT / "does-not-exist.safetensors",
            absolute_offset=0, data_offset=0, byte_length=2112 * 2816 * 2,
            shard_sha256="0" * 64,
        )
        with self.assertRaises(InvalidPlanError):
            self.load(tensors=malformed)

    def test_schema_is_strict_for_m06_contract(self) -> None:
        plan_schema = load_json(ROOT / "tools/gem16_compile/schemas/compiler-plan.schema.json")
        compilation_schema = load_json(ROOT / "tools/gem16_compile/schemas/gem16-compilation.schema.json")
        m06 = plan_schema["$defs"]["m06QuantizerParameters"]
        self.assertFalse(m06["additionalProperties"])
        self.assertIn("signed_zero", m06["required"])
        self.assertEqual(m06["properties"]["saturation"]["const"], "finite_saturation")
        self.assertEqual(
            compilation_schema["$defs"]["compilerM06"]["allOf"][1]["properties"]["native_encoder"]["properties"]["protocol"]["const"],
            "gem16-nvfp4-direct-v1",
        )
        self.assertFalse(compilation_schema["$defs"]["quantizationM06"]["additionalProperties"])

    def test_generator_and_config_are_deterministic_and_sampled(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/generate_gemma4_26b_nvfp4_plan.py"), "--check"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        config = load_json(OUTPUT_CONFIG)
        self.assertEqual(config["diagnostic_sample"]["logical_matrix_count"], 48)
        self.assertEqual(len(config["diagnostic_sample"]["records"]), 48)
        self.assertEqual(config["diagnostic_sample"]["acceptance"]["status"], "not_run")
        self.assertIn("relative_l2", config["diagnostic_sample"]["acceptance"]["per_matrix"]["required_metrics"])
        records = config["diagnostic_sample"]["records"]
        self.assertTrue(all("source_tensor" in item and "unsloth_component" in item for item in records))
        self.assertEqual({item["layer"] for item in records}, {0, 5, 24, 29})
        self.assertEqual({item["expert"] for item in records if "expert" in item}, {0, 63, 127})
        self.assertTrue(all(item["unsloth_component"].endswith(".weight_packed") for item in records))
        self.assertTrue(all("range" in item["unsloth_component_range"] for item in records))
        self.assertEqual(config["diagnostics"]["ordinary_vs_unsloth"], "not_run")
        self.assertEqual(config["counts"]["expert_shared_output_tensor_bytes"], 13_147_454_640)
        self.assertEqual(len(config["quantizer"]["spec_sha256"]), 64)
        spec = load_json(ROOT / "tools/gem16_compile/specs/nvfp4-experts-v1.json")
        self.assertEqual(config["quantizer"]["contract"], {k: spec[k] for k in config["quantizer"]["contract"]})


if __name__ == "__main__":
    unittest.main()
