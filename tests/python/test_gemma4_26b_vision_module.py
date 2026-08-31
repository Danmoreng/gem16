from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.gem16_compile.reader import TensorDescriptor
from tools.gem16_compile.vision_module import (
    BF16_COPY_COUNT,
    LINEAR_COUNT,
    LINEAR_PARAMETERS,
    OUTPUT_PAYLOAD_BYTES,
    OUTPUT_PADDING_BYTES,
    OUTPUT_TENSOR_BYTES,
    OUTPUT_TENSOR_COUNT,
    PROFILE,
    SOURCE_TENSOR_BYTES,
    SOURCE_TENSOR_COUNT,
    SPEC_PATH,
    TEXT_ARTIFACT_PROFILE,
    TENSOR_ALIGNMENT,
    expected_vision_specs,
    fp8_plan,
    output_plan,
    validate_vision_sources,
)
from tools.gem16_compile.common import DataError
from tools.gem16_compile.vision_module_verify import verify_vision_module


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "benchmarks/goldens/gemma4_26b/source-inventories/qat-bf16.json"


def _descriptors() -> dict[str, TensorDescriptor]:
    result: dict[str, TensorDescriptor] = {}
    for name, (_role, shape) in expected_vision_specs().items():
        byte_length = 2
        for dimension in shape:
            byte_length *= dimension
        result[name] = TensorDescriptor(
            name=name,
            dtype="BF16",
            shape=shape,
            shard="fixture.safetensors",
            path=Path("/fixture.safetensors"),
            absolute_offset=0,
            data_offset=0,
            byte_length=byte_length,
            shard_sha256="0" * 64,
        )
    return result


class Gemma426BVisionModuleTest(unittest.TestCase):
    def test_frozen_google_inventory_is_exactly_the_vision_source(self) -> None:
        inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
        tensors = {
            item["name"]: item
            for item in inventory["tensors"]
            if item["name"] == "model.embed_vision.embedding_projection.weight"
            or item["name"].startswith("model.vision_tower.")
        }
        expected = expected_vision_specs()
        self.assertEqual(set(tensors), set(expected))
        self.assertEqual(len(tensors), SOURCE_TENSOR_COUNT)
        self.assertEqual(sum(item["bytes"] for item in tensors.values()), SOURCE_TENSOR_BYTES)
        for name, (_role, shape) in expected.items():
            self.assertEqual(tensors[name]["dtype"], "BF16")
            self.assertEqual(tuple(tensors[name]["shape"]), shape)

    def test_output_plan_has_exact_fp8_and_bf16_payload_balance(self) -> None:
        selected = validate_vision_sources(_descriptors())
        outputs = output_plan(selected)
        fp8 = [item for item in outputs if item.kind == "fp8_weight"]
        scales = [item for item in outputs if item.kind == "fp8_scale"]
        copies = [item for item in outputs if item.kind == "bf16_copy"]
        self.assertEqual((len(fp8), len(scales), len(copies)), (LINEAR_COUNT, LINEAR_COUNT, BF16_COPY_COUNT))
        self.assertEqual(len(outputs), OUTPUT_TENSOR_COUNT)
        self.assertEqual(sum(item.byte_length for item in outputs), OUTPUT_TENSOR_BYTES)
        cursor = 0
        for item in outputs:
            cursor = (cursor + TENSOR_ALIGNMENT - 1) & -TENSOR_ALIGNMENT
            cursor += item.byte_length
        self.assertEqual(cursor, OUTPUT_PAYLOAD_BYTES)
        self.assertEqual(OUTPUT_PAYLOAD_BYTES - OUTPUT_TENSOR_BYTES, OUTPUT_PADDING_BYTES)
        self.assertEqual(sum(item.byte_length for item in fp8), LINEAR_PARAMETERS)
        self.assertEqual(len({item.name for item in outputs}), len(outputs))

    def test_native_plan_uses_no_hidden_shape_padding(self) -> None:
        selected = validate_vision_sources(_descriptors())
        plan = fp8_plan(selected)
        self.assertEqual(len(plan.tensors), LINEAR_COUNT * 2)
        by_name = {item.output_name: item for item in plan.tensors}
        down = by_name["model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight"]
        self.assertEqual(down.logical_shape, (1152, 4304))
        self.assertEqual(down.physical_shape, (1152, 4304))
        q_norm = selected["model.vision_tower.encoder.layers.0.self_attn.q_norm.weight"]
        self.assertEqual(q_norm.shape, (72,))
        self.assertEqual(plan.output_tensor_bytes, LINEAR_PARAMETERS + sum(
            item.physical_shape[0] * 2
            for item in plan.tensors
            if item.encoder == "fp8-rowwise-scale-v1"
        ))

    def test_profile_is_explicitly_trellis35_only(self) -> None:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        self.assertEqual(PROFILE, "gemma4_26b_trellis35_vision_fp8")
        self.assertEqual(spec["capability_profile"], PROFILE)
        self.assertEqual(spec["required_text_artifact_profile"], TEXT_ARTIFACT_PROFILE)
        self.assertNotIn("nvfp4", spec["required_text_artifact_profile"])
        self.assertEqual(spec["physical_shapes"]["attention_head_72"],
                         "logical 72; explicit kernel tail; no stored padding")
        self.assertEqual(spec["physical_shapes"]["mlp_down_k_4304"],
                         "logical and physical 4304; explicit kernel tail; no loader padding")

    def test_source_inventory_mismatch_fails_closed(self) -> None:
        descriptors = _descriptors()
        descriptors.pop("model.vision_tower.std_bias")
        with self.assertRaises(DataError):
            validate_vision_sources(descriptors)
        descriptors = _descriptors()
        original = descriptors["model.vision_tower.std_scale"]
        descriptors[original.name] = TensorDescriptor(
            original.name, original.dtype, (1151,), original.shard,
            original.path, original.absolute_offset, original.data_offset,
            original.byte_length - 2, original.shard_sha256,
        )
        with self.assertRaises(DataError):
            validate_vision_sources(descriptors)

    def test_verifier_rejects_noncanonical_file_set_before_payload_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("vision.gem16", "gem16_vision.json",
                         "vision_compilation.json", "vision.lock.json", "extra"):
                (root / name).touch()
            with self.assertRaises(DataError):
                verify_vision_module(root)


if __name__ == "__main__":
    unittest.main()
