from __future__ import annotations

from dataclasses import replace
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.gem16_compile.common import InvalidPlanError
from tools.gem16_compile.plan import load_quantization_plan
from tools.gem16_compile.profiles import (
    M25_PROFILE,
    M25_SOURCE_CONTRACT,
    M25_SOURCE_LOCK_SHA256,
)
from tools.gem16_compile.reader import LockedFile, TensorDescriptor, VerifiedSource
from tools.verify_gemma4_26b_assistant_hybrid import (
    decode_bf16,
    decode_e2m1,
    decode_e4m3fn,
)


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = (
    ROOT / "benchmarks/goldens/gemma4_26b/source-inventories/"
    "qat-q4_0-assistant-bf16.json"
)
PLAN = (
    ROOT / "benchmarks/goldens/gemma4_26b/mtp-assistant/"
    "qat-q4_0-hybrid-compiler-plan.json"
)
GENERATOR = ROOT / "tools/generate_gemma4_26b_assistant_plan.py"
METADATA = (
    "chat_template.jinja", "config.json", "generation_config.json",
    "tokenizer.json", "tokenizer_config.json",
)


def fake_sources() -> dict[str, TensorDescriptor]:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    return {
        item["name"]: TensorDescriptor(
            name=item["name"],
            dtype=item["dtype"],
            shape=tuple(item["shape"]),
            shard=item["shard"],
            path=ROOT / "missing-model.safetensors",
            absolute_offset=item["absolute_offset"],
            data_offset=0,
            byte_length=item["bytes"],
            shard_sha256="0" * 64,
        )
        for item in inventory["tensors"]
    }


def verified_source(lock_sha: str = M25_SOURCE_LOCK_SHA256) -> VerifiedSource:
    files = {
        name: LockedFile(name, ROOT / "missing" / name, 1, "0" * 64)
        for name in METADATA
    }
    return VerifiedSource(
        root=ROOT,
        lock_path=ROOT / "models/gemma4-26b-qat-q4_0-assistant.lock.json",
        lock_sha256=lock_sha,
        repository="google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant",
        revision="9537141506fe8875b3ed45b264af13580cb29166",
        resolved_at_utc="2026-08-25T14:23:10Z",
        files=files,
    )


class M25AssistantPlanContractTest(unittest.TestCase):
    def load(self, document: dict, sources=None, lock_sha=M25_SOURCE_LOCK_SHA256):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return load_quantization_plan(
                path,
                verified_source(lock_sha),
                sources or fake_sources(),
                M25_PROFILE.name,
                M25_PROFILE.head_format,
            )

    def test_generated_plan_and_exact_hybrid_contract(self) -> None:
        subprocess.run(
            [sys.executable, str(GENERATOR), "--check"], cwd=ROOT, check=True
        )
        plan = self.load(json.loads(PLAN.read_text(encoding="utf-8")))
        self.assertEqual(plan.source_contract, M25_SOURCE_CONTRACT)
        self.assertEqual(len(plan.tensors), 97)
        self.assertEqual(plan.output_tensor_bytes, 258_306_160)
        encoders = {tensor.encoder for tensor in plan.tensors}
        self.assertEqual(encoders, {
            "copy-v1", "fp8-rowwise-scale-v1", "fp8-rowwise-weight-v1",
            "nvfp4-input-divisor-v1", "nvfp4-local-scale-v1",
            "nvfp4-packed-v1", "nvfp4-weight-divisor-v1",
        })

    def test_independent_oracle_codecs_use_frozen_byte_conventions(self) -> None:
        self.assertEqual(decode_bf16(bytes((0x80, 0x3F))), [1.0])
        self.assertEqual(decode_bf16(bytes((0x20, 0xC0))), [-2.5])
        self.assertEqual(decode_e4m3fn(0x38), 1.0)
        self.assertEqual(decode_e4m3fn(0x7E), 448.0)
        self.assertEqual(decode_e2m1(0x07), 6.0)
        self.assertEqual(decode_e2m1(0x0F), -6.0)

    def test_wrong_source_identity_and_contract_are_rejected(self) -> None:
        document = json.loads(PLAN.read_text(encoding="utf-8"))
        with self.assertRaises(InvalidPlanError):
            self.load(document, lock_sha="0" * 64)
        wrong = copy.deepcopy(document)
        wrong["source_contract"] = "generic-assistant"
        with self.assertRaises(InvalidPlanError):
            self.load(wrong)

    def test_missing_kv_free_inventory_and_wrong_shape_are_rejected(self) -> None:
        document = json.loads(PLAN.read_text(encoding="utf-8"))
        sources = fake_sources()
        fabricated = replace(
            sources["model.layers.0.self_attn.q_proj.weight"],
            name="model.layers.0.self_attn.k_proj.weight",
        )
        sources[fabricated.name] = fabricated
        with self.assertRaises(InvalidPlanError):
            self.load(document, sources=sources)

        sources = fake_sources()
        name = "pre_projection.weight"
        sources[name] = replace(sources[name], shape=(1024, 2816), byte_length=5767168)
        with self.assertRaises(InvalidPlanError):
            self.load(document, sources=sources)

    def test_precision_substitution_and_mtp_omission_are_rejected(self) -> None:
        document = json.loads(PLAN.read_text(encoding="utf-8"))
        wrong_precision = copy.deepcopy(document)
        item = next(
            tensor for tensor in wrong_precision["tensors"]
            if tensor["output_name"] == "pre_projection.weight"
        )
        item["role"] = "assistant_attention_q_projection"
        with self.assertRaises(InvalidPlanError):
            self.load(wrong_precision)

        wrong_omission = copy.deepcopy(document)
        wrong_omission["omitted_families"].append("mtp")
        with self.assertRaises(InvalidPlanError):
            self.load(wrong_omission)


if __name__ == "__main__":
    unittest.main()
