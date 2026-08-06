import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Gemma426BM02EvidenceTest(unittest.TestCase):
    def test_locked_fixture_and_capability_evidence_match(self) -> None:
        fixture = ROOT / "tests/fixtures/gemma4_26b_config.json"
        evidence = json.loads(
            (ROOT / "docs/evidence/gemma4_26b/m02-model-variant-capability.json").read_text(
                encoding="utf-8"
            )
        )
        config = json.loads(fixture.read_text(encoding="utf-8"))
        lock_path = ROOT / "models/gemma4-26b-qat-bf16.lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))

        self.assertEqual(
            hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            evidence["source"]["model_lock_sha256"],
        )
        locked_config = next(item for item in lock["files"] if item["path"] == "config.json")
        self.assertEqual(locked_config["size"], evidence["source"]["config_bytes"])
        self.assertEqual(locked_config["sha256"], evidence["source"]["config_sha256"])
        self.assertEqual(fixture.stat().st_size, evidence["source"]["config_bytes"])
        self.assertEqual(
            hashlib.sha256(fixture.read_bytes()).hexdigest(),
            evidence["source"]["config_sha256"],
        )
        classification = evidence["classification"]
        self.assertEqual(config["architectures"][0], classification["architecture"])
        self.assertEqual(config["model_type"], classification["model_type"])
        self.assertEqual(config["text_config"]["model_type"], classification["text_model_type"])
        self.assertEqual(classification["model_variant"], "gemma4_moe_26b_a4b")
        self.assertTrue(classification["inspectable"])
        self.assertFalse(classification["runtime_supported"])
        self.assertFalse(classification["tensor_contract_validated"])

        text = config["text_config"]
        expected_dimensions = {
            "layer_count": text["num_hidden_layers"],
            "hidden_size": text["hidden_size"],
            "shared_intermediate_size": text["intermediate_size"],
            "moe_intermediate_size": text["moe_intermediate_size"],
            "expert_count": text["num_experts"],
            "top_k_experts": text["top_k_experts"],
            "query_heads": text["num_attention_heads"],
            "local_kv_heads": text["num_key_value_heads"],
            "global_kv_heads": text["num_global_key_value_heads"],
            "shared_kv_layer_count": text["num_kv_shared_layers"],
            "local_head_dimension": text["head_dim"],
            "global_head_dimension": text["global_head_dim"],
            "sliding_window": text["sliding_window"],
            "max_positions": text["max_position_embeddings"],
            "vocabulary_size": text["vocab_size"],
        }
        self.assertEqual(evidence["dimensions"], expected_dimensions)
        self.assertEqual(
            evidence["capabilities"],
            {"text": True, "vision": False, "audio": False, "video": False, "mtp": False},
        )


if __name__ == "__main__":
    unittest.main()
