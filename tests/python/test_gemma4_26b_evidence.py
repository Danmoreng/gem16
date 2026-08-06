from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "benchmarks/corpora/gemma4_26b"
GOLDENS = ROOT / "benchmarks/goldens/gemma4_26b"
MODELS = ROOT / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Gemma426BEvidenceTest(unittest.TestCase):
    def test_frozen_corpus_files_match_lock_and_are_disjoint(self) -> None:
        lock = json.loads((CORPUS / "splits.lock.json").read_text())
        self.assertEqual(lock["status"], "frozen")
        self.assertEqual(lock["split_audit"]["status"], "pass")
        self.assertEqual(lock["split_audit"]["overlaps"], [])
        documents: set[str] = set()
        token_spans: set[str] = set()
        for entry in lock["files"]:
            path = CORPUS / entry["path"]
            self.assertEqual(path.stat().st_size, entry["bytes"])
            self.assertEqual(sha256(path), entry["sha256"])
            split = json.loads(path.read_text())
            for record in split["records"]:
                self.assertNotIn(record["expanded_user_sha256"], documents)
                self.assertNotIn(record["input_token_ids_sha256_u32le"], token_spans)
                documents.add(record["expanded_user_sha256"])
                token_spans.add(record["input_token_ids_sha256_u32le"])

    def test_source_audit_is_linked_and_passes(self) -> None:
        audit = json.loads((GOLDENS / "source-audit.json").read_text())
        self.assertEqual(audit["status"], "pass")
        self.assertTrue(all(audit["inventory_lock_identity"].values()))
        self.assertTrue(audit["architecture"]["safetensors_text_config_exact"])
        self.assertTrue(audit["architecture"]["official_q4_0_metadata_exact"])
        self.assertTrue(audit["qat_vs_ordinary_bf16_tensor_contract"]["exact"])
        self.assertTrue(audit["tokenizer"]["qat_ordinary_unsloth_exact"])
        self.assertEqual(audit["tokenizer"]["official_q4_0"]["status"], "pass")
        for relative, expected in audit["input_sha256"].items():
            self.assertEqual(sha256(ROOT / relative), expected)

    def test_qat_bf16_golden_has_required_boundaries_and_hashes(self) -> None:
        directory = GOLDENS / "qat-bf16-selected"
        golden = json.loads((directory / "qat-bf16-selected.json").read_text())
        repeat = json.loads((directory / "repeat-report.json").read_text())
        self.assertEqual(
            golden["checkpoint"]["lock_sha256"],
            sha256(MODELS / "gemma4-26b-qat-bf16.lock.json"),
        )
        self.assertEqual(
            golden["software"]["lock_sha256"],
            sha256(MODELS / "gemma4-26b-reference-sources.lock.json"),
        )
        self.assertEqual(golden["checkpoint"]["source_dtype"], "BF16")
        self.assertTrue(golden["software"]["cuda_device"])
        self.assertEqual(golden["prompt"]["selected_layers"], [0, 5, 6, 29])
        self.assertTrue(golden["prompt"]["input_token_ids"])
        self.assertFalse(golden["execution"]["performance_eligible"])
        logits = directory / golden["final_logits"]["path"]
        self.assertEqual(logits.stat().st_size, golden["final_logits"]["bytes"])
        self.assertEqual(sha256(logits), golden["final_logits"]["sha256"])
        for layer in (0, 5, 6, 29):
            probabilities = golden["captures"][f"layer_{layer}.router_probabilities"]
            top_ids = golden["captures"][f"layer_{layer}.router_top_ids"]
            top_weights = golden["captures"][f"layer_{layer}.router_top_weights"]
            self.assertTrue(all(len(row["values_f32"]) == 128 for row in probabilities["rows"]))
            self.assertTrue(all(len(row["values_i64"]) == 8 for row in top_ids["rows"]))
            self.assertTrue(all(len(row["values_f32"]) == 8 for row in top_weights["rows"]))
        self.assertEqual(repeat["status"], "pass")
        self.assertIn("execution", repeat["compared_fields"])
        self.assertEqual(repeat["capture_count"], len(golden["captures"]))
        self.assertEqual(repeat["final_logits_sha256"], golden["final_logits"]["sha256"])

    def test_unsloth_reference_retains_token_repeat_and_logprob_limitation(self) -> None:
        reference = json.loads(
            (GOLDENS / "unsloth-nvfp4-reference.json").read_text()
        )
        self.assertEqual(
            reference["model"]["source_lock_sha256"],
            sha256(MODELS / "gemma4-26b-unsloth-nvfp4.lock.json"),
        )
        self.assertEqual(
            reference["software"]["lock_sha256"],
            sha256(MODELS / "gemma4-26b-reference-sources.lock.json"),
        )
        self.assertEqual(
            reference["prompt"]["corpus_sha256"],
            sha256(CORPUS / "calibration.json"),
        )
        self.assertEqual(reference["status"], "diagnostic_reference_token_deterministic")
        self.assertFalse(reference["benchmark_qualified"])
        self.assertFalse(reference["execution"]["performance_eligible"])
        self.assertEqual(reference["execution"]["warmup_runs"], 1)
        self.assertEqual(reference["execution"]["retained_runs"], 2)
        self.assertTrue(reference["execution"]["chunked_prefill"])
        self.assertEqual(reference["warmup_run"]["token_ids"], [7676, 236761])
        self.assertEqual(
            reference["repeat"],
            {
                "token_ids_exact": True,
                "text_exact": True,
                "logprobs_exact": False,
            },
        )
        self.assertEqual(
            [run["token_ids"] for run in reference["runs"]],
            [[7676, 236761], [7676, 236761]],
        )

    def test_q4_reference_has_source_software_device_and_repeat_identity(self) -> None:
        reference = json.loads((GOLDENS / "q4_0-reference.json").read_text())
        model = reference["model"]
        runtime = reference["runtime"]
        execution = reference["execution"]
        self.assertEqual(
            model["source_lock_sha256"],
            sha256(MODELS / "gemma4-26b-qat-q4_0.lock.json"),
        )
        self.assertEqual(
            model["inventory_sha256"],
            sha256(GOLDENS / "source-inventories/google-q4_0.json"),
        )
        self.assertEqual(
            runtime["software_lock_sha256"],
            sha256(MODELS / "gemma4-26b-reference-sources.lock.json"),
        )
        self.assertEqual(model["checkpoint_storage_types"], ["F32", "Q4_0", "Q6_K"])
        self.assertTrue(execution["device"])
        self.assertFalse(execution["trust_remote_code"])
        self.assertFalse(execution["performance_eligible"])
        self.assertTrue(reference["prompt"]["token_ids"])
        generation = reference["deterministic_generation"]
        self.assertTrue(generation["exact_repeat"])
        self.assertEqual(generation["run1_token_ids"], generation["run2_token_ids"])


if __name__ == "__main__":
    unittest.main()
