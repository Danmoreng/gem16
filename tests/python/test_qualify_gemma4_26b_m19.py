import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "qualify_gemma4_26b_m19.py"
)
SPEC = importlib.util.spec_from_file_location("qualify_gemma4_26b_m19", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
m19 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m19
SPEC.loader.exec_module(m19)


def write(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.content_hash = "a" * 64
        self.source_hash = "b" * 64
        self.q4_gguf_hash = "7" * 64
        self.runtime_profile = "native-test"
        self.artifact_profile = "artifact-test"
        self.corpus = write(
            root / "test.json",
            {
                "schema_version": 1,
                "split": "test",
                "policy": "held_out_final_quality_no_quantizer_tuning",
                "records": [
                    {
                        "id": "test-code",
                        "category": "code",
                        "input_token_ids_sha256_u32le": "c" * 64,
                        "selected_positions": [0, 7],
                        "selected_layers": [0, 2],
                    }
                ],
            },
        )
        self.corpus_lock = write(
            root / "splits.lock.json",
            {
                "schema_version": 1,
                "status": "frozen",
                "split_audit": {
                    "status": "pass",
                    "document_or_token_span_overlap_count": 0,
                },
            },
        )
        locked = {
            "config.json": "d" * 64,
            "generation_config.json": "e" * 64,
            "tokenizer.json": "f" * 64,
            "tokenizer_config.json": "1" * 64,
            "chat_template.jinja": "2" * 64,
            "gem16_compilation.json": "3" * 64,
        }
        self.artifact_lock = write(
            root / "artifact.lock.json",
            {
                "schema_version": 1,
                "artifact_content_sha256": self.content_hash,
                "artifact_profile": self.artifact_profile,
                "source_lock_sha256": self.source_hash,
                "files": [
                    {"path": name, "sha256": value, "size": 1}
                    for name, value in locked.items()
                ],
            },
        )
        self.q4_lock = write(
            root / "q4.lock.json",
            {
                "schema_version": 2,
                "revision": "d1c082be9cf3c8a514acf63b8761f4b41935842e",
                "files": [
                    {
                        "path": "gemma-4-26B_q4_0-it.gguf",
                        "sha256": self.q4_gguf_hash,
                    }
                ],
            },
        )
        self.acceptance = write(
            root / "acceptance.json",
            {
                "status": "accepted",
                "profile": self.runtime_profile,
                "qualified_artifact_content_sha256": self.content_hash,
            },
        )
        self.closure = write(
            root / "closure.json",
            {
                "status": "accepted_closure_hardening",
                "profile": self.runtime_profile,
                "qualified_artifact_content_sha256": self.content_hash,
            },
        )
        self.thresholds = write(
            root / "thresholds.json",
            {
                "schema_version": 1,
                "milestone": "M19",
                "status": "frozen_before_q4_heldout_execution",
                "identity": {
                    "artifact_content_sha256": self.content_hash,
                    "artifact_lock_sha256": digest(self.artifact_lock),
                    "artifact_profile": self.artifact_profile,
                    "runtime_profile": self.runtime_profile,
                    "source_lock_sha256": self.source_hash,
                    "reference_q4_lock_sha256": digest(self.q4_lock),
                    "reference_q4_gguf_sha256": self.q4_gguf_hash,
                    "reference_llama_cpp_revision": "llama-test",
                    "config_sha256": locked["config.json"],
                    "generation_config_sha256": locked["generation_config.json"],
                    "tokenizer_sha256": locked["tokenizer.json"],
                    "tokenizer_config_sha256": locked["tokenizer_config.json"],
                    "chat_template_sha256": locked["chat_template.jinja"],
                    "compilation_manifest_sha256": locked[
                        "gem16_compilation.json"
                    ],
                    "m17_acceptance_sha256": digest(self.acceptance),
                    "m17_closure_sha256": digest(self.closure),
                },
                "heldout": {
                    "split": "test",
                    "policy": "held_out_final_quality_no_quantizer_tuning",
                    "corpus_sha256": digest(self.corpus),
                    "split_lock_sha256": digest(self.corpus_lock),
                    "required_categories": ["code"],
                    "required_record_count": 1,
                },
                "evaluator": {
                    "sgl_eval_commit": "eval-commit",
                    "required_task_benchmarks": ["gsm8k", "gpqa", "aime26"],
                    "datasets": {
                        "gsm8k": {"examples": 1, "sha256": "8" * 64},
                        "gpqa": {"examples": 1, "sha256": "9" * 64},
                        "aime26": {"examples": 1, "source": "bundled"},
                    },
                },
                "thresholds": {
                    "teacher_forced_q4_reference_token_candidate_top5_fraction_min": 0.95,
                    "teacher_forced_q4_top1_agreement_fraction_min": 0.8,
                    "teacher_forced_selected_logprob_compared_fraction_min": 0.95,
                    "teacher_forced_mean_selected_logprob_absolute_delta_max": 0.5,
                    "teacher_forced_record_selected_logprob_absolute_delta_max": 2.0,
                    "task_absolute_delta_min": -0.02,
                    "task_relative_retention_min": 0.97,
                    "task_invalid_response_fraction_max": 0.01,
                    "prose_relative_retention_min": 0.9,
                    "prose_per_category_score_delta_min": -1.0,
                    "prose_invalid_response_count_max": 0,
                    "prose_minimum_independent_reviewers": 2,
                },
            },
        )

        captures = []
        for position in (0, 7):
            for layer in (0, 2):
                captures.append(
                    {
                        "position": position,
                        "layer": layer,
                        "relative_l2": 0.1,
                        "cosine": 0.99,
                        "router_top8_overlap": 7,
                        "attention_relative_l2": 0.1,
                        "attention_cosine": 0.99,
                    }
                )
        self.numerical = write(
            root / "numerical.json",
            {
                "schema_version": 1,
                "kind": "gemma4_26b_m19_numerical",
                "status": "complete",
                "artifact_content_sha256": self.content_hash,
                "runtime_profile": self.runtime_profile,
                "corpus_sha256": digest(self.corpus),
                "reference": {
                    "kind": "official_google_qat_q4_0",
                    "q4_lock_sha256": digest(self.q4_lock),
                    "gguf_sha256": self.q4_gguf_hash,
                    "llama_cpp_revision": "llama-test",
                },
                "records": [
                    {
                        "id": "test-code",
                        "category": "code",
                        "input_token_ids_sha256_u32le": "c" * 64,
                        "target_token_ids_sha256_u32le": "4" * 64,
                        "target_token_source": "official_q4_greedy_seed_0",
                        "q4_reference_capture_sha256": "5" * 64,
                        "candidate_capture_sha256": "6" * 64,
                        "all_logits_finite": True,
                        "teacher_forced": {
                            "scored_token_count": 10,
                            "q4_reference_token_candidate_top5_fraction": 1.0,
                            "top1_agreement_fraction": 0.9,
                            "selected_logprob_compared_fraction": 1.0,
                            "mean_selected_logprob_absolute_delta": 0.1,
                            "maximum_selected_logprob_absolute_delta": 0.2,
                        },
                    }
                ],
            },
        )
        pairs = []
        for benchmark in ("gsm8k", "gpqa", "aime26"):
            reasoning = "none" if benchmark == "gsm8k" else "high"
            base = {
                "schema_version": 1,
                "status": "complete",
                "benchmark": benchmark,
                "benchmark_source": {
                    "sgl_eval": {"commit": "eval-commit"},
                    "dataset": {
                        "sha256": (
                            "8" * 64 if benchmark == "gsm8k" else "9" * 64
                        )
                    },
                },
                "completed_examples": 1,
                "protocol": {
                    "generation": "checkpoint",
                    "reasoning": reasoning,
                    "repeats": 1,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 64,
                    "seed": 0,
                    "max_tokens": 512 if benchmark == "gsm8k" else 16384,
                },
                "examples": [
                    {
                        "id": "q1",
                        "samples": [
                            {
                                "score": 1,
                                "extracted": "answer",
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ],
            }
            reference_document = {**base, "endpoint": {"backend": "openai"}}
            candidate_document = {**base, "endpoint": {"backend": "gem16"}}
            reference = write(root / f"{benchmark}-reference.json", reference_document)
            candidate = write(root / f"{benchmark}-candidate.json", candidate_document)
            pairs.append(
                {
                    "benchmark": benchmark,
                    "reference_path": reference.name,
                    "reference_sha256": digest(reference),
                    "candidate_path": candidate.name,
                    "candidate_sha256": digest(candidate),
                    "reference_q4_lock_sha256": digest(self.q4_lock),
                    "candidate_artifact_content_sha256": self.content_hash,
                }
            )
        self.task_manifest = write(
            root / "tasks.json",
            {
                "schema_version": 1,
                "kind": "gemma4_26b_m19_task_pairs",
                "status": "complete",
                "artifact_content_sha256": self.content_hash,
                "reference_q4_lock_sha256": digest(self.q4_lock),
                "sgl_eval_commit": "eval-commit",
                "pairs": pairs,
            },
        )
        self.prose = write(
            root / "prose.json",
            {
                "schema_version": 1,
                "kind": "gemma4_26b_m19_blind_prose_review",
                "status": "complete",
                "artifact_content_sha256": self.content_hash,
                "corpus_sha256": digest(self.corpus),
                "independent_reviewer_count": 2,
                "records": [
                    {
                        "id": "test-code",
                        "category": "code",
                        "input_token_ids_sha256_u32le": "c" * 64,
                        "candidate_response_sha256": "5" * 64,
                        "q4_response_sha256": "6" * 64,
                        "rubric_sha256": "7" * 64,
                        "candidate_score": 4,
                        "q4_score": 4,
                        "maximum_score": 4,
                        "invalid_response": False,
                    }
                ],
            },
        )

    def args(self, **updates) -> argparse.Namespace:
        values = {
            "thresholds": self.thresholds,
            "corpus": self.corpus,
            "corpus_lock": self.corpus_lock,
            "artifact_lock": self.artifact_lock,
            "q4_lock": self.q4_lock,
            "m17_acceptance": self.acceptance,
            "m17_closure": self.closure,
            "numerical_report": self.numerical,
            "task_manifest": self.task_manifest,
            "prose_report": self.prose,
            "output": self.root / "summary.json",
        }
        values.update(updates)
        return argparse.Namespace(**values)


class QualifyGemma426BM19Test(unittest.TestCase):
    def test_complete_passing_evidence_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            code, result = m19.qualify(fixture.args())
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "qualification_pass")
            self.assertTrue(result["acceptance"])
            self.assertTrue(all(result["gates"].values()))

    def test_missing_run_evidence_is_an_explicit_blocker(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            code, result = m19.qualify(
                fixture.args(
                    numerical_report=None,
                    task_manifest=None,
                    prose_report=None,
                )
            )
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], "blocked_evidence_pending")
            self.assertEqual(len(result["blockers"]), 3)
            self.assertFalse(result["acceptance"])

    def test_category_regression_cannot_hide_in_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            numerical = json.loads(fixture.numerical.read_text(encoding="utf-8"))
            numerical["records"][0]["teacher_forced"][
                "mean_selected_logprob_absolute_delta"
            ] = 0.75
            write(fixture.numerical, numerical)
            code, result = m19.qualify(fixture.args())
            self.assertEqual(code, 1)
            self.assertIn("mean_selected_logprob_delta", result["failed_gates"])
            self.assertEqual(result["decision"], "diagnose_m18_triggered")

    def test_artifact_lock_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            artifact = json.loads(fixture.artifact_lock.read_text(encoding="utf-8"))
            artifact["artifact_profile"] = "changed"
            write(fixture.artifact_lock, artifact)
            with self.assertRaises(m19.QualificationError):
                m19.qualify(fixture.args())


if __name__ == "__main__":
    unittest.main()
