import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "qualify_gemma4_26b_m20.py"
SPEC = importlib.util.spec_from_file_location("qualify_gemma4_26b_m20", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
m20 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m20
SPEC.loader.exec_module(m20)


DIGEST = "a" * 64


def suite() -> dict:
    return {
        "schema_version": 1,
        "model": {
            "profile": m20.EXPECTED_PROFILE,
            "artifact_content_sha256": DIGEST,
            "artifact_lock_sha256": DIGEST,
            "source_lock_sha256": DIGEST,
        },
        "toolchain_lock_sha256": DIGEST,
        "native_instruction_evidence": {
            "observed": True,
            "binary_sha256": DIGEST,
            "disassembly_sha256": DIGEST,
            "required_mnemonics": ["MMA", "FP8"],
        },
        "scenarios": [
            {
                "id": m20.PROMOTION_SCENARIO,
                "prompt_tokens": m20.PROMOTION_PROMPT_TOKENS,
                "output_forwards": m20.PROMOTION_OUTPUT_FORWARDS,
                "context_tokens": m20.PROMOTION_CONTEXT_TOKENS,
                "prompt_manifest_sha256": m20.PROMOTION_PROMPT_SHA256,
                "prompt_manifest_path": "wikipedia-summary-16k.json",
                "sampling": {"mode": "greedy"},
                "kv_mode": "fp8",
            },
        ],
    }


def sample(scenario: dict) -> dict:
    return {
        "schema_version": 1,
        "status": "ok",
        "model": {
            "profile": m20.EXPECTED_PROFILE,
            "artifact_content_sha256": DIGEST,
            "artifact_lock_sha256": DIGEST,
            "source_lock_sha256": DIGEST,
        },
        "correctness": {
            "all_logits_finite": True,
            "finite_checks_completed": scenario["output_forwards"],
            "prompt_manifest_sha256": scenario["prompt_manifest_sha256"],
            "output_token_sha256": DIGEST,
            "output_checksum": 123,
        },
        "runtime_path": {
            "model_variant": m20.EXPECTED_VARIANT,
            "head_format": "nvfp4",
            "kv_mode": "fp8",
            "backend": "sm120",
            "prompt_cache": False,
            "cpu_weight_offload": False,
            "token_loop_allocations": False,
            "native_instruction_capability": True,
            "fallback_count": 0,
            "cuda_graph": {"enabled": True, "first_demotion_reason": "none"},
            "resolved_dispatch": {
                "attention_prefill": "native_fixed_sm120",
                "attention_decode": "native_fixed_sm120",
                "moe_decode": "native_sm120",
                "moe_prefill": "native_grouped_sm120",
                "embedding_head": "native_sm120",
            },
            "observations": {
                "prefill_calls": 1,
                "prefill_chunks": 1,
                "decode_graph_launches": scenario["output_forwards"] - 1,
                "token_selections": scenario["output_forwards"],
                "sliding_ring_wraps": 0,
                "maximum_global_position_exclusive":
                    scenario["prompt_tokens"] + scenario["output_forwards"] - 1,
                "recurring_allocation_count": 0,
            },
        },
        "performance": {
            "prompt_tokens": scenario["prompt_tokens"],
            "output_forwards": scenario["output_forwards"],
            "sampling": scenario["sampling"],
            "prompt_ms": 100.0,
            "ttft_ms": 105.0,
            "decode_ms": 630.0,
            "decode_tps": 100.0,
            "itl_ms": [10.0] * 63,
        },
        "memory": {
            "sampled_device_used_bytes": 15_500_000_000,
            "margin_bytes": 800_000_000,
            "recurring_allocation_observed": False,
        },
    }


class QualifyGemma426BM20Test(unittest.TestCase):
    def test_suite_requires_exact_bounded_promotion_row(self):
        document = suite()
        validated = m20.validate_suite(document)
        self.assertEqual(len(validated), 1)
        document["scenarios"][0]["prompt_tokens"] -= 1
        with self.assertRaisesRegex(m20.QualificationError, "prompt_tokens"):
            m20.validate_suite(document)

    def test_valid_sample_preserves_exact_timing_boundaries(self):
        document = suite()
        scenario = document["scenarios"][0]
        normalized = m20.validate_sample(sample(scenario), scenario, document)
        self.assertEqual(normalized["prompt_tps"], 163840.0)
        self.assertEqual(normalized["decode_tps"], 100.0)
        self.assertEqual(normalized["itl_ms"], [10.0] * 63)

    def test_prompt_manifest_is_content_addressed_and_counted(self):
        document = suite()
        scenario = document["scenarios"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / scenario["prompt_manifest_path"]
            path.write_text(
                __import__("json").dumps({
                    "schema_version": 1,
                    "token_ids": list(range(scenario["prompt_tokens"])),
                }),
                encoding="utf-8",
            )
            scenario["prompt_manifest_sha256"] = m20.sha256_file(path)
            bound = m20.bind_prompt_manifests([scenario], root)
            self.assertTrue(Path(bound[0]["prompt_manifest_path"]).samefile(path))
            scenario["prompt_manifest_sha256"] = DIGEST
            with self.assertRaisesRegex(m20.QualificationError, "hash mismatch"):
                m20.bind_prompt_manifests([scenario], root)

    def test_fallback_and_non_native_dispatch_fail_hard(self):
        document = suite()
        scenario = document["scenarios"][0]
        bad = sample(scenario)
        bad["runtime_path"]["fallback_count"] = 1
        with self.assertRaisesRegex(m20.QualificationError, "fallback"):
            m20.validate_sample(bad, scenario, document)
        bad = sample(scenario)
        bad["runtime_path"]["resolved_dispatch"]["moe_decode"] = "reference"
        with self.assertRaisesRegex(m20.QualificationError, "non-native"):
            m20.validate_sample(bad, scenario, document)

    def test_inconsistent_decode_boundary_fails(self):
        document = suite()
        scenario = document["scenarios"][0]
        bad = sample(scenario)
        bad["performance"]["decode_ms"] = 20.0
        with self.assertRaisesRegex(m20.QualificationError, "does not match ITLs"):
            m20.validate_sample(bad, scenario, document)

    def test_summary_requires_ten_retained_runs(self):
        document = suite()
        scenario = document["scenarios"][0]
        normalized = m20.validate_sample(sample(scenario), scenario, document)
        normalized["telemetry_process_peak_bytes"] = 15_490_000_000
        with self.assertRaisesRegex(m20.QualificationError, "exactly 10"):
            m20.summarize_runs([normalized] * 9)
        summary = m20.summarize_runs([normalized] * 10)
        self.assertEqual(summary["decode_tps"]["count"], 10)
        self.assertTrue(summary["deterministic_outputs"])
        self.assertEqual(summary["sampled_device_used_bytes"]["median"], 15_500_000_000.0)

    def test_instruction_evidence_is_bound_to_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "runner"
            executable.write_bytes(b"not-the-frozen-binary")
            with self.assertRaisesRegex(m20.QualificationError, "binary SHA-256"):
                m20.validate_instruction_evidence(
                    executable, suite()["native_instruction_evidence"]
                )

    def test_greedy_hidden_sampling_controls_are_rejected(self):
        document = suite()
        document["scenarios"][0]["sampling"]["seed"] = 0
        with self.assertRaisesRegex(m20.QualificationError, "hidden controls"):
            m20.validate_suite(document)

    def test_m21_gate_binds_exact_candidate_identity(self):
        document = suite()
        code = {"commit": "1" * 40, "dirty": False}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "m21.json"
            path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "milestone": "M21",
                    "status": "qualified",
                    "acceptance": True,
                    "exit_gate_pass": True,
                    "release_32k": True,
                    "base_64k_result": "passed",
                    "base_max_context": 65536,
                    "maximum_search_complete": True,
                    "code": code,
                    "candidate": {
                        "model": document["model"],
                        "toolchain_lock_sha256": document["toolchain_lock_sha256"],
                        "benchmark_binary_sha256": DIGEST,
                    },
                }),
                encoding="utf-8",
            )
            accepted = m20.m21_gate(path, document, code, DIGEST)
            self.assertTrue(accepted["pass"])
            rejected = m20.m21_gate(path, document, code, "b" * 64)
            self.assertFalse(rejected["pass"])
            self.assertFalse(rejected["checks"]["same_benchmark_binary"])

    def test_deferred_external_gate_still_reports_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "m19.json"
            path.write_text(
                json.dumps({"milestone": "M19", "status": "pending"}),
                encoding="utf-8",
            )
            report = m20.external_gate(path, "M19")
            self.assertTrue(report["available"])
            self.assertFalse(report["pass"])
            self.assertEqual(report["reported_milestone"], "M19")


if __name__ == "__main__":
    unittest.main()
