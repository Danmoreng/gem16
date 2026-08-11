from __future__ import annotations

import gc
import hashlib
import json
import mmap
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools.compare_manifests import build_reference_manifest
from tools.gem16_compile.common import (
    DataError,
    InvalidPlanError,
    ReproducibilityError,
    SourceVerificationError,
    canonical_json_bytes,
    peak_rss_bytes,
)
from tools.gem16_compile.compiler import (
    CompilerIdentity,
    CompilerRequest,
    compare_reproducibility,
    compile_artifact,
    plan_artifact,
    verify_artifact,
)
from tools.gem16_compile.encoders import CopyEncoder
from tools.generate_gemma4_26b_compiler_fixture import (
    git_blob_oid,
    safetensors_bytes,
    source_lock,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/gemma4_26b_compiler"
SOURCE = FIXTURE / "source"
LOCK = FIXTURE / "source.lock.json"
PLAN = FIXTURE / "compiler-plan.json"
EXPECTED = FIXTURE / "expected-artifact"
DEPENDENCIES = ROOT / "tools/gem16_compile/dependencies.lock.json"
PLAN_SCHEMA = ROOT / "tools/gem16_compile/schemas/compiler-plan.schema.json"
COMPILATION_SCHEMA = (
    ROOT / "tools/gem16_compile/schemas/gem16-compilation.schema.json"
)
CLI = ROOT / "tools/compile_gemma4_26b.py"
GENERATOR = ROOT / "tools/generate_gemma4_26b_compiler_fixture.py"


class FailingEncoder(CopyEncoder):
    def compile_tensor(self, plan, sources, output, workspace):
        raise DataError("injected encoder interruption")


def fixed_identity() -> CompilerIdentity:
    return CompilerIdentity(
        repository="Danmoreng/gem16",
        commit="0" * 40,
        dirty=False,
        environment={
            "system": "Linux",
            "machine": "x86_64",
            "python_implementation": "CPython",
            "python_version": "3.14.6",
            "python_major_minor": "3.14",
            "byteorder": "little",
            "locale": "C.UTF-8",
        },
    )


def request(
    source: Path = SOURCE,
    lock: Path = LOCK,
    plan: Path = PLAN,
    *,
    cap: int = 512 * 1024 * 1024,
    staging: int = 64 * 1024,
) -> CompilerRequest:
    return CompilerRequest(
        source_lock=lock,
        source_directory=source,
        compiler_manifest=plan,
        profile="synthetic-copy-v1",
        head_format="source",
        host_memory_cap_bytes=cap,
        staging_bytes=staging,
        dependencies_lock=DEPENDENCIES,
    )


def copy_fixture(directory: Path) -> tuple[Path, Path, Path]:
    source = directory / "source"
    shutil.copytree(SOURCE, source)
    lock = directory / "source.lock.json"
    plan = directory / "compiler-plan.json"
    shutil.copy2(LOCK, lock)
    shutil.copy2(PLAN, plan)
    return source, lock, plan


def make_large_fixture(directory: Path, payload_bytes: int) -> tuple[Path, Path, Path]:
    source = directory / "large-source"
    source.mkdir()
    text_payload = bytes((index * 29 + 7) % 256 for index in range(payload_bytes))
    vision_payload = bytes(range(16))
    shard = safetensors_bytes(
        {
            "model.language_model.embed_tokens.weight": (
                "BF16",
                [payload_bytes // 2],
                text_payload,
            ),
            "model.vision_tower.patch_embedder.input_proj.weight": (
                "BF16",
                [8],
                vision_payload,
            ),
        }
    )
    files = {
        "model.safetensors": shard,
        "config.json": canonical_json_bytes({"fixture": "large-streaming"}),
    }
    for name, payload in files.items():
        (source / name).write_bytes(payload)
    lock_payload = source_lock(files)
    lock = directory / "large.lock.json"
    lock.write_bytes(lock_payload)
    plan_document = {
        "schema_version": 1,
        "artifact_profile": "synthetic-copy-v1",
        "head_format": "source",
        "source_contract": "gemma4_26b_m04_large_streaming_v1",
        "source_lock_sha256": hashlib.sha256(lock_payload).hexdigest(),
        "target_shard_bytes": payload_bytes // 2,
        "approved_metadata_files": ["config.json"],
        "omitted_families": ["audio", "mtp", "video", "vision"],
        "tensors": [
            {
                "output_name": "model.language_model.embed_tokens.weight",
                "operation_id": "copy:model.language_model.embed_tokens.weight",
                "source_names": ["model.language_model.embed_tokens.weight"],
                "encoder": "copy-v1",
                "transformation": "identity-copy",
                "transformation_version": 1,
                "output_dtype": "BF16",
                "physical_shape": [payload_bytes // 2],
                "logical_dtype": "BF16",
                "logical_shape": [payload_bytes // 2],
                "axis_transformation": "identity",
                "quantizer_parameters": {},
                "dequantization_equation": "output = source",
                "role": "tied_embedding_and_output",
                "residency_class": "compiler_scaffold_text",
                "disk_layout": "source_bf16",
                "runtime_layout": "not_runtime_loadable",
                "aliased": True,
            }
        ],
        "excluded_tensors": [
            {
                "source_name": "model.vision_tower.patch_embedder.input_proj.weight",
                "family": "vision",
                "role": "vision_projection",
                "residency_class": "compile_excluded_vision",
                "reason": "large streaming test",
            }
        ],
        "reference_environment": {
            "system": "Linux",
            "machine": "x86_64",
            "python_implementation": "CPython",
            "python_major_minor": "3.14",
            "byteorder": "little",
            "locale": "C.UTF-8",
        },
    }
    plan = directory / "large-plan.json"
    plan.write_bytes(canonical_json_bytes(plan_document))
    del text_payload, vision_payload, shard, files, lock_payload, plan_document
    gc.collect()
    return source, lock, plan


class Gemma426BCheckpointCompilerTest(unittest.TestCase):
    def test_compiler_is_not_referenced_by_runtime_or_build_targets(self) -> None:
        search = subprocess.run(
            [
                "git",
                "grep",
                "-n",
                "compile_gemma4_26b.py\\|tools/gem16_compile",
                "--",
                "CMakeLists.txt",
                "include",
                "src",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(search.returncode, 1, search.stdout)

    def test_generated_fixture_is_current_and_plan_is_complete(self) -> None:
        subprocess.run(
            [sys.executable, str(GENERATOR), "--check"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = plan_artifact(request(), identity=fixed_identity())
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["source"]["tensor_count"], 4)
        self.assertEqual(report["output_tensor_count"], 3)
        self.assertEqual(report["output_tensor_bytes"], 176)
        self.assertEqual(report["excluded_tensor_count"], 1)
        self.assertEqual(report["excluded_tensor_bytes"], 16)
        self.assertEqual(
            [shard["payload_bytes"] for shard in report["projected_shards"]],
            [128, 48],
        )

    def test_two_runs_are_byte_identical_and_match_checked_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left"
            right = root / "right"
            left_report = compile_artifact(request(), left, identity=fixed_identity())
            right_report = compile_artifact(request(), right, identity=fixed_identity())
            self.assertEqual(left_report["output_tensor_count"], 3)
            comparison = compare_reproducibility(left, right)
            self.assertEqual(comparison["status"], "pass")
            fixture_comparison = compare_reproducibility(left, EXPECTED)
            self.assertEqual(fixture_comparison["status"], "pass")
            verification = verify_artifact(
                request(), left, identity=fixed_identity()
            )
            self.assertEqual(verification["status"], "pass")

    def test_manifest_traces_every_output_and_exact_vision_exclusion(self) -> None:
        plan_schema = json.loads(PLAN_SCHEMA.read_text(encoding="utf-8"))
        compilation_schema = json.loads(
            COMPILATION_SCHEMA.read_text(encoding="utf-8")
        )
        plan_document = json.loads(PLAN.read_text(encoding="utf-8"))
        document = json.loads(
            (EXPECTED / "gem16_compilation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(plan_document), set(plan_schema["required"]))
        self.assertEqual(set(document), set(compilation_schema["required"]))
        self.assertFalse(plan_schema["additionalProperties"])
        self.assertFalse(compilation_schema["additionalProperties"])

        independent = build_reference_manifest(EXPECTED)
        self.assertEqual(len(independent), 3)
        self.assertEqual(
            sum(tensor["byte_length"] for tensor in independent.values()), 176
        )
        self.assertFalse(
            document["quantization"]["production_quantization_implemented"]
        )
        self.assertEqual(document["byte_totals"]["source_tensor_count"], 4)
        self.assertEqual(document["byte_totals"]["output_tensor_count"], 3)
        self.assertEqual(document["byte_totals"]["output_tensor_bytes"], 176)
        self.assertEqual(document["byte_totals"]["excluded_tensor_count"], 1)
        self.assertEqual(document["byte_totals"]["excluded_tensor_bytes"], 16)
        vision = next(
            item
            for item in document["omitted_tensor_groups"]
            if item["group"] == "vision"
        )
        self.assertEqual(vision["source_tensor_count"], 1)
        self.assertEqual(vision["source_bytes"], 16)
        self.assertTrue(
            all(
                source["sha256"]
                for tensor in document["tensors"]
                for source in tensor["sources"]
            )
        )
        self.assertFalse(
            any("vision" in tensor["output_name"] for tensor in document["tensors"])
        )

    def test_corrupt_source_and_wrong_lock_fail_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, lock, plan = copy_fixture(root)
            shard = source / "model-00001-of-00002.safetensors"
            payload = bytearray(shard.read_bytes())
            payload[-1] ^= 0x01
            shard.write_bytes(payload)
            output = root / "corrupt-output"
            with self.assertRaisesRegex(SourceVerificationError, "SHA-256 mismatch"):
                compile_artifact(
                    request(source, lock, plan), output, identity=fixed_identity()
                )
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".incomplete").exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, lock, plan = copy_fixture(root)
            lock_document = json.loads(lock.read_text(encoding="utf-8"))
            lock_document["resolved_at_utc"] = "2026-08-12T00:00:00Z"
            lock.write_bytes(canonical_json_bytes(lock_document))
            with self.assertRaisesRegex(InvalidPlanError, "another source lock"):
                compile_artifact(
                    request(source, lock, plan),
                    root / "wrong-lock-output",
                    identity=fixed_identity(),
                )

    def test_changed_plan_and_corrupt_output_fail_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact"
            compile_artifact(request(), artifact, identity=fixed_identity())
            changed_plan = root / "changed-plan.json"
            plan_document = json.loads(PLAN.read_text(encoding="utf-8"))
            plan_document["tensors"][0]["role"] = "changed_role"
            changed_plan.write_bytes(canonical_json_bytes(plan_document))
            with self.assertRaisesRegex(DataError, "plan provenance mismatch"):
                verify_artifact(
                    request(plan=changed_plan), artifact, identity=fixed_identity()
                )

            shard = artifact / "model-00001-of-00002.safetensors"
            payload = bytearray(shard.read_bytes())
            payload[-1] ^= 0x80
            shard.write_bytes(payload)
            with self.assertRaisesRegex(DataError, "file hash/size mismatch"):
                verify_artifact(request(), artifact, identity=fixed_identity())

    def test_malformed_verified_header_and_incomplete_coverage_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, lock, plan = copy_fixture(root)
            shard_name = "model-00001-of-00002.safetensors"
            shard = source / shard_name
            payload = bytearray(shard.read_bytes())
            payload[:8] = (len(payload) + 1).to_bytes(8, "little")
            shard.write_bytes(payload)
            lock_document = json.loads(lock.read_text(encoding="utf-8"))
            entry = next(
                item for item in lock_document["files"] if item["path"] == shard_name
            )
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            entry["git_oid"] = git_blob_oid(payload)
            lock_payload = canonical_json_bytes(lock_document)
            lock.write_bytes(lock_payload)
            plan_document = json.loads(plan.read_text(encoding="utf-8"))
            plan_document["source_lock_sha256"] = hashlib.sha256(lock_payload).hexdigest()
            plan.write_bytes(canonical_json_bytes(plan_document))
            with self.assertRaisesRegex(
                SourceVerificationError, "invalid Safetensors header length"
            ):
                plan_artifact(request(source, lock, plan), identity=fixed_identity())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete_plan = root / "incomplete-plan.json"
            document = json.loads(PLAN.read_text(encoding="utf-8"))
            document["excluded_tensors"] = []
            incomplete_plan.write_bytes(canonical_json_bytes(document))
            with self.assertRaisesRegex(InvalidPlanError, "coverage mismatch"):
                plan_artifact(
                    request(plan=incomplete_plan), identity=fixed_identity()
                )

    def test_interruption_and_existing_output_never_publish_partial_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifact"
            with self.assertRaisesRegex(DataError, "injected encoder interruption"):
                compile_artifact(
                    request(),
                    output,
                    identity=fixed_identity(),
                    encoders={"copy-v1": FailingEncoder()},
                )
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".incomplete").exists())

            output.mkdir()
            marker = output / "keep"
            marker.write_text("unchanged", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "already exists"):
                compile_artifact(request(), output, identity=fixed_identity())
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")

    def test_bounded_window_compiles_tensor_larger_than_staging_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, lock, plan = make_large_fixture(root, 2 * 1024 * 1024)
            cap = peak_rss_bytes() + 16 * 1024 * 1024
            large_request = request(
                source,
                lock,
                plan,
                cap=cap,
                staging=4096,
            )
            report = compile_artifact(
                large_request, root / "artifact", identity=fixed_identity()
            )
            memory = report["memory"]
            self.assertEqual(memory["staging_buffer_bytes"], 4096)
            self.assertEqual(memory["maximum_tensor_bytes"], 2 * 1024 * 1024)
            self.assertGreater(
                memory["maximum_tensor_bytes"], memory["staging_buffer_bytes"]
            )
            self.assertLessEqual(
                memory["maximum_mapped_window_bytes"],
                memory["staging_buffer_bytes"] + mmap.ALLOCATIONGRANULARITY,
            )
            self.assertLessEqual(memory["peak_rss_bytes"], cap)

    def test_cli_help_restart_policy_and_reproducibility_exit_code(self) -> None:
        help_result = subprocess.run(
            [sys.executable, str(CLI), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0)
        for action in ("plan", "compile", "verify", "compare-reproducibility"):
            self.assertIn(action, help_result.stdout)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = [
                sys.executable,
                str(CLI),
                "compile",
                "--source-lock",
                str(LOCK),
                "--source-directory",
                str(SOURCE),
                "--compiler-manifest",
                str(PLAN),
                "--profile",
                "synthetic-copy-v1",
                "--head-format",
                "source",
                "--max-host-memory",
                str(512 * 1024 * 1024),
                "--staging-bytes",
                str(64 * 1024),
                "--output",
                str(root / "artifact"),
                "--report",
                str(root / "report.json"),
                "--resume",
            ]
            resume = subprocess.run(
                command, cwd=ROOT, check=False, capture_output=True, text=True
            )
            self.assertEqual(resume.returncode, 2)
            self.assertIn("restart", resume.stderr)

            left = root / "left"
            right = root / "right"
            shutil.copytree(EXPECTED, left)
            shutil.copytree(EXPECTED, right)
            (right / "config.json").write_text("changed\n", encoding="utf-8")
            compare = subprocess.run(
                [
                    sys.executable,
                    str(CLI),
                    "compare-reproducibility",
                    "--left",
                    str(left),
                    "--right",
                    str(right),
                    "--report",
                    str(root / "compare.json"),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(compare.returncode, 6)
            self.assertIn("config.json", compare.stderr)

    def test_artifact_symlink_is_rejected_even_when_target_bytes_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact"
            compile_artifact(request(), artifact, identity=fixed_identity())
            external = root / "external-config.json"
            shutil.copy2(artifact / "config.json", external)
            (artifact / "config.json").unlink()
            try:
                (artifact / "config.json").symlink_to(external)
            except OSError as error:
                if sys.platform == "win32" and getattr(error, "winerror", None) == 1314:
                    self.skipTest("Windows runner lacks symbolic-link privilege")
                raise
            with self.assertRaisesRegex(DataError, "contains a symlink"):
                verify_artifact(request(), artifact, identity=fixed_identity())

    def test_unsafe_metadata_path_and_dirty_release_identity_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_plan = root / "bad-plan.json"
            document = json.loads(PLAN.read_text(encoding="utf-8"))
            document["approved_metadata_files"].append("../escape.json")
            bad_plan.write_bytes(canonical_json_bytes(document))
            with self.assertRaisesRegex(InvalidPlanError, "unsafe"):
                plan_artifact(request(plan=bad_plan), identity=fixed_identity())

            clean = fixed_identity()
            dirty = CompilerIdentity(
                repository=clean.repository,
                commit=clean.commit,
                dirty=True,
                environment=clean.environment,
            )
            with self.assertRaisesRegex(InvalidPlanError, "clean repository"):
                compile_artifact(
                    request(), root / "dirty-artifact", identity=dirty
                )


if __name__ == "__main__":
    unittest.main()
