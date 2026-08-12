from __future__ import annotations

import gc
import hashlib
import json
import mmap
from pathlib import Path
import shutil
import struct
import subprocess
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

from tools.compare_manifests import build_reference_manifest
from tools.gem16_compile.common import (
    DataError,
    InvalidPlanError,
    OutputError,
    ReproducibilityError,
    SourceVerificationError,
    canonical_json_bytes,
    peak_rss_bytes,
)
from tools.gem16_compile.compiler import (
    CompilerIdentity,
    CompilerRequest,
    compare_reproducibility,
    _complete_statistics_record,
    compile_artifact,
    plan_artifact,
    verify_artifact,
)
from tools.gem16_compile.encoders import CopyEncoder
from tools.gem16_compile.profiles import (
    M05_DEQUANTIZATION_EQUATION,
    M05_QUANTIZER_PARAMETERS,
    M05_PROFILE,
    M05_SOURCE_CONTRACT,
    M05_VISION_EXCLUSION_REASON,
)
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


def rewrite_mutable_hashes_after_payload_tamper(artifact: Path) -> None:
    manifest_path = artifact / "gem16_compilation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensor = manifest["tensors"][0]
    shard = artifact / tensor["output_shard"]
    raw = bytearray(shard.read_bytes())
    header_size = struct.unpack_from("<Q", raw)[0]
    begin, _ = tensor["output_data_offsets"]
    raw[8 + header_size + begin] ^= 0x01
    shard.write_bytes(raw)
    tensor_begin = 8 + header_size + begin
    tensor_end = 8 + header_size + tensor["output_data_offsets"][1]
    tensor["sha256"] = hashlib.sha256(raw[tensor_begin:tensor_end]).hexdigest()
    for record in manifest["files"]:
        if record["path"] == shard.name:
            record["size"] = len(raw)
            record["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))


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
    def test_m06_debug_native_fails_before_source_load_or_output_staging(self) -> None:
        debug_native = ROOT / "build/Linux/host-debug/bin/gem16-checkpoint-compiler"
        if not debug_native.is_file():
            self.skipTest("host-debug native compiler is not built")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_m06 = CompilerRequest(
                source_lock=root / "missing.lock.json",
                source_directory=root / "missing-source",
                compiler_manifest=root / "missing-plan.json",
                profile="nvfp4-experts-partial-v1",
                head_format="deferred",
                host_memory_cap_bytes=512 * 1024 * 1024,
                staging_bytes=4096,
                dependencies_lock=DEPENDENCIES,
                native_encoder=debug_native,
                threads=4,
            )
            output = root / "artifact"
            with mock.patch(
                "tools.gem16_compile.compiler._load_request",
                side_effect=AssertionError("source loading must not run"),
            ):
                with self.assertRaisesRegex(InvalidPlanError, "Release build"):
                    compile_artifact(
                        request_m06, output, identity=fixed_identity(), allow_dirty=True
                    )
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".incomplete").exists())

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

    def test_tiny_m05_fixture_is_rejected_by_production_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            q_name = "model.language_model.layers.0.self_attn.q_proj.weight"
            k_name = "model.language_model.layers.0.self_attn.k_proj.weight"
            router_name = "model.language_model.layers.0.router.proj.weight"
            vision_name = "model.vision_tower.patch_embedder.input_proj.weight"
            q_payload = struct.pack("<8H", 0x3F80, 0xC000, 0x3F00, 0x8000,
                                    0x3F82, 0xBF82, 0x0000, 0x8000)
            k_payload = struct.pack("<8H", 0x4000, 0xC000, 0x3E80, 0x0000,
                                    0x3F80, 0xBF80, 0x0001, 0x8001)
            router_payload = bytes(16)
            vision_payload = bytes(8)
            source_files = {
                "model.safetensors": safetensors_bytes({
                    q_name: ("BF16", [2, 4], q_payload),
                    k_name: ("BF16", [2, 4], k_payload),
                    router_name: ("BF16", [2, 4], router_payload),
                    vision_name: ("BF16", [1, 4], vision_payload),
                })
            }
            (source_dir / "model.safetensors").write_bytes(source_files["model.safetensors"])
            lock_path = root / "source.lock.json"
            lock_bytes = source_lock(source_files)
            lock_path.write_bytes(lock_bytes)
            tensors = []
            for source_name, role_suffix in ((q_name, "q"), (k_name, "k")):
                stem = source_name.removesuffix(".weight")
                role = f"attention_{role_suffix}_projection"
                operation = f"fp8-attention:{stem}"
                for encoder, name, dtype, shape, transformation, disk in (
                    ("fp8-rowwise-weight-v1", source_name, "F8_E4M3", [2, 4],
                     "bf16-to-fp8-e4m3fn-rowwise-weight", "source_nk_fp8"),
                    ("fp8-rowwise-scale-v1", f"{stem}.weight_scale", "BF16", [2, 1],
                     "bf16-to-bf16-rowwise-scale", "row_bf16"),
                ):
                    tensors.append({
                        "output_name": name,
                        "operation_id": operation,
                        "source_names": [source_name],
                        "encoder": encoder,
                        "transformation": transformation,
                        "transformation_version": 1,
                        "output_dtype": dtype,
                        "physical_shape": shape,
                        "logical_dtype": "BF16",
                        "logical_shape": shape,
                        "axis_transformation": "identity",
                        "quantizer_parameters": dict(M05_QUANTIZER_PARAMETERS),
                        "dequantization_equation": M05_DEQUANTIZATION_EQUATION,
                        "role": role,
                        "residency_class": "immutable_device_text",
                        "disk_layout": disk,
                        "runtime_layout": disk,
                        "aliased": False,
                    })
            tensors.sort(key=lambda value: value["output_name"])
            plan_document = {
                "schema_version": 1,
                "artifact_profile": M05_PROFILE.name,
                "head_format": M05_PROFILE.head_format,
                "source_contract": M05_SOURCE_CONTRACT,
                "source_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
                "target_shard_bytes": 12,
                "approved_metadata_files": [],
                "omitted_families": ["audio", "mtp", "video", "vision"],
                "tensors": tensors,
                "excluded_tensors": [
                    {"source_name": router_name, "family": "deferred_non_attention",
                     "role": "router_projection", "residency_class": "m05_deferred_non_attention",
                     "reason": M05_PROFILE.deferred_reason},
                    {"source_name": vision_name, "family": "vision",
                     "role": "vision_projection", "residency_class": "compile_excluded_vision",
                     "reason": M05_VISION_EXCLUSION_REASON},
                ],
                "reference_environment": fixed_identity().environment,
            }
            plan_path = root / "compiler-plan.json"
            plan_path.write_bytes(canonical_json_bytes(plan_document))
            request_m05 = CompilerRequest(
                source_lock=lock_path,
                source_directory=source_dir,
                compiler_manifest=plan_path,
                profile=M05_PROFILE.name,
                head_format=M05_PROFILE.head_format,
                host_memory_cap_bytes=128 * 1024 * 1024,
                staging_bytes=4096,
                dependencies_lock=DEPENDENCIES,
            )
            with self.assertRaises(InvalidPlanError):
                plan_artifact(request_m05, identity=fixed_identity())
            return

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

    def test_statistics_report_record_retains_complete_provenance(self) -> None:
        written = SimpleNamespace(
            plan=SimpleNamespace(
                output_name="model.language_model.layers.0.self_attn.q_proj.weight",
                operation_id="fp8-attention:model.language_model.layers.0.self_attn.q_proj",
                source_names=("source.weight",),
                logical_dtype="BF16",
                logical_shape=(2, 4),
                quantizer_parameters={"contract_id": "gem16.fp8_attention_rowwise", "contract_version": 1},
            ),
            source_result=SimpleNamespace(
                source_sha256=("a" * 64,),
                statistics={"component": "weight", "elements": 8},
            ),
        )
        record = _complete_statistics_record(written)
        self.assertEqual(
            set(record),
            {
                "output_name", "operation_id", "source_names", "source_sha256",
                "logical_dtype", "logical_shape", "quantizer_parameters", "statistics",
            },
        )
        self.assertEqual(record["source_sha256"], ["a" * 64])
        self.assertEqual(record["logical_shape"], [2, 4])
        self.assertEqual(record["statistics"]["elements"], 8)

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
        self.assertEqual(
            compilation_schema["$defs"]["settingsM04"]["allOf"][1]["properties"]["threads"]["const"],
            1,
        )
        self.assertEqual(
            compilation_schema["$defs"]["settingsM05"]["allOf"][1]["properties"]["threads"]["minimum"],
            1,
        )
        self.assertEqual(
            compilation_schema["$defs"]["compilerM05"]["allOf"][1]["required"],
            ["native_encoder"],
        )
        m05_protocol = compilation_schema["$defs"]["compilerM05"]["allOf"][1]["properties"]["native_encoder"]["properties"]["protocol"]
        m06_protocol = compilation_schema["$defs"]["compilerM06"]["allOf"][1]["properties"]["native_encoder"]["properties"]["protocol"]
        self.assertEqual(m05_protocol, {"const": "gem16-fp8-batch-v1"})
        self.assertEqual(m06_protocol, {"const": "gem16-nvfp4-direct-v1"})
        self.assertNotEqual(m05_protocol, m06_protocol)
        self.assertEqual(
            compilation_schema["$defs"]["compilerM06"]["allOf"][1]["properties"]["native_encoder"]["properties"]["build"]["properties"]["build_type"],
            {"const": "Release"},
        )
        self.assertEqual(
            compilation_schema["$defs"]["compilerM04"]["allOf"][2]["not"]["required"],
            ["native_encoder"],
        )

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
            with self.assertRaisesRegex(DataError, "canonical layout|trusted encoder"):
                verify_artifact(request(), artifact, identity=fixed_identity())

    def test_mutable_manifest_hash_rewrite_is_reserved_for_m08_external_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact"
            compile_artifact(request(), artifact, identity=fixed_identity())
            rewrite_mutable_hashes_after_payload_tamper(artifact)
            report = verify_artifact(request(), artifact, identity=fixed_identity())
            self.assertNotIn("transformation_recomputed", report)

    def test_canonical_header_index_and_shard_names_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact"
            compile_artifact(request(), artifact, identity=fixed_identity())
            shard = artifact / "model-00001-of-00002.safetensors"
            raw = bytearray(shard.read_bytes())
            header_size = struct.unpack_from("<Q", raw)[0]
            raw[8 + header_size - 1] = ord(" ") if raw[8 + header_size - 1] != ord(" ") else ord("{")
            shard.write_bytes(raw)
            with self.assertRaisesRegex(DataError, "canonical .*header|file records"):
                verify_artifact(request(), artifact, identity=fixed_identity())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact"
            compile_artifact(request(), artifact, identity=fixed_identity())
            index = artifact / "model.safetensors.index.json"
            document = json.loads(index.read_text(encoding="utf-8"))
            document["metadata"]["total_size"] += 1
            index.write_bytes(canonical_json_bytes(document))
            with self.assertRaisesRegex(DataError, "canonical Safetensors index|file records"):
                verify_artifact(request(), artifact, identity=fixed_identity())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact"
            compile_artifact(request(), artifact, identity=fixed_identity())
            old = artifact / "model-00001-of-00002.safetensors"
            renamed = artifact / "model-00001-of-00002-renamed.safetensors"
            old.rename(renamed)
            with self.assertRaisesRegex(DataError, "file set differs from canonical plan"):
                verify_artifact(request(), artifact, identity=fixed_identity())

    def test_report_path_is_preflighted_and_collision_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifact"
            with self.assertRaisesRegex(OutputError, "must not be inside output"):
                compile_artifact(
                    request(), output, report_path=output / "report.json",
                    identity=fixed_identity(),
                )
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".incomplete").exists())

            collision = root / "collision.json"
            collision.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(OutputError, "already exists"):
                compile_artifact(
                    request(), root / "collision-artifact", report_path=collision,
                    identity=fixed_identity(),
                )
            self.assertEqual(collision.read_text(encoding="utf-8"), "keep")

    @unittest.skipUnless(hasattr(Path, "symlink_to"), "symbolic links unavailable")
    def test_verify_rejects_artifact_symlink_and_report_parent_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_parent = root / "real"
            real_parent.mkdir()
            artifact = real_parent / "artifact"
            compile_artifact(request(), artifact, identity=fixed_identity())

            artifact_link = root / "artifact-link"
            try:
                artifact_link.symlink_to(artifact, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symbolic links unavailable: {error}")
            with self.assertRaisesRegex(
                SourceVerificationError, "root must not be a symlink"
            ):
                verify_artifact(
                    request(), artifact_link, identity=fixed_identity()
                )

            report_parent_alias = root / "report-parent-alias"
            report_parent_alias.symlink_to(artifact, target_is_directory=True)
            aliased_report = report_parent_alias / "reports" / "verify.json"
            with self.assertRaisesRegex(OutputError, "must not be inside output"):
                verify_artifact(
                    request(), artifact, report_path=aliased_report,
                    identity=fixed_identity(),
                )
            self.assertFalse((artifact / "reports").exists())

            parent_alias = root / "parent-alias"
            parent_alias.symlink_to(real_parent, target_is_directory=True)
            report = root / "canonical-report.json"
            verification = verify_artifact(
                request(), parent_alias / "artifact", report_path=report,
                identity=fixed_identity(),
            )
            self.assertEqual(verification["artifact"], str(artifact.resolve()))
            self.assertTrue(report.is_file())

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
