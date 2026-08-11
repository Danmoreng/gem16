#!/usr/bin/env python3
"""Generate tiny deterministic M04 source and compiled Safetensors fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from typing import Any

try:
    from tools.gem16_compile.common import canonical_json_bytes, compact_json_bytes
    from tools.gem16_compile.compiler import (
        CompilerIdentity,
        CompilerRequest,
        compile_artifact,
    )
except ModuleNotFoundError:  # Direct execution from outside the repository root.
    from gem16_compile.common import canonical_json_bytes, compact_json_bytes
    from gem16_compile.compiler import CompilerIdentity, CompilerRequest, compile_artifact


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests/fixtures/gemma4_26b_compiler"
SOURCE_ROOT = FIXTURE_ROOT / "source"
EXPECTED_ROOT = FIXTURE_ROOT / "expected-artifact"
LOCK_PATH = FIXTURE_ROOT / "source.lock.json"
PLAN_PATH = FIXTURE_ROOT / "compiler-plan.json"
GOLDEN_PATH = (
    ROOT
    / "benchmarks/goldens/gemma4_26b/compiler/m04-synthetic-copy-hashes.json"
)
DEPENDENCIES_LOCK = ROOT / "tools/gem16_compile/dependencies.lock.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_blob_oid(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def safetensors_bytes(tensors: dict[str, tuple[str, list[int], bytes]]) -> bytes:
    header: dict[str, Any] = {"__metadata__": {"format": "pt", "fixture": "m04"}}
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        begin = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [begin, len(payload)],
        }
    encoded = compact_json_bytes(header)
    encoded += b" " * ((-len(encoded)) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + bytes(payload)


def source_outputs() -> dict[str, bytes]:
    embedding = bytes((index * 17 + 3) % 256 for index in range(16 * 4 * 2))
    vision = bytes((index * 11 + 5) % 256 for index in range(2 * 4 * 2))
    router = bytes((index * 7 + 9) % 256 for index in range(2 * 4 * 2))
    attention = bytes((index * 13 + 1) % 256 for index in range(4 * 4 * 2))
    shard1 = safetensors_bytes(
        {
            "model.language_model.embed_tokens.weight": (
                "BF16",
                [16, 4],
                embedding,
            ),
            "model.vision_tower.patch_embedder.input_proj.weight": (
                "BF16",
                [2, 4],
                vision,
            ),
        }
    )
    shard2 = safetensors_bytes(
        {
            "model.language_model.layers.0.router.proj.weight": (
                "BF16",
                [2, 4],
                router,
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight": (
                "BF16",
                [4, 4],
                attention,
            ),
        }
    )
    index = {
        "metadata": {
            "total_size": len(embedding) + len(vision) + len(router) + len(attention)
        },
        "weight_map": {
            "model.language_model.embed_tokens.weight": (
                "model-00001-of-00002.safetensors"
            ),
            "model.language_model.layers.0.router.proj.weight": (
                "model-00002-of-00002.safetensors"
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight": (
                "model-00002-of-00002.safetensors"
            ),
            "model.vision_tower.patch_embedder.input_proj.weight": (
                "model-00001-of-00002.safetensors"
            ),
        },
    }
    return {
        "model-00001-of-00002.safetensors": shard1,
        "model-00002-of-00002.safetensors": shard2,
        "model.safetensors.index.json": canonical_json_bytes(index),
        "config.json": canonical_json_bytes(
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "fixture": "m04-synthetic-copy-v1",
            }
        ),
        "generation_config.json": canonical_json_bytes({"eos_token_id": [1]}),
        "chat_template.jinja": b"{{ messages | length }}\n",
        "tokenizer.json": canonical_json_bytes(
            {"version": "1.0", "model": {"type": "BPE", "vocab": {}}}
        ),
        "tokenizer_config.json": canonical_json_bytes(
            {"model_type": "gemma4", "fixture": True}
        ),
    }


def source_lock(files: dict[str, bytes]) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": 2,
            "repository": "gem16/synthetic-gemma4-26b-m04",
            "revision": "1" * 40,
            "resolved_at_utc": "2026-08-11T00:00:00Z",
            "source_url": "https://example.invalid/gem16/m04-fixture",
            "terms_url": "project-generated-synthetic-fixture",
            "files": [
                {
                    "path": name,
                    "size": len(payload),
                    "sha256": sha256(payload),
                    "git_oid": git_blob_oid(payload),
                }
                for name, payload in sorted(files.items())
            ],
        }
    )


def compiler_plan(lock_payload: bytes) -> bytes:
    tensors = [
        {
            "output_name": "model.language_model.embed_tokens.weight",
            "operation_id": "copy:model.language_model.embed_tokens.weight",
            "source_names": ["model.language_model.embed_tokens.weight"],
            "encoder": "copy-v1",
            "transformation": "identity-copy",
            "transformation_version": 1,
            "output_dtype": "BF16",
            "physical_shape": [16, 4],
            "logical_dtype": "BF16",
            "logical_shape": [16, 4],
            "axis_transformation": "identity",
            "quantizer_parameters": {},
            "dequantization_equation": "output = source",
            "role": "tied_embedding_and_output",
            "residency_class": "compiler_scaffold_text",
            "disk_layout": "source_bf16",
            "runtime_layout": "not_runtime_loadable",
            "aliased": True,
        },
        {
            "output_name": "model.language_model.layers.0.router.proj.weight",
            "operation_id": "copy:model.language_model.layers.0.router.proj.weight",
            "source_names": ["model.language_model.layers.0.router.proj.weight"],
            "encoder": "copy-v1",
            "transformation": "identity-copy",
            "transformation_version": 1,
            "output_dtype": "BF16",
            "physical_shape": [2, 4],
            "logical_dtype": "BF16",
            "logical_shape": [2, 4],
            "axis_transformation": "identity",
            "quantizer_parameters": {},
            "dequantization_equation": "output = source",
            "role": "router_projection",
            "residency_class": "compiler_scaffold_text",
            "disk_layout": "source_bf16",
            "runtime_layout": "not_runtime_loadable",
            "aliased": False,
        },
        {
            "output_name": "model.language_model.layers.0.self_attn.q_proj.weight",
            "operation_id": "copy:model.language_model.layers.0.self_attn.q_proj.weight",
            "source_names": [
                "model.language_model.layers.0.self_attn.q_proj.weight"
            ],
            "encoder": "copy-v1",
            "transformation": "identity-copy",
            "transformation_version": 1,
            "output_dtype": "BF16",
            "physical_shape": [4, 4],
            "logical_dtype": "BF16",
            "logical_shape": [4, 4],
            "axis_transformation": "identity",
            "quantizer_parameters": {},
            "dequantization_equation": "output = source",
            "role": "attention_q_projection",
            "residency_class": "compiler_scaffold_text",
            "disk_layout": "source_bf16",
            "runtime_layout": "not_runtime_loadable",
            "aliased": False,
        },
    ]
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "artifact_profile": "synthetic-copy-v1",
            "head_format": "source",
            "source_contract": "gemma4_26b_m04_synthetic_source_v1",
            "source_lock_sha256": sha256(lock_payload),
            "target_shard_bytes": 96,
            "approved_metadata_files": [
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            "omitted_families": ["audio", "mtp", "video", "vision"],
            "tensors": tensors,
            "excluded_tensors": [
                {
                    "source_name": "model.vision_tower.patch_embedder.input_proj.weight",
                    "family": "vision",
                    "role": "vision_projection",
                    "residency_class": "compile_excluded_vision",
                    "reason": "text-only M04 synthetic fixture",
                }
            ],
            "reference_environment": {
                "system": "Linux",
                "machine": "x86_64",
                "python_implementation": "CPython",
                "python_version": "3.14.6",
                "python_major_minor": "3.14",
                "byteorder": "little",
                "locale": "C.UTF-8",
            },
        }
    )


def build_outputs() -> dict[Path, bytes]:
    source = source_outputs()
    lock = source_lock(source)
    plan = compiler_plan(lock)
    outputs: dict[Path, bytes] = {
        SOURCE_ROOT / name: payload for name, payload in source.items()
    }
    outputs[LOCK_PATH] = lock
    outputs[PLAN_PATH] = plan

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_root = root / "source"
        source_root.mkdir()
        for name, payload in source.items():
            destination = source_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        lock_path = root / "source.lock.json"
        plan_path = root / "compiler-plan.json"
        lock_path.write_bytes(lock)
        plan_path.write_bytes(plan)
        artifact = root / "artifact"
        request = CompilerRequest(
            source_lock=lock_path,
            source_directory=source_root,
            compiler_manifest=plan_path,
            profile="synthetic-copy-v1",
            head_format="source",
            host_memory_cap_bytes=512 * 1024 * 1024,
            staging_bytes=64 * 1024,
            dependencies_lock=DEPENDENCIES_LOCK,
        )
        identity = CompilerIdentity(
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
        compile_artifact(request, artifact, identity=identity)
        artifact_hashes: dict[str, dict[str, Any]] = {}
        for path in sorted(artifact.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(artifact).as_posix()
            payload = path.read_bytes()
            outputs[EXPECTED_ROOT / relative] = payload
            artifact_hashes[relative] = {
                "size": len(payload),
                "sha256": sha256(payload),
            }
    outputs[GOLDEN_PATH] = canonical_json_bytes(
        {
            "schema_version": 1,
            "milestone": "M04",
            "status": "deterministic_synthetic_copy_fixture",
            "source_lock_sha256": sha256(lock),
            "compiler_manifest_sha256": sha256(plan),
            "dependencies_lock_sha256": sha256(DEPENDENCIES_LOCK.read_bytes()),
            "compiler_identity": {
                "repository": "Danmoreng/gem16",
                "commit": "0" * 40,
                "dirty": False,
            },
            "files": artifact_hashes,
        }
    )
    return outputs


def main() -> int:
    args = parse_args()
    outputs = build_outputs()
    roots = (SOURCE_ROOT, EXPECTED_ROOT)
    stale = [
        path
        for path, payload in outputs.items()
        if not path.is_file() or path.read_bytes() != payload
    ]
    expected_paths = {path.resolve() for path in outputs}
    extra = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() not in expected_paths
    ]
    if args.check:
        if stale or extra:
            print(
                "stale M04 compiler fixtures: "
                + ", ".join(str(path.relative_to(ROOT)) for path in stale + extra)
            )
            return 1
    else:
        for root in roots:
            if root.exists():
                for path in sorted(root.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
        for path, payload in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    action = "verified" if args.check else "generated"
    print(f"M04 compiler fixtures {action}: {len(outputs)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
