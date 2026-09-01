#!/usr/bin/env python3
"""Prepare the consolidated 26B Target/Assistant/Vision Hub update.

The output is an additive update for danmoreng/gemma-4-26B-A4B-it-GEM16:
the qualified NVFP4 Target already in the repository root remains untouched,
while Trellis35, Assistant, and Vision components are added below bounded
subdirectories. Payload files are hardlinked when possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any


REPOSITORY = "danmoreng/gemma-4-26B-A4B-it-GEM16"
PROJECT_URL = "https://github.com/Danmoreng/gem16"
ASSISTANT_ARTIFACT_CONTENT_SHA256 = (
    "978f5e3804dd08b8ea5551883811e0bf6737a23ae5ae3dcfa0ef71dc4ffe532b"
)
NVFP4_ARTIFACT_CONTENT_SHA256 = (
    "471805f7dad8abb84300be78b2822a63dcb1d35bff5aa98426a162cc8532ee17"
)
NVFP4_MODEL_SHA256 = (
    "1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72"
)
TRELLIS_PROFILE = "gem16-trellis35-w4a8-v1"
VISION_PROFILE = "gemma4_26b_trellis35_vision_fp8"
EXPECTED_TARGET_METADATA = {
    "config.json": "8a647d5444c9e77b03bd80ac802683ffdbf64b3d9296ca5257ced8e50349ea16",
    "generation_config.json": "b69207f9be617e982d13cc273cce6fd88c98dda99a4bdc5e2d52ffe0a0d9f0a9",
    "chat_template.jinja": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json": "3ab5c7b94dc97d65ca7064496fa69b88ff875378e1cb7ee3e43070c3a8170999",
}


class PackageError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def resolved_regular(path: Path, allowed_root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        root = allowed_root.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise PackageError(f"cannot inspect {label} {path}: {error}") from error
    if not resolved.is_relative_to(root) or not stat.S_ISREG(metadata.st_mode):
        raise PackageError(f"unsafe or non-regular {label}: {path}")
    return resolved


def checked(path: Path, allowed_root: Path, expected_hash: str,
            expected_size: int | None = None) -> Path:
    resolved = resolved_regular(path, allowed_root, "component file")
    if expected_size is not None and resolved.stat().st_size != expected_size:
        raise PackageError(f"size mismatch: {path}")
    if sha256(resolved) != expected_hash:
        raise PackageError(f"SHA-256 mismatch: {path}")
    return resolved


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def prepare_empty(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise PackageError(f"output must be an empty directory: {path}")
    else:
        path.mkdir(parents=True)


def load_object(path: Path, allowed_root: Path) -> dict[str, Any]:
    resolved = resolved_regular(path, allowed_root, "JSON document")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PackageError(f"JSON root is not an object: {path}")
    return value


def package_trellis(source: Path, output: Path) -> dict[str, Any]:
    descriptor = load_object(source / "trellis35-checkpoint.json", source)
    if descriptor.get("checkpoint_profile") != TRELLIS_PROFILE:
        raise PackageError("unexpected Trellis35 checkpoint profile")
    records: list[tuple[str, int, str]] = []
    for name, expected_hash in EXPECTED_TARGET_METADATA.items():
        path = checked(source / name, source, expected_hash)
        records.append((name, path.stat().st_size, expected_hash))
    descriptor_hash = sha256(source / "trellis35-checkpoint.json")
    records.append(("trellis35-checkpoint.json",
                    (source / "trellis35-checkpoint.json").stat().st_size,
                    descriptor_hash))
    for section_name, name_key, hash_key, size_key in (
        ("non_routed", "artifact", "artifact_sha256", "artifact_bytes"),
        ("non_routed", "manifest", "manifest_sha256", None),
        ("routed_experts", "index", "index_sha256", None),
    ):
        section = descriptor.get(section_name)
        if not isinstance(section, dict):
            raise PackageError(f"missing Trellis35 {section_name} section")
        relative = section.get(name_key)
        expected_hash = section.get(hash_key)
        expected_size = section.get(size_key) if size_key else None
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise PackageError(f"invalid Trellis35 {name_key} record")
        path = checked(source / relative, source, expected_hash, expected_size)
        records.append((relative, path.stat().st_size, expected_hash))
    routed = descriptor["routed_experts"]
    layers = routed.get("layers")
    if not isinstance(layers, list) or len(layers) != 30:
        raise PackageError("Trellis35 checkpoint must contain exactly 30 layers")
    for layer_index, layer in enumerate(layers):
        if not isinstance(layer, dict) or layer.get("layer") != layer_index:
            raise PackageError("Trellis35 layer order is invalid")
        for name_key, hash_key, size_key in (
            ("artifact", "artifact_sha256", "artifact_bytes"),
            ("manifest", "manifest_sha256", None),
            ("verification", "verification_sha256", None),
        ):
            relative = layer.get(name_key)
            expected_hash = layer.get(hash_key)
            expected_size = layer.get(size_key) if size_key else None
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise PackageError("invalid Trellis35 layer record")
            path = checked(source / relative, source, expected_hash, expected_size)
            records.append((relative, path.stat().st_size, expected_hash))
    for relative, _size, _digest in records:
        link_or_copy(resolved_regular(source / relative, source, "Trellis35 file"),
                     output / relative)
    checksums = "".join(f"{digest}  {name}\n" for name, unused, digest in records)
    write_text(output / "SHA256SUMS", checksums)
    write_text(output / "README.md", """# GEM16 Trellis35 Target component

This directory contains the experimental GEM16 Trellis35 W4A8 Target for
Gemma 4 26B on NVIDIA Blackwell SM120. It is the required text component for
the sibling `vision/` module. It is not interchangeable with the qualified
NVFP4 Target in the repository root.
""")
    repository_root = Path(__file__).resolve().parents[1]
    link_or_copy(resolved_regular(repository_root / "LICENSE", repository_root,
                                  "license"), output / "LICENSE")
    write_text(output / "NOTICE", """GEM16 Gemma 4 26B Trellis35 component

Derived from the pinned Gemma 4 26B QAT source. The exact artifact identities,
layout, compiler provenance, and source locks are recorded in
trellis35-checkpoint.json and its referenced manifests.
""")
    return {
        "path": "trellis35",
        "profile": TRELLIS_PROFILE,
        "checkpoint_content_sha256": descriptor["checkpoint_content_sha256"],
        "checkpoint_descriptor_sha256": descriptor_hash,
        "weight_bytes": descriptor["arena"]["total_bytes"],
        "files": len(records) + 4,
    }


def package_assistant(source: Path, lock_path: Path,
                      output: Path) -> dict[str, Any]:
    lock = load_object(lock_path, lock_path.parent)
    if lock.get("repository") != "danmoreng/gemma-4-26B-A4B-it-assistant-GEM16":
        raise PackageError("unexpected Assistant lock repository")
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        raise PackageError("Assistant lock contains no files")
    allowed_root = source.parent.parent
    for record in files:
        if not isinstance(record, dict):
            raise PackageError("Assistant lock file record is invalid")
        relative = record.get("path")
        expected_hash = record.get("sha256")
        expected_size = record.get("size")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise PackageError("Assistant path must be a root filename")
        if not isinstance(expected_hash, str) or not isinstance(expected_size, int):
            raise PackageError("Assistant identity is invalid")
        path = checked(source / relative, allowed_root, expected_hash, expected_size)
        link_or_copy(path, output / relative)
    return {
        "path": "assistant",
        "artifact_content_sha256": ASSISTANT_ARTIFACT_CONTENT_SHA256,
        "source_repository": lock["repository"],
        "source_revision": lock["revision"],
        "files": len(files),
    }


def package_vision(source: Path, output: Path) -> dict[str, Any]:
    descriptor = load_object(source / "gem16_vision.json", source)
    lock = load_object(source / "vision.lock.json", source)
    if descriptor.get("capability_profile") != VISION_PROFILE:
        raise PackageError("unexpected Vision capability profile")
    artifact_name = descriptor.get("artifact")
    artifact_hash = descriptor.get("artifact_sha256")
    artifact_size = descriptor.get("artifact_size")
    compilation_name = descriptor.get("compilation_manifest")
    compilation_hash = descriptor.get("compilation_manifest_sha256")
    if not all(isinstance(value, str) for value in
               (artifact_name, artifact_hash, compilation_name, compilation_hash)):
        raise PackageError("Vision descriptor identity is invalid")
    if not isinstance(artifact_size, int):
        raise PackageError("Vision artifact size is invalid")
    checked(source / artifact_name, source, artifact_hash, artifact_size)
    checked(source / compilation_name, source, compilation_hash)
    if lock.get("artifact_sha256") != artifact_hash or \
            lock.get("required_text_artifact_profile") != TRELLIS_PROFILE:
        raise PackageError("Vision lock is incompatible with the Trellis Target")
    for name in ("vision.gem16", "gem16_vision.json",
                 "vision_compilation.json", "vision.lock.json"):
        link_or_copy(resolved_regular(source / name, source, "Vision file"),
                     output / name)
    write_text(output / "README.md", """# GEM16 FP8 Vision component

This directory contains the experimental FP8 Vision module for the GEM16
Gemma 4 26B Trellis35 profile. It requires the `trellis35/` Target from this
same immutable repository revision. It is not a standalone model and is not
compatible with the qualified NVFP4 Target in the repository root.
""")
    link_or_copy(resolved_regular(Path(__file__).resolve().parents[1] / "LICENSE",
                                  Path(__file__).resolve().parents[1], "license"),
                 output / "LICENSE")
    write_text(output / "NOTICE", """GEM16 Gemma 4 26B FP8 Vision component

Derived from google/gemma-4-26B-A4B-it-qat-q4_0-unquantized at revision
f1e06dc520982d9b9edd76859fdb7ab209449949. See vision.lock.json and
vision_compilation.json for the exact source and transformation provenance.
""")
    return {
        "path": "vision",
        "profile": VISION_PROFILE,
        "artifact_sha256": artifact_hash,
        "artifact_size": artifact_size,
        "required_text_artifact_profile": TRELLIS_PROFILE,
        "files": 7,
    }


def consolidated_readme(existing: str) -> str:
    marker = "## Consolidated GEM16 components"
    if marker in existing:
        raise PackageError("existing Target README already has a component section")
    return existing.rstrip() + f"""

{marker}

This repository is the single canonical source for the 26B Target,
fixed-D2 Assistant, and FP8 Vision module on NVIDIA Blackwell SM120. These
artifacts are offline-compiled for [{PROJECT_URL}]({PROJECT_URL}); they are not
generic Transformers, Safetensors, or GGUF checkpoints.

## Components

- repository root: qualified text-only NVFP4 Target (kept backward compatible);
- `trellis35/`: experimental compact Trellis35 Target required by Vision;
- `assistant/`: exact fixed-D2 Assistant;
- `vision/`: experimental FP8 Vision module;
- `gem16_components.json`: exact compatibility and qualification matrix.

Vision works only with `trellis35/`. The qualified NVFP4 Target in the root is
text-only. Fixed-D2 with Vision is enabled only for the exact component hashes
recorded in the compatibility manifest. The Vision profile supports one image,
batch one, and soft-token budgets 70, 140, or 280.

The separate historical Assistant repository remains immutable for existing
locks, but new GEM16 installations resolve every 26B component from this repo.
"""


def preserve_root_metadata(source: Path, lock_path: Path, output: Path) -> None:
    lock = load_object(lock_path, lock_path.parent)
    if lock.get("repository") != REPOSITORY:
        raise PackageError("unexpected Target lock repository")
    records = lock.get("files")
    if not isinstance(records, list):
        raise PackageError("Target lock files are invalid")
    by_path = {
        record.get("path"): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    allowed_root = source.parent.parent
    resolved: dict[str, Path] = {}
    for name in (".gitattributes", "LICENSE", "NOTICE", "README.md"):
        record = by_path.get(name)
        if not isinstance(record, dict):
            raise PackageError(f"Target lock does not contain {name}")
        digest = record.get("sha256")
        size = record.get("size")
        if not isinstance(digest, str) or not isinstance(size, int):
            raise PackageError(f"Target lock identity is invalid for {name}")
        resolved[name] = checked(source / name, allowed_root, digest, size)
    write_text(output / "README.md",
               consolidated_readme(resolved["README.md"].read_text(encoding="utf-8")))
    link_or_copy(resolved["LICENSE"], output / "LICENSE")
    write_text(output / "NOTICE",
               resolved["NOTICE"].read_text(encoding="utf-8").rstrip() + """

The same immutable repository revision also contains the Trellis35 Target,
fixed-D2 Assistant, and FP8 Vision components under their named directories.
See gem16_components.json for the exact compatibility contract.
""")
    attributes = resolved[".gitattributes"].read_text(encoding="utf-8").rstrip()
    additions = []
    for pattern in ("*.bin", "*.safetensors"):
        if pattern not in attributes:
            additions.append(f"{pattern} filter=lfs diff=lfs merge=lfs -text")
    write_text(output / ".gitattributes",
               attributes + ("\n" + "\n".join(additions) if additions else "") + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-lock", type=Path, required=True)
    parser.add_argument("--trellis-target", type=Path, required=True)
    parser.add_argument("--assistant", type=Path, required=True)
    parser.add_argument("--assistant-lock", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    output = options.output.resolve()
    prepare_empty(output)
    preserve_root_metadata(options.target.resolve(),
                           options.target_lock.resolve(), output)
    trellis = package_trellis(options.trellis_target.resolve(),
                              output / "trellis35")
    assistant = package_assistant(options.assistant.resolve(),
                                  options.assistant_lock.resolve(),
                                  output / "assistant")
    vision = package_vision(options.vision.resolve(), output / "vision")
    compatibility = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "components": {
            "nvfp4_target": {
                "path": ".",
                "artifact_content_sha256": NVFP4_ARTIFACT_CONTENT_SHA256,
                "model_sha256": NVFP4_MODEL_SHA256,
                "qualification": "qualified_text_only",
            },
            "trellis35_target": trellis,
            "fixed_d2_assistant": assistant,
            "fp8_vision": vision,
        },
        "compatible_profiles": [
            {
                "profile_id": "gemma4-26b-a4b-nvfp4",
                "components": ["nvfp4_target"],
                "qualification": "qualified",
            },
            {
                "profile_id": "gemma4-26b-a4b-nvfp4-d2",
                "components": ["nvfp4_target", "fixed_d2_assistant"],
                "qualification": "qualified",
            },
            {
                "profile_id": "gemma4-26b-a4b-trellis35-vision-fp8",
                "components": ["trellis35_target", "fp8_vision"],
                "qualification": "experimental",
            },
            {
                "profile_id": "gemma4-26b-a4b-trellis35-vision-fp8-d2",
                "components": ["trellis35_target", "fp8_vision",
                               "fixed_d2_assistant"],
                "qualification": "experimental_v14_accepted",
            },
        ],
    }
    compatibility["content_sha256"] = canonical_json_sha256(compatibility)
    write_text(output / "gem16_components.json",
               json.dumps(compatibility, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "repository": REPOSITORY,
        "output": str(output),
        "trellis35": trellis,
        "assistant": assistant,
        "vision": vision,
        "compatibility_content_sha256": compatibility["content_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError, PackageError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(4)
