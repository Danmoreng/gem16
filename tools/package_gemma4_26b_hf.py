#!/usr/bin/env python3
"""Prepare the two immutable Hugging Face packages for qualified 26B M25."""

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


TARGET_IMAGE_BYTES = 14_696_668_160
TARGET_IMAGE_SHA256 = (
    "1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72"
)
TARGET_SOURCE = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
ASSISTANT_SOURCE = (
    "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"
)


class PackageError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def regular(path: Path, description: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PackageError(f"cannot inspect {description} {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PackageError(f"{description} is not a regular file: {path}")


def load_lock(path: Path) -> dict[str, Any]:
    regular(path, "artifact lock")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("files"), list):
        raise PackageError(f"invalid artifact lock: {path}")
    return document


def link_or_copy(source: Path, destination: Path) -> None:
    regular(source, "package source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def checked_file(source: Path, record: dict[str, Any]) -> Path:
    name = record.get("path")
    expected_size = record.get("size")
    expected_hash = record.get("sha256")
    if not isinstance(name, str) or Path(name).name != name:
        raise PackageError(f"unsafe artifact path: {name!r}")
    path = source / name
    regular(path, "locked artifact")
    if path.stat().st_size != expected_size or sha256(path) != expected_hash:
        raise PackageError(f"locked artifact differs from its lock: {path}")
    return path


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def prepare_empty(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
            raise PackageError(f"package output must be an empty directory: {path}")
    else:
        path.mkdir(parents=True)


def target_card() -> str:
    return """---
license: gemma
library_name: gem16
pipeline_tag: text-generation
base_model: google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
tags:
- gemma
- gem16
- nvfp4
- sm120
---

# Gemma 4 26B A4B — GEM16 SM120

Qualified text-only GEM16 checkpoint for Gemma 4 26B A4B on NVIDIA Blackwell
SM120. `model.gem16` is an offline-derived, immutable GPU-resident weight arena
using the exact runtime layout consumed by gem16. It is not Safetensors, GGUF,
or a Transformers checkpoint and requires the gem16 runtime.

Source: `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`, pinned in
`gem16_compilation.json` and `gem16.lock.json`. The package is text-only;
vision, audio, and video weights are intentionally omitted.

Qualified product profile: fixed D2 MTP with the separately published GEM16
Assistant, one resident session, FP8 KV cache, maximum MTP context 86,016
tokens. The Target-only runtime remains qualified through 98,304 tokens.
"""


def assistant_card() -> str:
    return """---
license: gemma
library_name: gem16
pipeline_tag: text-generation
base_model: google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant
tags:
- gemma
- gem16
- nvfp4
- mtp
- sm120
---

# Gemma 4 26B A4B Assistant — GEM16 SM120

Qualified fixed-D2 MTP Assistant for the separately published GEM16 Gemma 4
26B A4B Target. The hybrid NVFP4/FP8 checkpoint uses a GEM16-specific runtime
layout and is not a standalone chat model.

Source: `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant`, pinned in
`gem16_compilation.json` and `gem16.lock.json`. Use only with the matching
qualified Target and gem16 runtime.
"""


def model_metadata(kind: str, lock: dict[str, Any]) -> dict[str, Any]:
    target = kind == "target"
    return {
        "schema_version": 1,
        "format": "gem16",
        "format_version": "sm120-device-image-v1" if target else "sm120-mtp-assistant-hybrid-v1",
        "kind": kind,
        "qualification_status": "qualified",
        "source_repository": TARGET_SOURCE if target else ASSISTANT_SOURCE,
        "source_revision": (
            "f1e06dc520982d9b9edd76859fdb7ab209449949"
            if target
            else "9537141506fe8875b3ed45b264af13580cb29166"
        ),
        "artifact_content_sha256": lock["artifact_content_sha256"],
        "model_file": "model.gem16" if target else None,
        "model_file_bytes": TARGET_IMAGE_BYTES if target else None,
        "model_file_sha256": TARGET_IMAGE_SHA256 if target else None,
        "text_only": True,
        "mtp_draft_tokens": 2,
        "maximum_context_tokens": 73_728,
        "target_only_maximum_context_tokens": 98_304 if target else None,
    }


def package_target(source: Path, image: Path, lock_path: Path, output: Path) -> None:
    lock = load_lock(lock_path)
    prepare_empty(output)
    for record in lock["files"]:
        name = record.get("path")
        if name == "model.safetensors.index.json" or Path(str(name)).suffix == ".safetensors":
            continue
        link_or_copy(checked_file(source, record), output / str(name))
    regular(image, "GEM16 Target image")
    if image.stat().st_size != TARGET_IMAGE_BYTES or sha256(image) != TARGET_IMAGE_SHA256:
        raise PackageError("Target device image does not match the qualified image")
    link_or_copy(image, output / "model.gem16")
    link_or_copy(lock_path, output / "gem16.lock.json")
    write_text(output / ".gitattributes", "*.gem16 filter=lfs diff=lfs merge=lfs -text\n")
    write_text(output / "README.md", target_card())
    write_text(
        output / "gem16_model.json",
        json.dumps(model_metadata("target", lock), indent=2, sort_keys=True) + "\n",
    )


def package_assistant(source: Path, lock_path: Path, output: Path) -> None:
    lock = load_lock(lock_path)
    prepare_empty(output)
    for record in lock["files"]:
        link_or_copy(checked_file(source, record), output / str(record["path"]))
    link_or_copy(lock_path, output / "gem16.lock.json")
    write_text(output / ".gitattributes", "*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    write_text(output / "README.md", assistant_card())
    write_text(
        output / "gem16_model.json",
        json.dumps(model_metadata("assistant", lock), indent=2, sort_keys=True) + "\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--target-lock", type=Path, required=True)
    parser.add_argument("--assistant", type=Path, required=True)
    parser.add_argument("--assistant-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package_target(
        options.target.resolve(), options.target_image.resolve(),
        options.target_lock.resolve(), output / "target",
    )
    package_assistant(
        options.assistant.resolve(), options.assistant_lock.resolve(),
        output / "assistant",
    )
    print(f"target_package={output / 'target'}")
    print(f"assistant_package={output / 'assistant'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, PackageError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(4)
