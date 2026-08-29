#!/usr/bin/env python3
"""Prepare the two immutable Hugging Face packages for qualified 26B M25.

The package cards retain the body of Google's pinned upstream model cards. The
GEM16-specific preamble and Hub metadata are generated here so the derived
repositories remain visibly linked to their exact upstream checkpoints.
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


TARGET_IMAGE_BYTES = 14_696_668_160
TARGET_IMAGE_SHA256 = (
    "1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72"
)
TARGET_SOURCE = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
ASSISTANT_SOURCE = (
    "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"
)
TARGET_SOURCE_REVISION = "f1e06dc520982d9b9edd76859fdb7ab209449949"
ASSISTANT_SOURCE_REVISION = "9537141506fe8875b3ed45b264af13580cb29166"
TARGET_REPOSITORY = "danmoreng/gemma-4-26B-A4B-it-GEM16"
ASSISTANT_REPOSITORY = "danmoreng/gemma-4-26B-A4B-it-assistant-GEM16"
PROJECT_URL = "https://github.com/Danmoreng/gem16"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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


def upstream_card_body(card: str) -> str:
    """Return an upstream card without its Hub YAML front matter."""
    normalized = card.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return normalized.lstrip()
    delimiter = normalized.find("\n---\n", 4)
    if delimiter < 0:
        raise PackageError("upstream model card has unterminated YAML front matter")
    return normalized[delimiter + len("\n---\n"):].lstrip()


def model_card(kind: str, upstream_card: str) -> str:
    target = kind == "target"
    if kind not in {"target", "assistant"}:
        raise PackageError(f"invalid package kind: {kind}")
    source = TARGET_SOURCE if target else ASSISTANT_SOURCE
    repository = TARGET_REPOSITORY if target else ASSISTANT_REPOSITORY
    companion = ASSISTANT_REPOSITORY if target else TARGET_REPOSITORY
    tags = ["gemma", "gem16", "quantized", "nvfp4", "sm120"]
    if not target:
        tags.append("mtp")
    tag_lines = "\n".join(f"- {tag}" for tag in tags)
    title = (
        "Gemma 4 26B A4B — GEM16 SM120"
        if target else "Gemma 4 26B A4B Assistant — GEM16 SM120"
    )
    role = (
        "a qualified, text-only Target checkpoint"
        if target else "the fixed-D2 MTP Assistant for the matching GEM16 Target"
    )
    format_warning = (
        "It is not a Transformers, Safetensors, or GGUF checkpoint, and it is "
        "not intended for other inference engines."
        if target else
        "It is not a standalone chat model and is not intended for other "
        "inference engines."
    )
    artifact = (
        "`model.gem16` is a 14,696,668,160-byte immutable GPU-resident arena "
        "in the exact layout consumed by GEM16."
        if target else
        "The model weights use GEM16's hybrid NVFP4/FP8 Assistant layout and "
        "are not a standalone chat model."
    )
    pairing = (
        f"For fixed-D2 MTP, install the matching [Assistant]"
        f"(https://huggingface.co/{companion})."
        if target else
        f"Use this repository only with the matching [Target]"
        f"(https://huggingface.co/{companion})."
    )
    return f"""---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
library_name: gem16
pipeline_tag: text-generation
base_model: {source}
base_model_relation: quantized
tags:
{tag_lines}
---

# {title}

> [!IMPORTANT]
> This is {role}, compiled specifically for the
> [GEM16 inference engine]({PROJECT_URL}) on NVIDIA Blackwell SM120 GPUs.
> {format_warning}

This repository is an offline-derived quantization/runtime-layout conversion
of [`{source}`](https://huggingface.co/{source}). Hugging Face records that
relationship through `base_model_relation: quantized`, allowing navigation in
both directions through the Hub model tree. {artifact}

{pairing}

## GEM16 runtime contract

- text generation only; upstream vision/audio/video capabilities are not
  included in this GEM16 package;
- NVIDIA Blackwell SM120 and the GEM16 runtime are required;
- fixed-D2 MTP supports up to 86,016 context tokens;
- Target-only execution supports up to 98,304 context tokens;
- source identity and transformations are pinned in `gem16.lock.json`,
  `gem16_compilation.json`, and `gem16_model.json`.

Repository: [`{repository}`](https://huggingface.co/{repository})

Engine and usage documentation: [{PROJECT_URL}]({PROJECT_URL})

The package remains under the upstream Apache License 2.0. See `LICENSE`,
`NOTICE`, and the upstream license link above.

---

## Upstream model card

The following card body is retained from the exact pinned Google source. Its
general capability and usage statements describe the upstream checkpoint;
the narrower GEM16 runtime contract above governs this derived package.

{upstream_card_body(upstream_card).rstrip()}
"""


def publication_notice(kind: str) -> str:
    target = kind == "target"
    source = TARGET_SOURCE if target else ASSISTANT_SOURCE
    revision = TARGET_SOURCE_REVISION if target else ASSISTANT_SOURCE_REVISION
    return f"""GEM16 Gemma 4 26B A4B {kind.capitalize()} artifact

This distribution contains a modified model artifact derived from:
  {source}
  revision {revision}

The upstream Gemma model and model card are provided by Google LLC under the
Apache License, Version 2.0. The GEM16 package preserves that license.

GEM16 modifications include offline quantization and/or tensor layout
conversion for the GEM16 SM120 runtime, provenance manifests, and omission of
non-text model families from the product package. The Target and Assistant are
separate artifacts and must be used as documented in README.md.

GEM16 source and documentation: {PROJECT_URL}
Upstream terms and notices: https://ai.google.dev/gemma/terms
"""


def write_publication_metadata(output: Path, kind: str, upstream_card_path: Path) -> None:
    regular(upstream_card_path, "upstream model card")
    link_or_copy(REPOSITORY_ROOT / "LICENSE", output / "LICENSE")
    write_text(output / "NOTICE", publication_notice(kind))
    write_text(
        output / "README.md",
        model_card(kind, upstream_card_path.read_text(encoding="utf-8")),
    )


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
            TARGET_SOURCE_REVISION if target else ASSISTANT_SOURCE_REVISION
        ),
        "artifact_content_sha256": lock["artifact_content_sha256"],
        "model_file": "model.gem16" if target else None,
        "model_file_bytes": TARGET_IMAGE_BYTES if target else None,
        "model_file_sha256": TARGET_IMAGE_SHA256 if target else None,
        "text_only": True,
        "mtp_draft_tokens": 2,
        "maximum_context_tokens": 86_016,
        "target_only_maximum_context_tokens": 98_304 if target else None,
    }


def package_target(
    source: Path, image: Path, lock_path: Path, upstream_card: Path, output: Path,
) -> None:
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
    write_publication_metadata(output, "target", upstream_card)
    write_text(
        output / "gem16_model.json",
        json.dumps(model_metadata("target", lock), indent=2, sort_keys=True) + "\n",
    )


def package_assistant(
    source: Path, lock_path: Path, upstream_card: Path, output: Path,
) -> None:
    lock = load_lock(lock_path)
    prepare_empty(output)
    for record in lock["files"]:
        link_or_copy(checked_file(source, record), output / str(record["path"]))
    link_or_copy(lock_path, output / "gem16.lock.json")
    write_text(output / ".gitattributes", "*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    write_publication_metadata(output, "assistant", upstream_card)
    write_text(
        output / "gem16_model.json",
        json.dumps(model_metadata("assistant", lock), indent=2, sort_keys=True) + "\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--target-lock", type=Path, required=True)
    parser.add_argument("--target-upstream-card", type=Path, required=True)
    parser.add_argument("--assistant", type=Path, required=True)
    parser.add_argument("--assistant-lock", type=Path, required=True)
    parser.add_argument("--assistant-upstream-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package_target(
        options.target.resolve(), options.target_image.resolve(),
        options.target_lock.resolve(), options.target_upstream_card.resolve(),
        output / "target",
    )
    package_assistant(
        options.assistant.resolve(), options.assistant_lock.resolve(),
        options.assistant_upstream_card.resolve(), output / "assistant",
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
