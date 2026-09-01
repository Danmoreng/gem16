#!/usr/bin/env python3
"""Generate Native Studio's immutable model catalog from repository locks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "nativeStudio/src/model_catalog.generated.h"
COMPONENTS = (
    (
        "Gemma4Unified12BTarget",
        "gemma4-12b-target",
        "12B Target",
        ROOT / "models/gemma4-12b-nvfp4.lock.json",
    ),
    (
        "Gemma4Unified12BAssistant",
        "gemma4-12b-assistant",
        "12B Assistant",
        ROOT / "models/gemma4-12b-mtp-assistant.lock.json",
    ),
    (
        "Gemma4Moe26BA4BTarget",
        "gemma4-26b-target",
        "26B Target",
        ROOT / "models/gemma4-26b-gem16-target.lock.json",
    ),
    (
        "Gemma4Moe26BA4BAssistant",
        "gemma4-26b-assistant",
        "26B Assistant",
        ROOT / "models/gemma4-26b-gem16-assistant.lock.json",
    ),
    (
        "Gemma4Moe26BTrellis35Target",
        "gemma4-26b-trellis35-target",
        "26B Trellis35 Target",
        ROOT / "models/gemma4-26b-trellis35-target.lock.json",
    ),
    (
        "Gemma4Moe26BVisionFp8",
        "gemma4-26b-vision-fp8",
        "26B FP8 Vision",
        ROOT / "models/gemma4-26b-vision-fp8.lock.json",
    ),
)
HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
REPOSITORY = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z"
)
PATH_SEGMENT = re.compile(r"[A-Za-z0-9._-]+\Z")
VISION_RUNTIME_FILES = {
    "gem16_vision.json",
    "vision.gem16",
    "vision.lock.json",
    "vision_compilation.json",
}


def quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def validate_repository(value: Any, context: str) -> str:
    if not isinstance(value, str) or REPOSITORY.fullmatch(value) is None:
        raise ValueError(f"{context}: invalid Hugging Face repository")
    return value


def validate_revision(value: Any, context: str) -> str:
    if not isinstance(value, str) or HEX_40.fullmatch(value) is None:
        raise ValueError(f"{context}: revision must be a lowercase 40-hex commit")
    return value


def validate_relative_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context}: invalid portable relative path")
    parts = value.split("/")
    if any(
        part in {"", ".", ".."} or PATH_SEGMENT.fullmatch(part) is None
        for part in parts
    ):
        raise ValueError(f"{context}: unsafe relative path")
    return value


def load_lock(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"lock is not an object: {path}")
    if document.get("schema_version") not in {1, 2}:
        raise ValueError(f"unsupported lock schema: {path}")
    validate_repository(document.get("repository"), str(path))
    validate_revision(document.get("revision"), str(path))
    if not isinstance(document.get("files"), list) or not document["files"]:
        raise ValueError(f"lock has no files: {path}")
    seen: set[str] = set()
    for index, entry in enumerate(document["files"]):
        context = f"{path}: files[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{context}: file entry is not an object")
        locked_path = validate_relative_path(entry.get("path"), context)
        if locked_path in seen:
            raise ValueError(f"{context}: duplicate path {locked_path}")
        seen.add(locked_path)
        size = entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{context}: size must be a non-negative integer")
        if not isinstance(entry.get("sha256"), str) or HEX_64.fullmatch(entry["sha256"]) is None:
            raise ValueError(f"{context}: invalid SHA-256")
        blob_id = entry.get("lfs_oid") or entry.get("git_oid")
        if not isinstance(blob_id, str) or not (
            HEX_40.fullmatch(blob_id) or HEX_64.fullmatch(blob_id)
        ):
            raise ValueError(f"{context}: invalid Hub blob identity")
        source = entry.get("source")
        if source is not None:
            if not isinstance(source, dict):
                raise ValueError(f"{context}: source must be an object")
            validate_repository(source.get("repository"), f"{context}: source")
            validate_revision(source.get("revision"), f"{context}: source")
            validate_relative_path(source.get("path"), f"{context}: source")
    return document


def render_component(symbol: str, component_id: str, label: str, lock: dict[str, Any]) -> str:
    repository = lock["repository"]
    revision = lock["revision"]
    external = False
    rows: list[str] = []
    entries = lock["files"]
    if component_id == "gemma4-26b-vision-fp8":
        locked_paths = {entry["path"] for entry in entries}
        if not VISION_RUNTIME_FILES.issubset(locked_paths):
            missing = sorted(VISION_RUNTIME_FILES - locked_paths)
            raise ValueError(f"{component_id}: missing runtime files: {missing}")
        # The immutable publication lock also records component-local notices.
        # They stay in the Hub repository, but the strict runtime module view
        # must contain exactly the four V00 contract files.
        entries = [entry for entry in entries if entry["path"] in VISION_RUNTIME_FILES]
    for entry in entries:
        path = entry["path"]
        source = entry.get("source") or {}
        source_repository = source.get("repository", repository)
        source_revision = source.get("revision", revision)
        source_path = source.get("path", path)
        external |= (
            source_repository != repository
            or source_revision != revision
            or source_path != path
        )
        blob_id = entry.get("lfs_oid") or entry.get("git_oid")
        rows.append(
            "    ModelCatalogFile{"
            f"{quoted(path)}, {int(entry['size'])}ULL, {quoted(entry['sha256'])}, "
            f"{quoted(blob_id)}, {quoted(source_repository)}, "
            f"{quoted(source_revision)}, {quoted(source_path)}"
            "},"
        )
    files = "\n".join(rows)
    view_suffix = lock.get("component", "")
    if not isinstance(view_suffix, str) or (
        view_suffix and PATH_SEGMENT.fullmatch(view_suffix) is None
    ):
        raise ValueError(f"{component_id}: invalid composed-view suffix")
    return f"""inline constexpr std::array k{symbol}Files{{
{files}
}};
inline constexpr ModelComponentCatalog k{symbol}{{
    {quoted(component_id)}, {quoted(label)}, {quoted(repository)},
    {quoted(revision)}, k{symbol}Files, {str(external).lower()},
    {quoted(view_suffix)}}};
"""


def render() -> str:
    components = "\n".join(
        render_component(symbol, component_id, label, load_lock(lock_path))
        for symbol, component_id, label, lock_path in COMPONENTS
    )
    return f"""// Generated by tools/generate_native_studio_model_catalog.py. Do not edit.
#pragma once

#include "model_catalog.h"

#include <array>

namespace gem16::studio::generated {{

{components}
}}  // namespace gem16::studio::generated
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    expected = render()
    if options.check:
        if not options.output.is_file() or options.output.read_text(encoding="utf-8") != expected:
            print(f"generated catalog is stale: {options.output}", file=sys.stderr)
            return 1
        print(f"generated catalog is current: {options.output}")
        return 0
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(expected, encoding="utf-8", newline="\n")
    print(f"wrote {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
