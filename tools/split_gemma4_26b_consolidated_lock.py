#!/usr/bin/env python3
"""Split one immutable 26B Hub revision into installable component locks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

try:
    from tools.fetch_model import validate_lock
except ModuleNotFoundError:  # Direct execution from outside the repository root.
    from fetch_model import validate_lock


REPOSITORY = "danmoreng/gemma-4-26B-A4B-it-GEM16"
COMPONENTS = {
    "target": ("", "gemma4-26b-gem16-target.lock.json"),
    "trellis35": ("trellis35/", "gemma4-26b-trellis35-target.lock.json"),
    "assistant": ("assistant/", "gemma4-26b-gem16-assistant.lock.json"),
    "vision": ("vision/", "gemma4-26b-vision-fp8.lock.json"),
}
REQUIRED_COMPONENT_FILES = {
    "target": {"model.gem16", "gem16_model.json", "gem16_components.json"},
    "trellis35": {
        "model.gem16",
        "gem16_model.json",
        "gem16_compilation.json",
        "gem16.lock.json",
        "SHA256SUMS",
    },
    "assistant": {
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        "gem16_model.json",
    },
    "vision": {
        "vision.gem16",
        "gem16_vision.json",
        "vision_compilation.json",
        "vision.lock.json",
    },
}
LEGACY_TRELLIS_REQUIRED_FILES = {
    "non-routed.gem16",
    "trellis35-checkpoint.json",
    "trellis35-experts.json",
    "SHA256SUMS",
}


class SplitError(RuntimeError):
    pass


def load_lock(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_lock(document)
    if document.get("repository") != REPOSITORY:
        raise SplitError(f"unexpected consolidated repository: {document.get('repository')}")
    return document


def selected_entries(lock: dict[str, Any], component: str,
                     prefix: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for original in lock["files"]:
        remote_path = original["path"]
        if prefix:
            if not remote_path.startswith(prefix):
                continue
            relative = remote_path.removeprefix(prefix)
        else:
            if "/" in remote_path:
                continue
            relative = remote_path
        if not relative or relative.startswith("/"):
            raise SplitError(f"invalid {component} component path: {remote_path}")
        entry = dict(original)
        entry["path"] = relative
        if prefix:
            entry["source"] = {
                "repository": lock["repository"],
                "revision": lock["revision"],
                "path": remote_path,
            }
        result.append(entry)
    result.sort(key=lambda entry: entry["path"])
    paths = {entry["path"] for entry in result}
    required = REQUIRED_COMPONENT_FILES[component]
    if component == "trellis35" and "model.gem16" not in paths:
        required = LEGACY_TRELLIS_REQUIRED_FILES
    missing = required - paths
    if missing:
        raise SplitError(
            f"{component} component is missing required files: {sorted(missing)}"
        )
    if len(paths) != len(result):
        raise SplitError(f"{component} component contains duplicate paths")
    return result


def component_lock(lock: dict[str, Any], component: str,
                   prefix: str) -> dict[str, Any]:
    result = {
        "schema_version": 2,
        "repository": lock["repository"],
        "revision": lock["revision"],
        "resolved_at_utc": lock["resolved_at_utc"],
        "source_url": lock["source_url"],
        "terms_url": lock["terms_url"],
        "component": component,
        "component_path": prefix.removesuffix("/") or ".",
        "files": selected_entries(lock, component, prefix),
    }
    validate_lock(result)
    return result


def write_lock(path: Path, value: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise SplitError(f"refusing to overwrite existing lock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-lock", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    options = parser.parse_args()
    lock = load_lock(options.full_lock)
    known_prefixes = tuple(prefix for prefix, _name in COMPONENTS.values() if prefix)
    unknown = [
        entry["path"]
        for entry in lock["files"]
        if "/" in entry["path"]
        and not entry["path"].startswith(known_prefixes)
    ]
    if unknown:
        raise SplitError(f"unassigned repository paths: {unknown[:5]}")
    for component, (prefix, filename) in COMPONENTS.items():
        value = component_lock(lock, component, prefix)
        destination = options.output_directory / filename
        write_lock(destination, value, options.force)
        print(
            f"wrote {destination}: files={len(value['files'])} "
            f"bytes={sum(entry['size'] for entry in value['files'])}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError, SplitError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(4)
