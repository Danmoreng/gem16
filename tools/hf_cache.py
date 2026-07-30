#!/usr/bin/env python3
"""Resolve gem16's immutable checkpoints inside the shared Hugging Face cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
TARGET_LOCK = ROOT / "models/gemma4-12b-nvfp4.lock.json"
ASSISTANT_LOCK = ROOT / "models/gemma4-12b-mtp-assistant.lock.json"


def hub_cache_root(
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    env = os.environ if environment is None else environment
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    if env.get("XDG_CACHE_HOME"):
        return Path(env["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    return (Path.home() if home is None else home) / ".cache" / "huggingface" / "hub"


def repository_root(repository: str, cache_root: Path | None = None) -> Path:
    return (cache_root or hub_cache_root()) / f"models--{repository.replace('/', '--')}"


def snapshot_path(
    repository: str,
    revision: str,
    cache_root: Path | None = None,
) -> Path:
    return repository_root(repository, cache_root) / "snapshots" / revision


def load_lock(lock_path: Path) -> dict[str, object]:
    return json.loads(lock_path.read_text(encoding="utf-8"))


def locked_snapshot_path(lock_path: Path, cache_root: Path | None = None) -> Path:
    lock = load_lock(lock_path)
    repository = str(lock["repository"])
    revision = str(lock["revision"])
    has_external_file = any(
        isinstance(entry.get("source"), dict)
        and (
            entry["source"].get("repository", repository) != repository
            or entry["source"].get("revision", revision) != revision
            or entry["source"].get("path", entry["path"]) != entry["path"]
        )
        for entry in lock["files"]
    )
    if not has_external_file:
        return snapshot_path(repository, revision, cache_root)
    root = cache_root or hub_cache_root()
    name = f"{repository.replace('/', '--')}--{revision}"
    return root / ".gem16" / "snapshots" / name


def default_target_model() -> Path:
    return locked_snapshot_path(TARGET_LOCK)


def default_assistant_model() -> Path:
    return locked_snapshot_path(ASSISTANT_LOCK)
