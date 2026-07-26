#!/usr/bin/env python3
"""Download and verify an immutable Hugging Face snapshot without executing repo code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import urllib.error
import urllib.parse
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("models/gemma4-12b-nvfp4.lock.json"))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe_target(root: Path, relative: str) -> Path:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or len(parsed.parts) != 1:
        raise ValueError(f"unsafe checkpoint path in lock: {relative!r}")
    return root / parsed.name


def source_path(relative: str) -> str:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
        raise ValueError(f"unsafe source path in lock: {relative!r}")
    return parsed.as_posix()


def immutable_revision(value: object, description: str) -> str:
    revision = str(value)
    if len(revision) != 40 or revision == "main" or any(
        character not in "0123456789abcdefABCDEF" for character in revision
    ):
        raise ValueError(f"{description} must be a full immutable commit SHA")
    return revision


def resolve_source(
    lock: dict[str, object], entry: dict[str, object]
) -> tuple[str, str, str]:
    source = entry.get("source")
    if source is None:
        source = {}
    if not isinstance(source, dict):
        raise ValueError(f"source for {entry.get('path')!r} must be an object")
    repository = str(source.get("repository", lock["repository"]))
    if not repository or repository.startswith("/") or repository.endswith("/"):
        raise ValueError(f"invalid Hugging Face repository: {repository!r}")
    revision = immutable_revision(
        source.get("revision", lock["revision"]),
        f"source revision for {entry.get('path')!r}",
    )
    relative = source_path(str(source.get("path", entry["path"])))
    return repository, revision, relative


def verify(path: Path, entry: dict[str, object]) -> bool:
    expected_size = int(entry["size"])
    expected_hash = str(entry["sha256"])
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    actual = digest(path)
    if actual != expected_hash:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected_hash}, got {actual}")
    print(f"verified {path.name} ({expected_size} bytes)")
    return True


def download(
    url: str, destination: Path, expected_size: int, expected_hash: str
) -> None:
    partial = destination.with_name(destination.name + ".part")
    partial_metadata = destination.with_name(destination.name + ".part.json")
    identity = {
        "url": url,
        "size": expected_size,
        "sha256": expected_hash,
    }
    identity_text = json.dumps(identity, sort_keys=True) + "\n"
    metadata_matches = False
    if partial_metadata.is_file():
        metadata_matches = partial_metadata.read_text(encoding="utf-8") == identity_text
    if partial.exists() and not metadata_matches:
        partial.unlink()
    if partial_metadata.exists() and not metadata_matches:
        partial_metadata.unlink()
    if not partial.exists():
        partial_metadata.write_text(identity_text, encoding="utf-8")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise RuntimeError(f"partial file is larger than lock size: {partial}")
    headers = {"User-Agent": "gem16gb-fetch-model/1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    mode = "ab" if offset else "wb"
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open(mode) as output:
            if offset and response.status != 206:
                raise RuntimeError(f"server refused resume for {destination.name}")
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                offset += len(block)
                print(f"\r{destination.name}: {offset}/{expected_size} bytes", end="", flush=True)
    except urllib.error.URLError as error:
        raise RuntimeError(f"download failed for {destination.name}: {error}") from error
    print()
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch after download for {destination.name}")
    actual_hash = digest(partial)
    if actual_hash != expected_hash:
        partial.unlink()
        partial_metadata.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for downloaded {destination.name}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    partial.replace(destination)
    partial_metadata.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    schema_version = int(lock.get("schema_version", 0))
    if schema_version not in (1, 2):
        raise ValueError(f"unsupported model lock schema_version: {schema_version}")
    immutable_revision(lock["revision"], "model lock revision")
    args.destination.mkdir(parents=True, exist_ok=True)
    failures = 0
    for entry in lock["files"]:
        target = safe_target(args.destination, str(entry["path"]))
        try:
            if verify(target, entry):
                continue
            if args.verify_only:
                print(f"missing or wrong size: {target}", file=sys.stderr)
                failures += 1
                continue
            repository, revision, relative = resolve_source(lock, entry)
            quoted_path = urllib.parse.quote(relative, safe="/")
            url = f"https://huggingface.co/{repository}/resolve/{revision}/{quoted_path}"
            download(url, target, int(entry["size"]), str(entry["sha256"]))
            if not verify(target, entry):
                raise RuntimeError(f"verification unexpectedly failed: {target}")
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
