#!/usr/bin/env python3
"""Download and verify an immutable Hugging Face snapshot without executing repo code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request

try:
    from tools.hf_cache import hub_cache_root, locked_snapshot_path, repository_root
except ModuleNotFoundError:  # Direct execution from outside the repository root.
    from hf_cache import hub_cache_root, locked_snapshot_path, repository_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("models/gemma4-12b-nvfp4.lock.json"))
    parser.add_argument(
        "--destination",
        type=Path,
        help="optional extra linked view; defaults to the shared Hugging Face cache",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="override HF_HUB_CACHE for this invocation",
    )
    parser.add_argument(
        "--import-from",
        type=Path,
        help="import an existing verified checkpoint directory instead of downloading",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate lock and capacity without downloading or linking files",
    )
    parser.add_argument(
        "--max-new-bytes",
        type=nonnegative_int,
        help="reject before download when missing cache blobs exceed this budget",
    )
    parser.add_argument(
        "--min-free-bytes",
        type=nonnegative_int,
        default=0,
        help="required free filesystem bytes after all missing blobs are downloaded",
    )
    return parser.parse_args()


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe_relative_path(relative: str, description: str) -> PurePosixPath:
    if not relative or "\\" in relative or "\x00" in relative:
        raise ValueError(f"unsafe {description}: {relative!r}")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or any(
        part in ("", ".", "..") for part in parsed.parts
    ):
        raise ValueError(f"unsafe {description}: {relative!r}")
    return parsed


def safe_target(root: Path, relative: str) -> Path:
    parsed = safe_relative_path(relative, "checkpoint path in lock")
    resolved_root = root.expanduser().resolve()
    target = resolved_root.joinpath(*parsed.parts)
    current = resolved_root
    for part in parsed.parts[:-1]:
        current = current / part
        if current.exists() or current.is_symlink():
            try:
                current.resolve(strict=True).relative_to(resolved_root)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"checkpoint path escapes through a symlink: {relative!r}"
                ) from error
    return target


def source_path(relative: str) -> str:
    return safe_relative_path(relative, "source path in lock").as_posix()


def hugging_face_repository(value: object, description: str) -> str:
    repository = str(value)
    parts = repository.split("/")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if (
        len(parts) != 2
        or any(not part or part in (".", "..") for part in parts)
        or any(character not in allowed for part in parts for character in part)
    ):
        raise ValueError(f"invalid Hugging Face {description}: {repository!r}")
    return repository


def exact_hex(value: object, length: int, description: str) -> str:
    text = str(value)
    if len(text) != length or any(
        character not in "0123456789abcdefABCDEF" for character in text
    ):
        raise ValueError(f"{description} must be exactly {length} hexadecimal characters")
    return text.lower()


def immutable_revision(value: object, description: str) -> str:
    try:
        return exact_hex(value, 40, description)
    except ValueError as error:
        raise ValueError(f"{description} must be a full immutable commit SHA") from error


def validate_lock(lock: object) -> list[dict[str, object]]:
    if not isinstance(lock, dict):
        raise ValueError("model lock must be an object")
    schema_version = int(lock.get("schema_version", 0))
    if schema_version not in (1, 2):
        raise ValueError(f"unsupported model lock schema_version: {schema_version}")
    hugging_face_repository(lock.get("repository", ""), "repository")
    immutable_revision(lock.get("revision"), "model lock revision")
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("model lock files must be a non-empty array")
    validated: list[dict[str, object]] = []
    paths: set[str] = set()
    for index, candidate in enumerate(files):
        if not isinstance(candidate, dict):
            raise ValueError(f"model lock file {index} must be an object")
        relative = source_path(str(candidate.get("path", "")))
        if relative in paths:
            raise ValueError(f"duplicate model lock path: {relative!r}")
        paths.add(relative)
        size = candidate.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid size for model lock path {relative!r}")
        exact_hex(candidate.get("sha256"), 64, f"SHA-256 for {relative!r}")
        if candidate.get("git_oid") is not None:
            exact_hex(candidate["git_oid"], 40, f"Git OID for {relative!r}")
        if candidate.get("lfs_oid") is not None:
            lfs_oid = exact_hex(candidate["lfs_oid"], 64, f"LFS OID for {relative!r}")
            if lfs_oid != str(candidate["sha256"]).lower():
                raise ValueError(f"LFS OID differs from SHA-256 for {relative!r}")
        if candidate.get("xet_hash") is not None:
            exact_hex(candidate["xet_hash"], 64, f"Xet hash for {relative!r}")
        resolve_source(lock, candidate)
        blob_id(candidate)
        validated.append(candidate)
    return validated


def resolve_source(
    lock: dict[str, object], entry: dict[str, object]
) -> tuple[str, str, str]:
    source = entry.get("source")
    if source is None:
        source = {}
    if not isinstance(source, dict):
        raise ValueError(f"source for {entry.get('path')!r} must be an object")
    repository = hugging_face_repository(
        source.get("repository", lock["repository"]),
        f"source repository for {entry.get('path')!r}",
    )
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


def blob_id(entry: dict[str, object]) -> str:
    value = entry.get("lfs_oid") or entry.get("git_oid")
    if not value:
        raise ValueError(f"lock entry has no blob identity: {entry.get('path')!r}")
    return str(value)


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and source.samefile(destination):
        return
    destination.unlink(missing_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def verification_marker(repository: str, identity: str, cache_root: Path) -> Path:
    return repository_root(repository, cache_root) / ".gem16-verified" / f"{identity}.sha256"


def missing_blob_bytes(
    lock: dict[str, object], entries: list[dict[str, object]], cache_root: Path
) -> int:
    missing = 0
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        repository, _, _ = resolve_source(lock, entry)
        identity = blob_id(entry)
        key = (repository, identity)
        if key in seen:
            continue
        seen.add(key)
        blob = repository_root(repository, cache_root) / "blobs" / identity
        if not blob.is_file() or blob.stat().st_size != int(entry["size"]):
            missing += int(entry["size"])
    return missing


def capacity_preflight(
    lock: dict[str, object],
    entries: list[dict[str, object]],
    cache_root: Path,
    max_new_bytes: int | None,
    min_free_bytes: int,
    available_bytes: int | None = None,
) -> int:
    required = missing_blob_bytes(lock, entries, cache_root)
    if max_new_bytes is not None and required > max_new_bytes:
        raise RuntimeError(
            f"capacity preflight rejected {required} missing bytes; "
            f"configured maximum is {max_new_bytes}"
        )
    available = shutil.disk_usage(cache_root).free if available_bytes is None else available_bytes
    if required + min_free_bytes > available:
        raise RuntimeError(
            f"capacity preflight rejected {required} missing bytes with "
            f"{min_free_bytes} bytes reserved; only {available} bytes are free"
        )
    print(
        f"capacity preflight: missing={required} free={available} "
        f"required_reserve={min_free_bytes}"
    )
    return required


def install_entry(
    lock: dict[str, object],
    entry: dict[str, object],
    cache_root: Path,
    import_root: Path | None,
    verify_only: bool,
) -> bool:
    repository, revision, relative = resolve_source(lock, entry)
    identity = blob_id(entry)
    blob = repository_root(repository, cache_root) / "blobs" / identity
    blob.parent.mkdir(parents=True, exist_ok=True)
    if not verify(blob, entry):
        if verify_only:
            print(f"missing or wrong size: {blob}", file=sys.stderr)
            return False
        if import_root is not None:
            imported = safe_target(import_root, str(entry["path"]))
            if not verify(imported, entry):
                raise RuntimeError(f"missing or invalid import source: {imported}")
            blob.with_name(blob.name + ".incomplete").unlink(missing_ok=True)
            link_file(imported, blob)
        else:
            quoted_path = urllib.parse.quote(relative, safe="/")
            url = f"https://huggingface.co/{repository}/resolve/{revision}/{quoted_path}"
            download(url, blob, int(entry["size"]), str(entry["sha256"]))
        if not verify(blob, entry):
            raise RuntimeError(f"verification unexpectedly failed: {blob}")

    marker = verification_marker(repository, identity, cache_root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{entry['sha256']}\n", encoding="utf-8")
    source_snapshot = repository_root(repository, cache_root) / "snapshots" / revision
    link_file(blob, safe_target(source_snapshot, relative))
    return True


def download(
    url: str, destination: Path, expected_size: int, expected_hash: str
) -> None:
    partial = destination.with_name(destination.name + ".incomplete")
    partial_metadata = destination.with_name(destination.name + ".incomplete.json")
    identity = {
        "url": url,
        "size": expected_size,
        "sha256": expected_hash,
    }
    identity_text = json.dumps(identity, sort_keys=True) + "\n"
    # Compose writes the same `<blob>.incomplete` file without sidecar metadata.
    # A sidecar is therefore optional; when present it must match this exact lock.
    metadata_matches = not partial_metadata.exists()
    if partial_metadata.is_file():
        metadata_matches = partial_metadata.read_text(encoding="utf-8") == identity_text
    if partial.exists() and not metadata_matches:
        partial.unlink()
    if partial_metadata.exists() and not metadata_matches:
        partial_metadata.unlink()
    if not partial_metadata.exists():
        partial_metadata.write_text(identity_text, encoding="utf-8")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise RuntimeError(f"partial file is larger than lock size: {partial}")
    headers = {"User-Agent": "gem16-fetch-model/1"}
    token = hugging_face_token()
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


def hugging_face_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    token_file = hf_home.expanduser() / "token"
    if token_file.is_file():
        value = token_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def main() -> int:
    args = parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    entries = validate_lock(lock)
    cache_root = (args.cache_dir or hub_cache_root()).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        capacity_preflight(
            lock,
            entries,
            cache_root,
            args.max_new_bytes,
            args.min_free_bytes,
        )
    if args.preflight_only:
        return 0
    locked_view = locked_snapshot_path(args.lock, cache_root)
    import_root = args.import_from.resolve(strict=True) if args.import_from else None
    failures = 0
    for entry in entries:
        try:
            if not install_entry(
                lock,
                entry,
                cache_root,
                import_root,
                args.verify_only,
            ):
                failures += 1
                continue
            repository, _, _ = resolve_source(lock, entry)
            blob = repository_root(repository, cache_root) / "blobs" / blob_id(entry)
            link_file(blob, safe_target(locked_view, str(entry["path"])))
            if args.destination is not None:
                link_file(blob, safe_target(args.destination, str(entry["path"])))
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            failures += 1
    if failures == 0:
        print(f"locked snapshot: {locked_view}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
