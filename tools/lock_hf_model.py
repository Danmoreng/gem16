#!/usr/bin/env python3
"""Create a complete immutable Hugging Face model lock without executing repo code."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

try:
    from tools.fetch_model import (
        exact_hex,
        hugging_face_repository,
        hugging_face_token,
        immutable_revision,
        safe_relative_path,
    )
except ModuleNotFoundError:  # Direct execution from outside the repository root.
    from fetch_model import (
        exact_hex,
        hugging_face_repository,
        hugging_face_token,
        immutable_revision,
        safe_relative_path,
    )


MAX_API_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_INLINE_BYTES = 64 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terms-url", required=True)
    parser.add_argument("--max-inline-bytes", type=int, default=DEFAULT_MAX_INLINE_BYTES)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def request_headers() -> dict[str, str]:
    headers = {"User-Agent": "gem16-lock-hf-model/1"}
    token = hugging_face_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def read_bounded(response: object, maximum: int) -> bytes:
    payload = response.read(maximum + 1)  # type: ignore[attr-defined]
    if len(payload) > maximum:
        raise RuntimeError(f"response exceeds {maximum} bytes")
    return payload


def git_blob_oid(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def immutable_file_url(repository: str, revision: str, relative: str) -> str:
    quoted = urllib.parse.quote(relative, safe="/")
    return f"https://huggingface.co/{repository}/resolve/{revision}/{quoted}"


def fetch_inline_identity(
    repository: str,
    revision: str,
    relative: str,
    expected_size: int,
    expected_git_oid: str,
    maximum: int,
) -> str:
    if expected_size > maximum:
        raise RuntimeError(
            f"non-LFS file {relative!r} is {expected_size} bytes; limit is {maximum}"
        )
    request = urllib.request.Request(
        immutable_file_url(repository, revision, relative), headers=request_headers()
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = read_bounded(response, maximum)
    if len(payload) != expected_size:
        raise RuntimeError(
            f"size mismatch for {relative!r}: expected {expected_size}, got {len(payload)}"
        )
    observed_git_oid = git_blob_oid(payload)
    if observed_git_oid != expected_git_oid:
        raise RuntimeError(
            f"Git blob mismatch for {relative!r}: expected {expected_git_oid}, "
            f"got {observed_git_oid}"
        )
    return hashlib.sha256(payload).hexdigest()


def fetch_xet_hash(repository: str, revision: str, relative: str) -> str | None:
    request = urllib.request.Request(
        immutable_file_url(repository, revision, relative),
        headers=request_headers(),
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        etag = (response.headers.get("ETag") or "").strip().removeprefix("W/").strip('"')
    if not etag:
        return None
    try:
        return exact_hex(etag, 64, f"Xet ETag for {relative!r}")
    except ValueError:
        return None


def create_entry(
    repository: str,
    revision: str,
    sibling: object,
    max_inline_bytes: int,
) -> dict[str, object]:
    if not isinstance(sibling, dict):
        raise ValueError("Hugging Face sibling entry must be an object")
    relative = safe_relative_path(
        str(sibling.get("rfilename", "")), "Hugging Face repository path"
    ).as_posix()
    size = sibling.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"invalid remote size for {relative!r}")
    git_oid = exact_hex(sibling.get("blobId"), 40, f"Git OID for {relative!r}")
    lfs = sibling.get("lfs")
    entry: dict[str, object] = {"path": relative, "size": size}
    if lfs is None:
        entry["sha256"] = fetch_inline_identity(
            repository, revision, relative, size, git_oid, max_inline_bytes
        )
        entry["git_oid"] = git_oid
        return entry
    if not isinstance(lfs, dict):
        raise ValueError(f"invalid LFS metadata for {relative!r}")
    lfs_size = lfs.get("size")
    if lfs_size != size:
        raise ValueError(
            f"LFS size mismatch for {relative!r}: file={size!r}, lfs={lfs_size!r}"
        )
    sha256 = exact_hex(lfs.get("sha256"), 64, f"LFS SHA-256 for {relative!r}")
    entry["sha256"] = sha256
    entry["git_oid"] = git_oid
    entry["lfs_oid"] = sha256
    xet_hash = fetch_xet_hash(repository, revision, relative)
    if xet_hash is not None:
        entry["xet_hash"] = xet_hash
    return entry


def fetch_model_document(repository: str, revision: str) -> dict[str, object]:
    quoted_repository = urllib.parse.quote(repository, safe="/")
    url = (
        f"https://huggingface.co/api/models/{quoted_repository}/revision/"
        f"{revision}?blobs=true"
    )
    request = urllib.request.Request(url, headers=request_headers())
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = read_bounded(response, MAX_API_BYTES)
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("Hugging Face model response must be an object")
    return document


def build_lock(
    repository: str,
    revision: str,
    terms_url: str,
    max_inline_bytes: int = DEFAULT_MAX_INLINE_BYTES,
) -> dict[str, object]:
    repository = hugging_face_repository(repository, "repository")
    revision = immutable_revision(revision, "requested revision")
    if max_inline_bytes < 0:
        raise ValueError("max inline bytes must be nonnegative")
    document = fetch_model_document(repository, revision)
    resolved = immutable_revision(document.get("sha"), "resolved revision")
    if resolved != revision:
        raise RuntimeError(f"resolved revision differs: requested {revision}, got {resolved}")
    if bool(document.get("private")):
        raise RuntimeError("refusing to create a distributable lock for a private repository")
    siblings = document.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise ValueError("Hugging Face model response has no files")
    files = [
        create_entry(repository, revision, sibling, max_inline_bytes)
        for sibling in siblings
    ]
    files.sort(key=lambda entry: str(entry["path"]))
    if len({str(entry["path"]) for entry in files}) != len(files):
        raise ValueError("Hugging Face model response contains duplicate paths")
    return {
        "schema_version": 2,
        "repository": repository,
        "revision": revision,
        "resolved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_url": f"https://huggingface.co/{repository}/tree/{revision}",
        "terms_url": terms_url,
        "files": files,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing lock: {args.output}")
    lock = build_lock(
        args.repository,
        args.revision,
        args.terms_url,
        args.max_inline_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: files={len(lock['files'])} "
        f"bytes={sum(int(entry['size']) for entry in lock['files'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
