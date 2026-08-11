#!/usr/bin/env python3
"""Write a deterministic tensor inventory from bounded Safetensors headers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from tools.compare_manifests import build_reference_manifest, load_json
    from tools.fetch_model import validate_lock
except ModuleNotFoundError:  # Direct execution from outside the repository root.
    from compare_manifests import build_reference_manifest, load_json
    from fetch_model import validate_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source-family", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_inventory(
    model: Path,
    lock_path: Path,
    source_family: str,
) -> dict[str, object]:
    lock = load_json(lock_path)
    entries = validate_lock(lock)
    reference = build_reference_manifest(model)
    aliases: dict[tuple[str, int, int], list[str]] = {}
    for name, tensor in reference.items():
        key = (
            str(tensor["source_shard"]),
            int(tensor["byte_offset"]),
            int(tensor["byte_length"]),
        )
        aliases.setdefault(key, []).append(name)
    tensors = []
    for name in sorted(reference):
        tensor = reference[name]
        key = (
            str(tensor["source_shard"]),
            int(tensor["byte_offset"]),
            int(tensor["byte_length"]),
        )
        peers = sorted(peer for peer in aliases[key] if peer != name)
        tensors.append(
            {
                "name": name,
                "dtype": tensor["storage_dtype"],
                "shape": tensor["shape"],
                "bytes": tensor["byte_length"],
                "shard": tensor["source_shard"],
                "absolute_offset": tensor["byte_offset"],
                "aliases": peers,
            }
        )
    return {
        "schema_version": 1,
        "status": "raw_source_inventory",
        "source_family": source_family,
        "source": {
            "repository": lock["repository"],
            "revision": lock["revision"],
            "lock_path": lock_path.as_posix(),
            "lock_sha256": file_sha256(lock_path),
            "locked_file_count": len(entries),
            "locked_repository_bytes": sum(int(entry["size"]) for entry in entries),
        },
        "checkpoint": {
            "tensor_count": len(tensors),
            "tensor_payload_bytes": sum(int(tensor["bytes"]) for tensor in tensors),
            "shards": sorted({str(tensor["shard"]) for tensor in tensors}),
        },
        "tensors": tensors,
    }


def main() -> int:
    args = parse_args()
    inventory = make_inventory(args.model, args.lock, args.source_family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: {inventory['checkpoint']['tensor_count']} tensors, "
        f"{inventory['checkpoint']['tensor_payload_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
