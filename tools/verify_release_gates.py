#!/usr/bin/env python3
"""Fail closed unless both platform artifacts have matching qualification evidence."""

import argparse
import json
from pathlib import Path
import re

from package_server import sha256

GATES = {
    "host",
    "gpu_12b",
    "gpu_compact_vision",
    "nvfp4_regression",
    "sdk_agent",
    "cancellation_saturation",
    "clean_machine",
}
PLATFORMS = {"linux-x64", "windows-x64"}


def local_file(root, name):
    if (
        not isinstance(name, str)
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError("evidence path must be relative and contained")
    path = root / name
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError("symlink in release evidence path")
    path.resolve(strict=True).relative_to(root.resolve(strict=True))
    if not path.is_file():
        raise ValueError("evidence must be a regular file")
    return path


def verify(manifest_path, commit, version):
    manifest = json.loads(manifest_path.read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid commit")
    if (
        manifest["commit"] != commit
        or manifest["version"] != version
        or manifest.get("dirty") is not False
    ):
        raise ValueError("release revision/version/clean-state mismatch")
    platforms = manifest["platforms"]
    if set(platforms) != PLATFORMS:
        raise ValueError("both platforms required")
    root = manifest_path.parent
    for platform, entry in platforms.items():
        archive = local_file(root, entry["artifact"])
        artifact_hash = sha256(archive)
        if artifact_hash != entry["sha256"]:
            raise ValueError("artifact hash mismatch")
        if set(entry["gates"]) != GATES:
            raise ValueError("missing qualification gates")
        for gate, reference in entry["gates"].items():
            evidence_path = local_file(root, reference["path"])
            if sha256(evidence_path) != reference["sha256"]:
                raise ValueError("evidence hash mismatch")
            evidence = json.loads(evidence_path.read_text())
            if (
                evidence.get("status") != "passed"
                or evidence.get("commit") != commit
                or evidence.get("version") != version
                or evidence.get("platform") != platform
                or evidence.get("gate") != gate
                or evidence.get("artifact_sha256") != artifact_hash
            ):
                raise ValueError("qualification does not match release artifact")
            samples = evidence.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError("raw evidence required")
            for sample in samples:
                if sha256(local_file(root, sample["path"])) != sample["sha256"]:
                    raise ValueError("raw sample hash mismatch")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    verify(args.manifest, args.commit, args.version)
    print(
        "release artifact/evidence integrity gates passed; owner publication authorization still required"
    )
