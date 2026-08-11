#!/usr/bin/env python3
"""Regenerate and verify the repository-maintained Gemma 4 26B plan metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote


REPOSITORY_ANCHOR = "Danmoreng/gem16@1c4287965d318ba32a68e597f9d7b6678b883376"
EXTERNAL_REFERENCE_ANCHOR = "kekzl/imp@a392904d4216388828d0d56317de046f4ca49627"
PACKAGE = "gem16-26b-codex-implementation-plan-v4-repository-maintained"
GENERATED_DATE = "2026-08-11"
SCHEMA_VERSION = 4
METADATA_PATHS = {
    "PACKAGE_INTEGRITY.md",
    "PACKAGE_MANIFEST.json",
    "PACKAGE_MANIFEST.md",
    "SHA256SUMS.txt",
    "VALIDATION_REPORT.json",
}
EXPECTED_MILESTONES = {f"M{index:02d}" for index in range(26)}
REQUIRED_PATHS = {
    "00_MASTER_IMPLEMENTATION_PLAN.md",
    "02_AGENT_OPERATING_CONTRACT.md",
    "INDEX.md",
    "MILESTONE_STATUS_BOARD.md",
    "README_DE.md",
    "START_HERE_CODEX.md",
    "references/imp/README.md",
}
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def relative_files(plan_root: Path) -> list[Path]:
    return sorted(
        (path.relative_to(plan_root) for path in plan_root.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix(),
    )


def authored_manifest(plan_root: Path) -> tuple[dict[str, object], bytes]:
    files = []
    for relative_path in relative_files(plan_root):
        name = relative_path.as_posix()
        if name in METADATA_PATHS:
            continue
        data = (plan_root / relative_path).read_bytes()
        files.append({"bytes": len(data), "path": name, "sha256": sha256(data)})

    manifest = {
        "authored_file_count": len(files),
        "external_reference_anchor": EXTERNAL_REFERENCE_ANCHOR,
        "files": files,
        "generated_date": GENERATED_DATE,
        "package": PACKAGE,
        "repository_anchor": REPOSITORY_ANCHOR,
        "schema_version": SCHEMA_VERSION,
    }

    lines = [
        "# Package manifest",
        "",
        f"**Generated:** {GENERATED_DATE}",
        f"**Package:** `{PACKAGE}`",
        f"**Repository anchor:** `{REPOSITORY_ANCHOR}`",
        f"**External reference:** `{EXTERNAL_REFERENCE_ANCHOR}`",
        "",
        "This table covers authored plan files before package-integrity metadata. SHA-256 values are for the exact",
        "UTF-8/file bytes in this repository-maintained plan.",
        "",
        "| Path | Bytes | SHA-256 |",
        "|---|---:|---|",
    ]
    for entry in files:
        lines.append(
            f"| `{entry['path']}` | {entry['bytes']:,} | `{entry['sha256']}` |"
        )
    lines.append("")
    return manifest, "\n".join(lines).encode("utf-8")


def markdown_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = re.split(r"\s+[\"']", target, maxsplit=1)[0]
    return unquote(target.split("#", maxsplit=1)[0])


def source_validation(plan_root: Path) -> dict[str, object]:
    files = relative_files(plan_root)
    markdown_files = [path for path in files if path.suffix.lower() == ".md"]
    empty_markdown: list[str] = []
    markdown_without_heading: list[str] = []
    utf8_errors: list[str] = []
    broken_relative_links: list[dict[str, str]] = []
    markdown_link_count = 0

    for relative_path in markdown_files:
        path = plan_root / relative_path
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            utf8_errors.append(relative_path.as_posix())
            continue
        if not text.strip():
            empty_markdown.append(relative_path.as_posix())
        if not re.search(r"(?m)^#\s+\S", text):
            markdown_without_heading.append(relative_path.as_posix())
        for match in MARKDOWN_LINK_RE.finditer(text):
            markdown_link_count += 1
            raw_target = match.group(1)
            target = markdown_target(raw_target)
            if not target or raw_target.lstrip().startswith("#") or URL_SCHEME_RE.match(target):
                continue
            destination = (path.parent / Path(target)).resolve()
            if not destination.exists():
                broken_relative_links.append(
                    {"path": relative_path.as_posix(), "target": raw_target}
                )

    milestone_counts: dict[str, int] = {}
    for relative_path in files:
        if relative_path.parent.as_posix() != "milestones" or relative_path.suffix != ".md":
            continue
        match = re.match(r"(M\d{2})_", relative_path.name)
        if match:
            milestone_id = match.group(1)
            milestone_counts[milestone_id] = milestone_counts.get(milestone_id, 0) + 1

    milestone_ids = set(milestone_counts)
    duplicate_milestones = sorted(
        milestone_id for milestone_id, count in milestone_counts.items() if count != 1
    )
    missing_milestones = sorted(EXPECTED_MILESTONES - milestone_ids)
    extra_milestones = sorted(milestone_ids - EXPECTED_MILESTONES)
    missing_required_files = sorted(
        path for path in REQUIRED_PATHS if not (plan_root / path).exists()
    )
    passed = not any(
        (
            broken_relative_links,
            duplicate_milestones,
            empty_markdown,
            extra_milestones,
            markdown_without_heading,
            missing_milestones,
            missing_required_files,
            utf8_errors,
        )
    )

    return {
        "broken_relative_links": broken_relative_links,
        "duplicate_milestones": duplicate_milestones,
        "empty_markdown": empty_markdown,
        "extra_milestones": extra_milestones,
        "file_count_before_checksum": len([path for path in files if path.as_posix() != "SHA256SUMS.txt"]),
        "markdown_file_count": len(markdown_files),
        "markdown_link_count": markdown_link_count,
        "markdown_without_heading": markdown_without_heading,
        "milestone_ids": sorted(milestone_ids),
        "missing_milestones": missing_milestones,
        "missing_required_files": missing_required_files,
        "source_validation_passed": passed,
        "utf8_errors": utf8_errors,
    }


def expected_outputs(plan_root: Path) -> dict[str, bytes]:
    manifest, manifest_markdown = authored_manifest(plan_root)
    report = {
        "external_reference_anchor": EXTERNAL_REFERENCE_ANCHOR,
        "final_markdown_file_count": len(list(plan_root.rglob("*.md"))),
        "final_package_file_count": len(relative_files(plan_root)),
        "generated_date": GENERATED_DATE,
        "package": PACKAGE,
        "repository_anchor": REPOSITORY_ANCHOR,
        "schema_version": SCHEMA_VERSION,
        "source_tree": source_validation(plan_root),
        "zip_validation": {
            "note": "This repository-maintained revision is validated in place; SHA256SUMS and source-tree validation are authoritative.",
            "status": "not_applicable",
        },
    }
    outputs = {
        "PACKAGE_MANIFEST.json": json_bytes(manifest),
        "PACKAGE_MANIFEST.md": manifest_markdown,
        "VALIDATION_REPORT.json": json_bytes(report),
    }

    checksum_lines = []
    for relative_path in relative_files(plan_root):
        name = relative_path.as_posix()
        if name == "SHA256SUMS.txt":
            continue
        data = outputs.get(name, (plan_root / relative_path).read_bytes())
        checksum_lines.append(f"{sha256(data)}  {name}")
    outputs["SHA256SUMS.txt"] = ("\n".join(checksum_lines) + "\n").encode("utf-8")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated metadata without writing")
    args = parser.parse_args()

    plan_root = repository_root() / "docs" / "plans" / "gemma4-26b"
    outputs = expected_outputs(plan_root)
    stale = [
        name
        for name, expected in outputs.items()
        if not (plan_root / name).exists() or (plan_root / name).read_bytes() != expected
    ]
    if args.check:
        if stale:
            print("stale generated plan metadata: " + ", ".join(stale), file=sys.stderr)
            return 1
    else:
        for name, data in outputs.items():
            (plan_root / name).write_bytes(data)

    report = json.loads(outputs["VALIDATION_REPORT.json"])
    if not report["source_tree"]["source_validation_passed"]:
        print(json.dumps(report["source_tree"], ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(f"Gemma 4 26B plan metadata {'verified' if args.check else 'regenerated'}." )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
