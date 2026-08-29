#!/usr/bin/env python3
"""Create a compact ChatGPT Pro review snapshot of the current worktree.

The archive intentionally contains project source, tests, documentation, compact
performance evidence, and curated reference-engine source. Large model files,
build trees, raw profiler captures, caches, and generated binaries are excluded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "review-archives"
BINARY_SUFFIXES = {
    ".a", ".bin", ".class", ".dll", ".exe", ".gguf", ".jar", ".jpeg",
    ".jpg", ".lib", ".ncu-rep", ".npy", ".npz", ".nsys-rep", ".o",
    ".obj", ".partial", ".png", ".pyc", ".safetensors", ".sha256", ".so",
    ".sqlite", ".svg", ".webp", ".zip",
}
SKIP_DIRS = {
    ".git", ".gradle", ".kotlin", "__pycache__", "build", "node_modules",
    "review-archives",
}
PROJECT_TREES = (
    "gradle", "src", "include", "tests", "tools", "scripts", "cmake",
    "examples", "toolchains", "docs", "studioApp/src",
    "benchmarks/baselines", "benchmarks/corpora", "benchmarks/prompts",
    "benchmarks/quality", "benchmarks/schemas", "nativeStudio/src",
    "nativeStudio/tests", "nativeStudio/licenses",
)
ROOT_FILES = (
    "AGENTS.md", "README.md", "LICENSE", "VERSION", "CMakeLists.txt",
    "CMakePresets.json", "build.gradle.kts", "settings.gradle.kts", "gradlew",
    "gradlew.bat", "nativeStudio/CMakeLists.txt", "nativeStudio/THIRD_PARTY.md",
)
DEFAULT_RAW_EVIDENCE_DIRS = {
    "m25", "main-promotion-bd5b1af", "max-performance",
    "external-mtp-20260825", "perf-target", "vllm",
}
TEXT_EVIDENCE_SUFFIXES = {".json", ".csv", ".md", ".txt"}


def run_git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="archive directory (default: review-archives/)",
    )
    parser.add_argument(
        "--max-size-mib", type=float, default=10.0,
        help="fail and remove the archive above this size (default: 10)",
    )
    parser.add_argument(
        "--focus", action="append", default=[],
        help="review focus bullet embedded in REVIEW_REQUEST.md; repeatable",
    )
    parser.add_argument(
        "--raw-evidence-dir", action="append", default=[],
        help="additional artifacts/raw subdirectory to include compact text from",
    )
    parser.add_argument(
        "--no-reference-engines", action="store_true",
        help="omit curated llama.cpp and vLLM reference source",
    )
    return parser.parse_args()


def version_tuple(path: Path) -> tuple[int, ...]:
    match = re.search(r"vllm-([0-9]+(?:\.[0-9]+)*)-env$", path.name)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def find_vllm_package() -> tuple[Path | None, str | None]:
    candidates = sorted(
        (p for p in (ROOT / "third_party/cache").glob("vllm-*-env") if p.is_dir()),
        key=version_tuple,
        reverse=True,
    )
    for environment in candidates:
        packages = sorted(environment.glob("lib/python*/site-packages/vllm"))
        if packages:
            version = ".".join(str(part) for part in version_tuple(environment))
            return packages[-1], version
    return None, None


def should_skip(path: Path) -> bool:
    return path.suffix.lower() in BINARY_SUFFIXES


class ArchiveBuilder:
    def __init__(self, archive: zipfile.ZipFile) -> None:
        self.archive = archive
        self.names: set[str] = set()

    def add_file(self, source: Path, name: str | None = None) -> bool:
        if source.is_symlink() or not source.is_file() or should_skip(source):
            return False
        archive_name = (name or source.relative_to(ROOT).as_posix()).lstrip("/")
        if archive_name in self.names:
            return False
        self.archive.write(source, archive_name)
        self.names.add(archive_name)
        return True

    def add_text(self, name: str, content: str) -> None:
        if name in self.names:
            raise RuntimeError(f"duplicate archive entry: {name}")
        self.archive.writestr(name, content)
        self.names.add(name)

    def add_tree(self, source: Path, archive_prefix: str) -> int:
        if not source.is_dir():
            return 0
        count = 0
        for current, dirs, files in os.walk(source):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            current_path = Path(current)
            for filename in sorted(files):
                path = current_path / filename
                relative = path.relative_to(source).as_posix()
                if self.add_file(path, f"{archive_prefix}/{relative}"):
                    count += 1
        return count


def add_compact_artifacts(builder: ArchiveBuilder, raw_dirs: set[str]) -> None:
    artifacts = ROOT / "artifacts"
    if not artifacts.is_dir():
        return
    for path in sorted(artifacts.iterdir()):
        if path.name == "raw":
            continue
        if path.is_file():
            builder.add_file(path)
        elif path.is_dir():
            builder.add_tree(path, f"artifacts/{path.name}")

    raw = artifacts / "raw"
    for directory in sorted(raw_dirs):
        source = raw / directory
        if not source.is_dir():
            continue
        for current, dirs, files in os.walk(source):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for filename in sorted(files):
                path = Path(current) / filename
                if (path.suffix.lower() in TEXT_EVIDENCE_SUFFIXES and
                        path.stat().st_size < 2_000_000):
                    builder.add_file(path)


def add_llama_reference(builder: ArchiveBuilder) -> str:
    llama = ROOT / "third_party/cache/llama.cpp"
    if not llama.is_dir():
        return "not found"
    for relative in ("src", "common", "ggml/include", "ggml/src/ggml-cuda"):
        builder.add_tree(llama / relative, f"reference/llama.cpp/{relative}")
    for relative in (
        "ggml/src/ggml.c", "ggml/src/ggml.cpp", "ggml/src/ggml-quants.c",
        "ggml/src/ggml-quants.h",
    ):
        builder.add_file(llama / relative, f"reference/llama.cpp/{relative}")
    result = subprocess.run(
        ["git", "-C", str(llama), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    return result.stdout.strip() or "unknown"


def add_vllm_reference(builder: ArchiveBuilder) -> str:
    package, version = find_vllm_package()
    if package is None or version is None:
        return "not found"
    prefix = f"reference/vllm-{version}"
    for relative in ("model_executor/layers", "v1/worker", "v1/core", "config"):
        source = package / relative
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*.py")):
            if "__pycache__" not in path.parts:
                builder.add_file(path, f"{prefix}/{path.relative_to(package).as_posix()}")
    for relative in (
        "__init__.py", "sequence.py", "outputs.py", "sampling_params.py",
        "envs.py", "forward_context.py", "_custom_ops.py", "logprobs.py",
        "version.py",
    ):
        builder.add_file(package / relative, f"{prefix}/{relative}")
    return version


def review_request(focus: list[str], commit: str, dirty: bool) -> str:
    bullets = focus or [
        "Review correctness, security, maintainability, and regression risk.",
        "Perform static performance analysis without changing model semantics.",
        "Check evidence and claims against the included active contracts.",
    ]
    lines = [
        "# ChatGPT Pro review request", "", f"Commit: `{commit}`",
        f"Worktree dirty: `{'yes' if dirty else 'no'}`", "", "## Focus", "",
    ]
    lines.extend(f"- {bullet}" for bullet in bullets)
    lines.extend([
        "", "## Constraints", "",
        "- Preserve both qualified 12B and 26B product paths.",
        "- Do not silently change precision, formats, context, cache, sampling, or timing boundaries.",
        "- Treat EXL3 or any new quantization path as unqualified until conversion, loader, CUDA, quality, memory, and product gates pass.",
        "- No runtime quantization, CPU weight offload, expert streaming, or duplicate persistent weight representation on the primary path.",
        "- Separate hypotheses from demonstrated facts and identify the evidence needed for every performance or VRAM claim.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.max_size_mib <= 0:
        raise SystemExit("--max-size-mib must be positive")

    commit = run_git("rev-parse", "HEAD")
    short_commit = run_git("rev-parse", "--short=8", "HEAD")
    branch = run_git("branch", "--show-current") or "detached"
    status = run_git("status", "--short")
    dirty = bool(status)
    date = dt.datetime.now().astimezone().date().isoformat()
    suffix = "-dirty" if dirty else ""
    filename = f"gem16-chatgpt-review-{date}-{short_commit}{suffix}.zip"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / filename
    temporary = output.with_suffix(".zip.partial")
    temporary.unlink(missing_ok=True)

    raw_dirs = DEFAULT_RAW_EVIDENCE_DIRS | set(args.raw_evidence_dir)
    references: list[str] = []
    try:
        with zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9,
            strict_timestamps=False,
        ) as archive:
            builder = ArchiveBuilder(archive)
            for relative in ROOT_FILES:
                builder.add_file(ROOT / relative)
            for relative in PROJECT_TREES:
                builder.add_tree(ROOT / relative, relative)
            add_compact_artifacts(builder, raw_dirs)

            if not args.no_reference_engines:
                references.append(f"llama.cpp: {add_llama_reference(builder)}")
                references.append(f"vLLM: {add_vllm_reference(builder)}")

            metadata = (
                f"Generated: {dt.datetime.now().astimezone().isoformat()}\n"
                f"Branch: {branch}\nCommit: {commit}\nDirty: {dirty}\n\n"
                f"Git status:\n{status or '(clean)'}\n\nRecent commits:\n"
                f"{run_git('log', '--oneline', '-35')}\n"
            )
            builder.add_text("CURRENT_GIT_STATE.txt", metadata)
            builder.add_text("WORKTREE_CHANGES.patch", run_git("diff", "--binary") + "\n")
            builder.add_text("REFERENCE_REVISIONS.txt", "\n".join(references) + "\n")
            builder.add_text("REVIEW_REQUEST.md", review_request(args.focus, commit, dirty))
            builder.add_text(
                "ARCHIVE_SCOPE.txt",
                "Included: project source/tests/docs/build contracts; compact milestone and selected raw text evidence; curated reference source.\n"
                "Excluded: models, build trees, Git objects, caches, benchmark result trees, profiler/database captures, binaries, images, and vendored Native Studio ImGui.\n"
                f"Selected artifacts/raw directories: {', '.join(sorted(raw_dirs))}\n",
            )

        size = temporary.stat().st_size
        limit = int(args.max_size_mib * 1024 * 1024)
        if size > limit:
            temporary.unlink(missing_ok=True)
            raise SystemExit(
                f"archive would be {size / 1048576:.2f} MiB, above "
                f"the {args.max_size_mib:.2f} MiB limit"
            )
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"ZIP integrity check failed at {bad}")
        temporary.replace(output)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        checksum = output.with_suffix(output.suffix + ".sha256")
        checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
        print(output)
        print(f"size_mib={size / 1048576:.2f}")
        print(f"sha256={digest}")
        return 0
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
