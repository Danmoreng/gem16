#!/usr/bin/env python3
"""Package the headless server from an explicit binary using a fresh allowlist stage."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package(binary, output, platform_name):
    binary = binary.resolve(strict=True)
    version = (ROOT / "VERSION").read_text().strip()
    actual = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if actual != f"gem16-server {version}":
        raise ValueError("server VERSION mismatch")
    output.mkdir(parents=True, exist_ok=True)
    name = f"gem16-server-{version}-{platform_name}"
    with tempfile.TemporaryDirectory(prefix="gem16-server-stage-") as temporary:
        stage = Path(temporary) / name
        (stage / "bin").mkdir(parents=True)
        files = {
            f"bin/{binary.name}": binary,
            "VERSION": ROOT / "VERSION",
            "LICENSE": ROOT / "LICENSE",
            "docs/SERVER.md": ROOT / "docs/SERVER.md",
            "docs/AGENT_COMPATIBILITY.md": ROOT / "docs/AGENT_COMPATIBILITY.md",
            "docs/OPENAI_AGENT_CORE_V1.md": ROOT / "docs/OPENAI_AGENT_CORE_V1.md",
            "tools/fetch_model.py": ROOT / "tools/fetch_model.py",
            "tools/hf_cache.py": ROOT / "tools/hf_cache.py",
        }
        for lock in [
            "gemma4-12b-nvfp4",
            "gemma4-12b-mtp-assistant",
            "gemma4-26b-trellis35-target",
            "gemma4-26b-vision-fp8",
            "gemma4-26b-gem16-assistant",
        ]:
            relative = f"models/{lock}.lock.json"
            files[relative] = ROOT / relative
        readme = stage / "README.txt"
        readme.write_text(
            "GEM16 headless development candidate (not release-qualified).\n"
            "Requires a supported Blackwell SM120/SM120a GPU and NVIDIA driver.\n"
            "No Studio or remote inference account is required; model weights download separately.\n"
            "Python 3.11+ is needed only for the included model acquisition tools.\n"
            "Run acquisition commands in docs/SERVER.md from this archive root.\n"
            "Replace the source-build server path in that guide with bin/gem16-server\n"
            "(Windows: bin/gem16-server.exe). The recommended public Compact Vision\n"
            "context is 220000 on Linux / 170000 on Windows, subject to admission.\n"
            "Links to source/tests/evidence in the guides refer to the repository.\n"
            "System library/clean-machine qualification remains an open release gate.\n"
        )
        # Server embeds image/audio decoders and uses cpp-httplib.
        for dependency, filename in [
            ("stb", "LICENSE"),
            ("miniaudio", "LICENSE"),
            ("cpp-httplib", "LICENSE"),
        ]:
            source = ROOT / "third_party" / dependency / filename
            if not source.is_file():
                raise ValueError(f"missing dependency notice: {source}")
            files[f"licenses/{dependency}.txt"] = source
        for relative, source in files.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        manifest = {
            "version": version,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
            ),
            "platform": platform_name,
            "backend": "Blackwell SM120/SM120a",
            "qualification": "unqualified candidate; release gates required",
            "files": {
                relative: sha256(stage / relative)
                for relative in sorted([*files, "README.txt"])
            },
            "model_locks": {
                str(p.relative_to(ROOT)): sha256(p)
                for p in sorted((ROOT / "models").glob("*.lock.json"))
            },
            "toolchain_lock_sha256": sha256(ROOT / "toolchains/blackwell16gb.lock"),
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if platform_name == "windows-x64":
            archive = output / f"{name}.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as result:
                for item in sorted(stage.rglob("*")):
                    if item.is_file():
                        result.write(item, str(item.relative_to(stage.parent)))
        else:
            archive = output / f"{name}.tar.gz"
            with tarfile.open(archive, "w:gz") as result:
                result.add(stage, arcname=name)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n"
    )
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "build/packages")
    parser.add_argument(
        "--platform", choices=["linux-x64", "windows-x64"], required=True
    )
    args = parser.parse_args()
    print(package(args.binary, args.output, args.platform))
