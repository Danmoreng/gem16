#!/usr/bin/env python3
"""Build/run a diagnostic replay against verified current central llama.cpp.

This Linux laboratory writes new evidence only; its timings are not primary
cross-engine throughput. Loaded modules are reported by the driver itself.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from benchmark_wikipedia_workload import resolve_central_llama_server

ROOT = Path(__file__).resolve().parents[1]


def identity(path):
    path = path.resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def loaded_modules(stderr, library_dir):
    result = {}
    for line in stderr.splitlines():
        if not line.startswith("GEM16_REPLAY_MODULE_HEX="):
            continue
        path = Path(os.fsdecode(bytes.fromhex(line.split("=", 1)[1])))
        if path.name == "linux-vdso.so.1":
            continue
        if not path.is_absolute():
            raise ValueError("loaded module has no absolute identity")
        item = identity(path)
        if path.name.startswith(("libllama", "libggml")):
            if path.resolve().parent != library_dir.resolve():
                raise ValueError("replay loaded a noncentral llama/ggml library")
        result[str(path)] = item
    if not any(Path(p).name.startswith("libllama") for p in result):
        raise ValueError("replay did not report the actually loaded libllama")
    if not any(Path(p).name.startswith("libggml-cuda") for p in result):
        raise ValueError("replay did not report the CUDA backend")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--mode", choices=("ordinary", "d2", "d2-windows"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("never overwrite replay evidence")
    server = resolve_central_llama_server(None)
    source = server.parents[2]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    directory = server.parent
    manifest_path = args.executable.with_suffix(".build.json")
    sources = [ROOT / "tests/cuda/llama_decode_replay_driver.cpp",
               ROOT / "tests/cuda/decode_replay_support.h"]
    runtime_files = {str(p.resolve()): identity(p) for p in directory.glob("*.so*")}
    if args.build:
        if args.executable.exists() or manifest_path.exists():
            raise ValueError("never overwrite a replay build")
        command = ["g++", "-std=c++20", "-O3", "-DNDEBUG",
                   '-DGEM16_LLAMA_REPLAY_COMMIT="' + commit + '"',
                   "-I" + str(source / "include"), "-I" + str(source / "ggml/include"),
                   "-I" + str(ROOT / "tests/cuda"), str(sources[0]),
                   "-L" + str(directory), "-Wl,-rpath," + str(directory),
                   "-lllama", "-lggml", "-lggml-base", "-ldl", "-o", str(args.executable)]
        subprocess.run(command, check=True)
        result = {"diagnostic_only": True, "source_commit": commit,
                  "central_server": identity(server), "command": command,
                  "compiler": subprocess.check_output(["g++", "--version"], text=True),
                  "central_libraries": runtime_files,
                  "launcher": identity(Path(__file__)),
                  "sources": [identity(p) for p in sources],
                  "executable": identity(args.executable)}
        manifest_path.write_text(json.dumps(result, indent=2) + "\n")
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
        return
    if not all((args.model, args.prompt, args.history, args.mode)):
        parser.error("run requires model, prompt, history and mode")
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_commit"] != commit or manifest["executable"] != identity(args.executable):
        raise ValueError("mixed replay build/source/executable")
    if manifest["central_libraries"] != runtime_files or manifest["launcher"] != identity(Path(__file__)):
        raise ValueError("replay launcher or central libraries changed after build")
    for item in manifest["sources"]:
        if identity(Path(item["path"])) != item:
            raise ValueError("replay source changed after build")
    before = {str(p): identity(p) for p in directory.glob("*.so*")}
    for suffix in (".stderr.log", ".stdout.log"):
        if args.output.with_suffix(suffix).exists():
            raise ValueError("never overwrite a recorded replay attempt")
    env = os.environ.copy()
    if env.get("LD_PRELOAD"):
        raise ValueError("replay provenance run cannot preload another observer")
    env.update(GEM16_LLAMA_BACKEND_DIR=str(directory), LD_LIBRARY_PATH=str(directory))
    command = [str(args.executable.resolve()), str(args.model.resolve()),
               str(args.prompt.resolve()), str(args.history.resolve()), args.mode]
    proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    with args.output.with_suffix(".stderr.log").open("x") as stream:
        stream.write(proc.stderr)
    with args.output.with_suffix(".stdout.log").open("x") as stream:
        stream.write(proc.stdout)
    proc.check_returncode()
    data = json.loads(proc.stdout)
    if data["engine"] != "llama.cpp-" + commit + "-q4_0":
        raise ValueError("replay driver source identity mismatch")
    actual = loaded_modules(proc.stderr, directory)
    if before != {str(p): identity(p) for p in directory.glob("*.so*")}:
        raise ValueError("central libraries changed during replay")
    result = {"diagnostic_only": True, "source_commit": commit,
              "build": manifest, "central_server": identity(server),
              "loaded_modules": actual, "command": command,
              "inputs": [identity(p) for p in (args.model, args.prompt, args.history)],
              "result": data,
              "timing_scope": "diagnostic C API replay; not primary engine comparison"}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
