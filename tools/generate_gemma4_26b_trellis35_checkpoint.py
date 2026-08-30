#!/usr/bin/env python3
"""Resumable all-layer GEM16-Trellis35 routed-expert compiler.

Each layer receives its own exact calibration activation capture. Completed
layers are verified before their temporary candidate/cache files are removed.
The final checkpoint index is emitted only when all 30 layers pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

try:
    from tools.gem16_compile.common import (
        BoundedWorkspace,
        CompilerError,
        InvalidPlanError,
        canonical_json_bytes,
    )
    from tools.gem16_compile.reader import read_source_tensors, verify_source_lock
    from tools.gem16_compile.trellis35 import read_calibration_capture
    from tools.gem16_compile.trellis35_artifact import sha256_file
except ModuleNotFoundError:  # Direct execution with tools/ as sys.path[0].
    from gem16_compile.common import (
        BoundedWorkspace,
        CompilerError,
        InvalidPlanError,
        canonical_json_bytes,
    )
    from gem16_compile.reader import read_source_tensors, verify_source_lock
    from gem16_compile.trellis35 import read_calibration_capture
    from gem16_compile.trellis35_artifact import sha256_file


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "models/gemma4-26b-qat-bf16.lock.json"
CORPUS = ROOT / "benchmarks/corpora/gemma4_26b/calibration.json"
PROFILE = "gem16-trellis35-w4a8-v1"
LAYERS = 30
RECORDS = 1258
LAYER_BYTES = 345_147_392


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-calibration", type=Path, required=True)
    parser.add_argument("--native-expand-down", type=Path, required=True)
    parser.add_argument("--native-quantize", type=Path, required=True)
    parser.add_argument("--native-ldlq", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--last-layer", type=int, default=29)
    parser.add_argument("--keep-intermediates", action="store_true")
    return parser.parse_args()


def run(arguments: list[str]) -> None:
    result = subprocess.run(arguments, cwd=ROOT, text=True, check=False)
    if result.returncode != 0:
        raise InvalidPlanError(
            f"Trellis35 checkpoint child failed ({result.returncode}): {arguments[0]}"
        )


def corpus_token_files(work: Path) -> tuple[list[Path], str]:
    document = json.loads(CORPUS.read_text(encoding="utf-8"))
    records = document.get("records")
    if not isinstance(records, list) or len(records) != 4:
        raise InvalidPlanError("Trellis35 calibration corpus must contain four records")
    token_dir = work / "token_ids"
    token_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    total = 0
    for record in records:
        name, values = record.get("id"), record.get("input_token_ids")
        if (
            not isinstance(name, str)
            or not isinstance(values, list)
            or not values
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 or value >= 262144 for value in values)
        ):
            raise InvalidPlanError("Trellis35 calibration token record is invalid")
        path = token_dir / f"{name}.u32le"
        payload = np.asarray(values, dtype="<u4").tobytes()
        if path.exists() and path.read_bytes() != payload:
            raise InvalidPlanError("existing Trellis35 token file disagrees with corpus")
        if not path.exists():
            path.write_bytes(payload)
        paths.append(path)
        total += len(values)
    if total != RECORDS:
        raise InvalidPlanError("Trellis35 calibration corpus token total changed")
    return paths, sha256_file(CORPUS)


def layer_is_complete(layer_dir: Path, layer: int, source_root: Path) -> bool:
    artifact = layer_dir / f"layer-{layer:02d}.trellis35.bin"
    manifest = layer_dir / f"layer-{layer:02d}.trellis35.json"
    report = layer_dir / "verification.json"
    if not artifact.exists() or not manifest.exists():
        return False
    run([
        sys.executable,
        str(ROOT / "tools/verify_gemma4_26b_trellis35_layer.py"),
        "--artifact", str(artifact), "--manifest", str(manifest),
        "--source-root", str(source_root), "--report", str(report),
    ])
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    return (
        metadata.get("checkpoint_profile") == PROFILE
        and metadata.get("layer") == layer
        and metadata.get("artifact", {}).get("bytes") == LAYER_BYTES
    )


def cleanup_layer_intermediates(layer_dir: Path) -> None:
    candidate_dir = (layer_dir / "candidates").resolve()
    if candidate_dir.parent != layer_dir.resolve() or candidate_dir.name != "candidates":
        raise InvalidPlanError("unsafe Trellis35 candidate cleanup target")
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    for name in ("calibration.bin", "gate-inputs.f32", "down-inputs.f32"):
        (layer_dir / name).unlink(missing_ok=True)


def compile_layer(
    layer: int,
    layer_dir: Path,
    token_files: list[Path],
    source_root: Path,
    reference_model: Path,
    tools: dict[str, Path],
    gate_source: object,
    keep_intermediates: bool,
) -> None:
    layer_dir.mkdir(parents=True, exist_ok=True)
    if layer_is_complete(layer_dir, layer, source_root):
        if not keep_intermediates:
            cleanup_layer_intermediates(layer_dir)
        print(f"trellis35_checkpoint_resume layer={layer} status=verified", flush=True)
        return
    capture = layer_dir / "calibration.bin"
    gate = layer_dir / "gate-inputs.f32"
    down = layer_dir / "down-inputs.f32"
    command = [
        str(tools["calibration"]), "--model", str(reference_model),
        "--output", str(capture), "--layer", str(layer),
        "--context", "1258",
    ]
    for path in token_files:
        command.extend(("--token-ids", str(path)))
    print(f"trellis35_checkpoint_capture layer={layer}", flush=True)
    run(command)
    parsed = read_calibration_capture(capture)
    if parsed.layer != layer or parsed.positions.size != RECORDS:
        raise InvalidPlanError("Trellis35 combined calibration capture is incomplete")
    gate.write_bytes(parsed.gate_up_inputs.astype("<f4", copy=False).tobytes())
    print(f"trellis35_checkpoint_expand_down layer={layer}", flush=True)
    run([
        str(tools["expand"]), str(RECORDS), str(gate), str(gate_source.path),
        str(gate_source.absolute_offset), str(down),
    ])
    print(f"trellis35_checkpoint_quantize layer={layer}", flush=True)
    run([
        sys.executable,
        str(ROOT / "tools/generate_gemma4_26b_trellis35_layer.py"),
        "--source-root", str(source_root), "--layer", str(layer),
        "--records", str(RECORDS), "--gate-inputs-f32", str(gate),
        "--down-inputs-f32", str(down),
        "--native-quantize", str(tools["quantize"]),
        "--native-ldlq", str(tools["ldlq"]),
        "--output-dir", str(layer_dir),
    ])
    if not layer_is_complete(layer_dir, layer, source_root):
        raise InvalidPlanError(f"Trellis35 layer {layer} did not pass final verification")
    if not keep_intermediates:
        cleanup_layer_intermediates(layer_dir)
    print(f"trellis35_checkpoint_layer_ok layer={layer}", flush=True)


def write_checkpoint_index(output: Path, corpus_sha256: str) -> None:
    layers = []
    total = 0
    for layer in range(LAYERS):
        directory = output / f"layer-{layer:02d}"
        artifact = directory / f"layer-{layer:02d}.trellis35.bin"
        manifest = directory / f"layer-{layer:02d}.trellis35.json"
        report = directory / "verification.json"
        if not all(path.is_file() and not path.is_symlink()
                   for path in (artifact, manifest, report)):
            raise InvalidPlanError("all 30 Trellis35 layers are required for the checkpoint index")
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata.get("layer") != layer or metadata.get("checkpoint_profile") != PROFILE:
            raise InvalidPlanError("Trellis35 layer identity disagrees with checkpoint order")
        size = artifact.stat().st_size
        if size != LAYER_BYTES:
            raise InvalidPlanError("Trellis35 layer byte count changed")
        layers.append({
            "layer": layer,
            "artifact": str(artifact.relative_to(output)),
            "artifact_bytes": size,
            "artifact_sha256": sha256_file(artifact),
            "manifest": str(manifest.relative_to(output)),
            "manifest_sha256": sha256_file(manifest),
            "verification": str(report.relative_to(output)),
            "verification_sha256": sha256_file(report),
        })
        total += size
    content = {
        "schema_version": 1,
        "format": "GEM16-Trellis35",
        "format_version": 1,
        "checkpoint_profile": PROFILE,
        "status": "wp2_complete_30_layer_routed_expert_artifact",
        "source_lock_sha256": "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230",
        "calibration_corpus_sha256": corpus_sha256,
        "layer_count": LAYERS,
        "routed_expert_bytes": total,
        "payload_bpw_encoded": 3.5,
        "gate_up_padding": "none",
        "down_padding": "input_zero_pad_704_to_768",
        "layers": layers,
    }
    content["checkpoint_content_sha256"] = __import__("hashlib").sha256(
        canonical_json_bytes(content)
    ).hexdigest()
    (output / "trellis35-experts.json").write_bytes(canonical_json_bytes(content))


def main() -> int:
    args = arguments()
    if not 0 <= args.first_layer <= args.last_layer < LAYERS:
        raise InvalidPlanError("Trellis35 requested layer range is invalid")
    source_root = args.source_root.resolve(strict=True)
    reference_model = args.reference_model.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise InvalidPlanError("Trellis35 checkpoint output must be a real directory")
    output.mkdir(parents=True, exist_ok=True)
    tools = {
        "calibration": args.native_calibration.resolve(strict=True),
        "expand": args.native_expand_down.resolve(strict=True),
        "quantize": args.native_quantize.resolve(strict=True),
        "ldlq": args.native_ldlq.resolve(strict=True),
    }
    if any(not path.is_file() or path.is_symlink() for path in tools.values()):
        raise InvalidPlanError("Trellis35 native tools must be regular non-symlink files")
    workspace = BoundedWorkspace(8 * 1024**3, 1024 * 1024)
    verified = verify_source_lock(LOCK, source_root, workspace)
    tensors = read_source_tensors(verified, workspace)
    work = output / ".work"
    work.mkdir(exist_ok=True)
    token_files, corpus_sha256 = corpus_token_files(work)
    for layer in range(args.first_layer, args.last_layer + 1):
        name = f"model.language_model.layers.{layer}.experts.gate_up_proj"
        if name not in tensors:
            raise InvalidPlanError(f"Trellis35 source tensor is missing: {name}")
        compile_layer(
            layer, output / f"layer-{layer:02d}", token_files,
            source_root, reference_model, tools, tensors[name],
            args.keep_intermediates,
        )
    if all((output / f"layer-{layer:02d}").is_dir() for layer in range(LAYERS)):
        write_checkpoint_index(output, corpus_sha256)
        print("trellis35_checkpoint_ok layers=30", flush=True)
    else:
        print(
            f"trellis35_checkpoint_partial first={args.first_layer} last={args.last_layer}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CompilerError, OSError, ValueError) as error:
        print(f"trellis35_checkpoint_error: {error}", file=sys.stderr)
        raise SystemExit(getattr(error, "exit_code", 1))
