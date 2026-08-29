#!/usr/bin/env python3
"""Compile one Gemma 4 26B routed-expert layer to GEM16-Trellis35."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from gem16_compile.common import BoundedWorkspace, CompilerError, canonical_json_bytes
from gem16_compile.reader import (
    read_source_tensors,
    tensor_source_identity,
    verify_source_lock,
)
from gem16_compile.trellis35 import finalize_hessian, read_source_expert, regularize_weight
from gem16_compile.trellis35_artifact import (
    NativeCandidate,
    NativeTools,
    sha256_file,
    write_layer_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "models/gemma4-26b-qat-bf16.lock.json"
SEED = 0x47454D3136543335
SIGMA_REG = 0.025


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--gate-inputs-f32", type=Path, required=True)
    parser.add_argument("--down-inputs-f32", type=Path, required=True)
    parser.add_argument("--native-quantize", type=Path, required=True)
    parser.add_argument("--native-ldlq", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def checked_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    expected = int(np.prod(shape, dtype=np.int64)) * 4
    if path.stat().st_size != expected:
        raise ValueError(f"calibration input has wrong size: {path} expected={expected}")
    values = np.memmap(path, dtype="<f4", mode="r", shape=shape)
    if not np.isfinite(values).all():
        raise ValueError(f"calibration input is non-finite: {path}")
    return values


def candidate_metadata_path(prefix: Path, rate: int) -> Path:
    return prefix.with_suffix(f".k{rate}.json")


def load_cached_candidate(prefix: Path, rate: int, fingerprint: dict[str, object]) -> NativeCandidate | None:
    metadata_path = candidate_metadata_path(prefix, rate)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint or metadata.get("rate_bits") != rate:
            return None
        payload = prefix.with_suffix(f".k{rate}.payload")
        suh = prefix.with_suffix(f".k{rate}.suh.f16")
        svh = prefix.with_suffix(f".k{rate}.svh.f16")
        files = metadata["files"]
        for name, path in (("payload", payload), ("suh", suh), ("svh", svh)):
            if path.stat().st_size != files[name]["bytes"] or sha256_file(path) != files[name]["sha256"]:
                return None
        return NativeCandidate(
            rate_bits=rate,
            global_scale=float(metadata["global_scale"]),
            proxy_error=float(metadata["proxy_error"]),
            payload_path=payload,
            payload_bytes=int(files["payload"]["bytes"]),
            payload_sha256=str(files["payload"]["sha256"]),
            suh_path=suh,
            svh_path=svh,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_candidate(candidate: NativeCandidate, prefix: Path, fingerprint: dict[str, object]) -> None:
    metadata = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "rate_bits": candidate.rate_bits,
        "global_scale": candidate.global_scale,
        "proxy_error": candidate.proxy_error,
        "files": {
            "payload": {"bytes": candidate.payload_bytes, "sha256": candidate.payload_sha256},
            "suh": {"bytes": candidate.suh_path.stat().st_size, "sha256": sha256_file(candidate.suh_path)},
            "svh": {"bytes": candidate.svh_path.stat().st_size, "sha256": sha256_file(candidate.svh_path)},
        },
    }
    candidate_metadata_path(prefix, candidate.rate_bits).write_bytes(canonical_json_bytes(metadata))


def main() -> int:
    args = arguments()
    if not 0 <= args.layer < 30 or args.records <= 0 or args.records > 4096:
        raise ValueError("layer or calibration record count is invalid")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "candidates"
    cache.mkdir(exist_ok=True)
    tools = NativeTools(args.native_quantize, args.native_ldlq)
    gate_values = checked_memmap(args.gate_inputs_f32.resolve(strict=True), (args.records, 2816))
    down_values = checked_memmap(args.down_inputs_f32.resolve(strict=True), (128, args.records, 704))
    calibration = {
        "records": args.records,
        "gate_inputs_sha256": sha256_file(args.gate_inputs_f32),
        "down_inputs_sha256": sha256_file(args.down_inputs_f32),
    }

    workspace = BoundedWorkspace(8 * 1024**3, 1024 * 1024)
    verified = verify_source_lock(LOCK, args.source_root, workspace)
    tensors = read_source_tensors(verified, workspace)
    names = {
        family: f"model.language_model.layers.{args.layer}.experts.{suffix}"
        for family, suffix in (("gate_up", "gate_up_proj"), ("down", "down_proj"))
    }
    if any(name not in tensors for name in names.values()):
        raise CompilerError("verified source lacks a requested Trellis35 expert tensor")
    source_tensors = {family: tensors[name] for family, name in names.items()}
    native_identity = {
        "quantize_sha256": sha256_file(tools.quantize),
        "ldlq_sha256": sha256_file(tools.ldlq),
    }
    compiler_identity = {
        path.name: sha256_file(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "tools/gem16_compile/trellis35.py",
            ROOT / "tools/gem16_compile/trellis35_artifact.py",
            ROOT / "tools/gem16_compile/trellis35_quant.py",
        )
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()

    print("finalize gate_up Hessian", flush=True)
    gate_capture = gate_values.astype(np.float64).T @ gate_values.astype(np.float64)
    gate_finalized = finalize_hessian(
        gate_capture, args.records, seed=SEED,
        domain=f"layer{args.layer}:gate_up:input", sigma_reg=SIGMA_REG,
    )
    del gate_capture
    families: dict[str, list[tuple[NativeCandidate, NativeCandidate]]] = {
        "gate_up": [], "down": [],
    }
    for family in ("gate_up", "down"):
        for expert in range(128):
            if family == "gate_up":
                finalized = gate_finalized
            else:
                print(f"finalize down Hessian expert={expert}", flush=True)
                values = down_values[expert].astype(np.float64)
                capture = np.zeros((768, 768), dtype=np.float64)
                capture[:704, :704] = values.T @ values
                finalized = finalize_hessian(
                    capture, args.records, seed=SEED,
                    domain=f"layer{args.layer}:down:expert{expert}:input", sigma_reg=SIGMA_REG,
                )
                del values, capture
            source = read_source_expert(source_tensors[family], expert, family)
            regularized = regularize_weight(
                source, finalized.input_signs, seed=SEED,
                domain=f"layer{args.layer}:{family}:expert{expert}", global_scale=1.0,
            )
            del source
            prefix = cache / f"{family}-expert-{expert:03d}"
            pair: list[NativeCandidate] = []
            for rate in (3, 4):
                fingerprint = {
                    "schema_version": 1,
                    "producer_revision": revision,
                    "layer": args.layer,
                    "family": family,
                    "expert": expert,
                    "rate_bits": rate,
                    "seed": SEED,
                    "sigma_reg": SIGMA_REG,
                    "source": tensor_source_identity(source_tensors[family]),
                    "calibration": calibration,
                    "native": native_identity,
                    "compiler_sources": compiler_identity,
                }
                candidate = load_cached_candidate(prefix, rate, fingerprint)
                if candidate is None:
                    print(f"compile {family} expert={expert} K{rate}", flush=True)
                    candidate = tools.compile_candidate(
                        regularized, finalized, rate, prefix
                    )
                    save_candidate(candidate, prefix, fingerprint)
                else:
                    print(f"reuse {family} expert={expert} K{rate}", flush=True)
                pair.append(candidate)
            families[family].append((pair[0], pair[1]))
            del regularized
            if family == "down":
                del finalized

    manifest_base = {
        "status": "experimental_single_layer_wp2",
        "producer_revision": revision,
        "source_lock_sha256": verified.lock_sha256,
        "source_repository": verified.repository,
        "source_revision": verified.revision,
        "source_tensors": {
            family: {"name": descriptor.name, **tensor_source_identity(descriptor)}
            for family, descriptor in source_tensors.items()
        },
        "calibration": calibration,
        "compiler": {
            "seed": SEED,
            "sigma_reg": SIGMA_REG,
            "native": native_identity,
            "sources": compiler_identity,
            "scale_search": "coarse_0.1_to_1.9_then_fine_0.075_parabolic",
        },
        "logical_shapes": {"gate_up": [128, 1408, 2816], "down": [128, 2816, 704]},
        "physical_shapes": {"gate_up": [128, 2816, 1408], "down": [128, 768, 2816]},
        "logical_axis_order": "expert,output,input; compiler matrices are input,output",
    }
    artifact = output / f"layer-{args.layer:02d}.trellis35.bin"
    manifest_path = output / f"layer-{args.layer:02d}.trellis35.json"
    manifest = write_layer_artifact(
        artifact, manifest_path, layer=args.layer, families=families, manifest_base=manifest_base
    )
    print(
        f"trellis35_layer_ok layer={args.layer} bytes={manifest['artifact']['bytes']} "
        f"sha256={manifest['artifact']['sha256']}", flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CompilerError, OSError, ValueError) as error:
        print(f"trellis35_layer_error: {error}", file=sys.stderr)
        raise SystemExit(getattr(error, "exit_code", 1))
