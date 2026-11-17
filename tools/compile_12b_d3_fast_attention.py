#!/usr/bin/env python3
"""Offline compile and embed the opt-in D3 FP8 attention; runtime never JITs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.backends.nvidia.compiler import get_ptxas, get_ptxas_version

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tma", action="store_true")
    args = parser.parse_args()
    source = root / "src/cuda/attention/twelve_b_d3_fast_triton.py"
    spec = importlib.util.spec_from_file_location("gem16_d3_offline", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signature = {"Q": "*fp32", "K": "*fp8e4nv", "V": "*fp8e4nv",
                 "KS": "*bf16", "VS": "*bf16", "CONTROL": "*u64",
                 "PARTIAL": "*fp32", "LSE": "*fp32"}
    constants = {"HEAD_DIM": 512, "HEADS": 16, "SEGMENTS": 16, "TILE": 32,
                 "USE_TMA": args.tma}
    signature.update({name: "constexpr" for name in constants})
    compiled = triton.compile(ASTSource(module.gem16_d3_global_fp8,
                                       signature=signature, constexprs=constants),
                              target=GPUTarget("cuda", 120, 32),
                              options={"num_warps": 4, "num_stages": 3})
    blob = compiled.asm["cubin"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "attention.cubin").write_bytes(blob)
    (args.output / "attention.ptx").write_text(compiled.asm["ptx"])
    rows = [", ".join(f"0x{x:02x}" for x in blob[i:i+16]) for i in range(0, len(blob), 16)]
    header = ("// Generated offline by tools/compile_12b_d3_fast_attention.py; do not edit.\n"
              "#pragma once\n#include <cstddef>\nnamespace gem16::internal::d3_fast {\n"
              "alignas(16) inline constexpr unsigned char kCubin[] = {\n" +
              ",\n".join(rows) + "\n};\n" +
              f'inline constexpr char kKernel[] = "{compiled.metadata.name}";\n' +
              f"inline constexpr unsigned kShared = {compiled.metadata.shared};\n" +
              f"inline constexpr unsigned kGlobalScratchPerCTA = {compiled.metadata.global_scratch_size};\n" +
              f"inline constexpr unsigned kFixedCapacity = {17519 if args.tma else 0};\n" +
              "} // namespace gem16::internal::d3_fast\n")
    output_name = "twelve_b_d3_fast_embedded.cuh"
    (args.output / output_name).write_text(header)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assembler = get_ptxas(120)
    metadata = {"status": "experimental_unqualified_changed_attention_math",
        "inputs": {str(p.relative_to(root)): sha(p) for p in [source, Path(__file__)]},
        "triton_version": triton.__version__, "target": "sm120",
        "ptxas": {"path": assembler.path, "version": get_ptxas_version(120),
                  "sha256": sha(Path(assembler.path))},
        "options": {"num_warps": 4, "num_stages": 3},
        "tma_kv_loads": args.tma,
        "supported_capacity": 17519 if args.tma else "4..17519",
        "native_kv_layout_and_scales_preserved": True,
        "changed_operators": ["global D3 queries rounded/clamped to E4M3 (scale1)",
            "QK and PV use FP8 Tensor Core dot with FP32 accumulation",
            "online softmax tiles rounded to E4M3 after scaling256",
            "16 dynamic segments /32-token online softmax tiles"],
        "metadata": compiled.metadata._asdict(),
        "nvcc": subprocess.check_output(["/opt/cuda/bin/nvcc", "--version"], text=True),
        "outputs": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print("Compiled", len(blob), "cubin bytes;", compiled.metadata.shared, "shared bytes")


if __name__ == "__main__":
    main()
