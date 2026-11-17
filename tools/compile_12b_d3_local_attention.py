#!/usr/bin/env python3
"""Offline build of the fixed12B T4 local ring-attention experimental cubin."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas, get_ptxas_version
    from triton.compiler import ASTSource
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tile", type=int, choices=(32, 64), default=32)
    args = parser.parse_args()
    source = root / "src/cuda/attention/twelve_b_d3_local_triton.py"
    spec = importlib.util.spec_from_file_location("gem16_local_offline", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signature = {"Q": "*fp32", **{n: "*fp8e4nv" for n in ["K", "V", "SK", "SV"]},
                 "KS": "*bf16", "VS": "*bf16", "CONTROL": "*i64", "PARTIAL": "*fp32", "LSE": "*fp32",
                 "SEGMENTS": "constexpr", "TILE": "constexpr"}
    compiled = triton.compile(ASTSource(module.gem16_d3_local_fp8, signature=signature,
        constexprs={"SEGMENTS": 4, "TILE": args.tile}), target=GPUTarget("cuda", 120, 32),
        options={"num_warps": 4, "num_stages": 3})
    assert compiled.metadata.global_scratch_size == 0
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ["cubin", "ptx"]:
        data = compiled.asm[name]
        (args.output / ("local." + name)).write_bytes(data if isinstance(data, bytes) else data.encode())
    rows = [", ".join(f"0x{x:02x}" for x in compiled.asm["cubin"][i:i+16])
            for i in range(0, len(compiled.asm["cubin"]), 16)]
    header = ("// Generated offline by tools/compile_12b_d3_local_attention.py; do not edit.\n"
        "#pragma once\nnamespace gem16::internal::d3_local {\n"
        "alignas(16) inline constexpr unsigned char kCubin[] = {\n" + ",\n".join(rows) + "\n};\n" +
        f'inline constexpr char kKernel[] = "{compiled.metadata.name}";\n' +
        f"inline constexpr unsigned kShared = {compiled.metadata.shared};\n" + "}\n")
    output = args.output / "twelve_b_d3_local_embedded.cuh"
    output.write_text(header)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assembler = get_ptxas(120)
    (args.output / "provenance.json").write_text(json.dumps({
        "status": "experimental_unqualified_changed_local_attention_math",
        "inputs": {str(p.relative_to(root)): sha(p) for p in [source, Path(__file__)]},
        "triton_version": triton.__version__, "ptxas": {"path": assembler.path,
            "version": get_ptxas_version(120), "sha256": sha(Path(assembler.path))},
        "metadata": compiled.metadata._asdict(),
        "tile": args.tile,
        "scope": "12B D3 T4 QH16/KVH8/H256;1024 native ring plus4 speculative rows;4segments; FP8 Q/P TensorCore QK/PV",
        "outputs": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
    }, indent=2, default=str) + "\n")
    print("Compiled local", len(compiled.asm["cubin"]), "bytes", compiled.metadata.shared, "shared")


if __name__ == "__main__":
    main()
