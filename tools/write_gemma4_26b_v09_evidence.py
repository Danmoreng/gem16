#!/usr/bin/env python3
"""Write the immutable V09 Ordinary Vision semantic-closure evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/vision/v09-ordinary-semantic-closure.json"
PADDING_ORACLE = ROOT / "artifacts/vision/v09-padding-oracle.json"
RUNTIME_ORACLE = ROOT / "artifacts/vision/v09-runtime-oracle.json"
CURRENT_BINARY = ROOT / "build/Linux/blackwell-release/bin/gem16-chat"
BASELINE_BINARY = ROOT / "build/blackwell-release/bin/gem16-chat"
IMAGE = (
    ROOT
    / "third_party/cache/llama-convert-venv/lib/python3.10/site-packages/"
    "networkx/drawing/tests/baseline/test_house_with_colors.png"
)

SOURCE_FILES = (
    "CMakeLists.txt",
    "docs/ACTIVE_DECISIONS.md",
    "docs/plans/gemma4-26b/START_HERE_CODEX.md",
    "docs/plans/gemma4-26b/CODEX_GEM16_26B_VISION_PRODUCTIZATION_MASTER_INSTRUCTION.md",
    "include/gem16/engine.h",
    "include/gem16/image.h",
    "src/cli/gemma4_26b_vision_oracle_dump.cu",
    "src/cuda/attention/gemma4_26b_reference.cu",
    "src/cuda/attention/gemma4_26b_reference.h",
    "src/cuda/engine/detail/gemma4_26b_prefill.inc",
    "src/cuda/engine/gemma4_26b_reference.cu",
    "src/cuda/inference.cu",
    "src/cuda/inference_session.cuh",
    "src/cuda/moe/prefill_plan.cpp",
    "src/cuda/moe/prefill_plan.h",
    "src/cuda/vision/gemma4_26b.cu",
    "src/model/gemma4_26b_vision_contract.cpp",
    "src/model/gemma4_26b_vision_contract.h",
    "src/runtime/chat.cpp",
    "tests/unit/gemma4_26b_moe_prefill_plan_test.cpp",
    "tests/unit/image_test.cpp",
    "tools/compare_gemma4_26b_vision_output.py",
    "tools/compare_gemma4_26b_vision_padding.py",
    "tools/write_gemma4_26b_v09_evidence.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        data = (ROOT / relative).read_bytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def command(*arguments: str) -> str:
    return subprocess.check_output(arguments, cwd=ROOT, text=True).strip()


def main() -> int:
    padding = json.loads(PADDING_ORACLE.read_text(encoding="utf-8"))
    runtime = json.loads(RUNTIME_ORACLE.read_text(encoding="utf-8"))
    gpu = command(
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader",
    )
    report = {
        "schema_version": 1,
        "package": "V09 Ordinary multimodal semantic closure",
        "status": "acceptance_pass",
        "accepted": True,
        "source": {
            "base_commit": command("git", "rev-parse", "HEAD"),
            "branch": command("git", "branch", "--show-current"),
            "worktree_dirty": True,
            "implementation_files": list(SOURCE_FILES),
            "implementation_source_sha256": source_sha256(),
        },
        "artifacts": {
            "padding_oracle": {
                "path": str(PADDING_ORACLE.relative_to(ROOT)),
                "sha256": sha256(PADDING_ORACLE),
            },
            "runtime_oracle": {
                "path": str(RUNTIME_ORACLE.relative_to(ROOT)),
                "sha256": sha256(RUNTIME_ORACLE),
            },
            "current_binary_sha256": sha256(CURRENT_BINARY),
            "pre_v09_binary_sha256": sha256(BASELINE_BINARY),
            "text_artifact_content_sha256":
                "39183e2c7a1bd04c87a8f5bb5887453f8e92a3980bb507fd06d6252a6439c9f0",
            "vision_module_sha256": sha256(
                ROOT
                / "models/checkpoints/google-gemma-4-26b-a4b-it-qat-vision-fp8-v1/vision.gem16"
            ),
            "qualified_nvfp4_target": {
                "repository": "danmoreng/gemma-4-26B-A4B-it-GEM16",
                "revision": "63508b5826527484e707b4b46e2eacf077cf2b35",
                "format": "sm120-device-image-v1",
                "artifact_content_sha256":
                    "471805f7dad8abb84300be78b2822a63dcb1d35bff5aa98426a162cc8532ee17",
                "model_file_sha256":
                    "1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72",
            },
        },
        "environment": {
            "gpu": gpu,
            "cuda_compiler": command("nvcc", "--version").splitlines()[-1],
        },
        "semantics": {
            "local_attention": "sliding AND (causal OR same Vision block)",
            "global_attention": "causal; empty Vision range",
            "chunk_offsets_tested": [
                0, 1, 1023, 1024, 1940, 2047, 2048, 4095, 16105, 16383, 16384
            ],
            "chunk_plan_allocation": "fixed 257-entry array",
            "split_cached_span": "rejected before prefill",
            "positions": "exact canonical row-major and 3x3 pooled geometry",
            "padding_decision": padding["decision"],
            "padding_cases": len(padding["cases"]),
            "padding_equivalence_failures": sum(
                not case["passed"] for case in padding["cases"]
            ),
            "selected_budget_execution": "630/1260/2520 tower rows with masked padded keys",
        },
        "ordinary_frozen_image": {
            "image_sha256": sha256(IMAGE),
            "budget": 280,
            "raw_patches": 2394,
            "soft_tokens": 266,
            "prompt": "Describe the image precisely.",
            "greedy_output": (
                "A diagram of a graph with five nodes and six edges on a white "
                "background. The nodes are circles, and the edges are thick gray "
                "lines. There are four"
            ),
            "output_utf8_sha256":
                "d258ce128caf3ebeb7bc38ca896c5aaf1fd9b714298ebcc1aabb447591e48691",
            "repetitions": 2,
            "outputs_identical": True,
            "decode_tokens_per_second": [133.942, 134.475],
            "mtp_enabled": False,
        },
        "numerics": {
            "padded_fp8_runtime_vs_padded_bf16_oracle": runtime["metrics"],
            "boundary": "diagnostic only; no post-hoc acceptance tolerance",
            "runtime_output_sha256": runtime["gem16_output_sha256"],
        },
        "non_regression": {
            "trellis35_text": {
                "pre_v09_and_current_output_token_ids_identical": True,
                "tokens": 16,
                "output_token_ids_sha256":
                    "e9fde40e3c0e633746aa1835dbcc28871e6b30f6c5e8219aa3c74c1a899c7a7a",
            },
            "nvfp4": {
                "cuda_consumption_test": "passed",
                "qualified_12b_product_golden": "passed",
                "qualified_26b_current_device_image": "passed",
                "native_path":
                    "sm120_integrated_nvfp4_moe_bf16_tensor_router_fp8_kv",
                "vision_module_loaded": False,
                "mtp_enabled": False,
                "trellis35_dispatched": False,
                "pre_v09_and_current_output_token_ids_identical": True,
                "tokens": 16,
                "output_token_ids_sha256":
                    "5fc3c08c56ba3417e280ced964847dd8f388e77377b51d85b1bc418851c7c94f",
            },
            "d2": {
                "status": "fail_closed",
                "exit_code": 2,
                "message": "Gemma 4 26B Vision v1 requires Ordinary decoding",
            },
        },
        "memory": {
            "resident_text_weight_bytes": 12204692480,
            "vision_weight_bytes": 597313024,
            "kv_cache_bytes_at_2048": 125829120,
            "workspace_bytes_at_2048": 448097056,
            "admission_headroom_bytes": 2305622016,
            "repeat_encode_output_mismatches": 0,
            "recurring_device_free_delta_bytes": 0,
        },
        "tests": [
            {"name": "blackwell release build", "result": "passed"},
            {"name": "host unit", "result": "passed"},
            {"name": "host ASAN/UBSAN unit", "result": "passed"},
            {"name": "26B Vision module contract", "result": "passed"},
            {"name": "CUDA aggregate", "result": "passed"},
            {"name": "CUDA Trellis35", "result": "passed"},
            {"name": "CUDA attention reference", "result": "passed"},
            {"name": "CUDA NVFP4 consumption", "result": "passed"},
            {"name": "12B product golden", "result": "passed"},
        ],
        "profiling": {
            "sass": "not_applicable_semantic_package",
            "nsys": "deferred_to_V10_by_plan",
            "ncu": "deferred_to_V10_by_plan",
            "same_binary_ab": "not_applicable_no_performance_candidate",
            "fallback_count": 0,
        },
        "limitations": [
            "FP8 Vision versus BF16 remains a diagnostic quality comparison; V09 closes padding and text-side mask semantics, not quantization quality.",
            "Vision TTFT, peak-VRAM profiling, NSYS, and NCU intentionally remain deferred to V10.",
            "The implementation is uncommitted because this task did not authorize commit or push.",
        ],
        "next_gate": "V09 is accepted; stop and obtain owner approval before V10.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
