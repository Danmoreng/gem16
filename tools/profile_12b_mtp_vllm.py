#!/usr/bin/env python3
"""Bounded diagnostic D3 trace; annotations do not qualify primary throughput."""
from pathlib import Path
import functools
import json
import os
import sys


class D3TraceExtension:
    def init_worker_extension(self):
        import torch
        from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
            TargetDependentARSpeculator,
        )

        patched = []
        for name in ("propose", "_run_model", "_generate_draft",
                     "_multi_step_decode", "_fused_multi_step_decode"):
            original = getattr(TargetDependentARSpeculator, name)
            if getattr(original, "_gem16_d3_diagnostic", False):
                continue

            def decorate(fn, label):
                @functools.wraps(fn)
                def traced(*args, **kwargs):
                    with torch.profiler.record_function("gem16_diag.assistant." + label):
                        return fn(*args, **kwargs)
                traced._gem16_d3_diagnostic = True
                return traced

            setattr(TargetDependentARSpeculator, name, decorate(original, name))
            patched.append(name)
        directory = Path(os.environ["GEM16_VLLM_TRACE_DIR"])
        (directory / "worker-annotation.json").write_text(json.dumps({
            "diagnostic_only": True, "patched_python_boundaries": patched,
            "model_tensor_math_modified": False,
            "wrapper": str(Path(__file__).resolve()),
        }, indent=2) + "\n")


def main():
    import vllm
    import benchmark_wikipedia_workload as core
    from experiment_scratch import check_ram_storage

    def option(name):
        return sys.argv[sys.argv.index(name) + 1]

    if option("--engine") != "vllm" or option("--mtp-draft-tokens") != "3":
        raise SystemExit("this diagnostic requires vLLM with MTP D3")
    traces = Path(os.environ["GEM16_VLLM_TRACE_DIR"]).resolve()
    check_ram_storage(traces, expected_gib=2)
    traces.mkdir(parents=True, exist_ok=False)
    delay = int(os.environ.get("GEM16_VLLM_TRACE_DELAY", "24"))
    limit = int(os.environ.get("GEM16_VLLM_TRACE_STEPS", "16"))
    vllm.LLM = functools.partial(vllm.LLM,
        worker_extension_cls="profile_12b_mtp_vllm.D3TraceExtension",
        profiler_config={
            "profiler": "torch", "torch_profiler_dir": str(traces),
            "ignore_frontend": True, "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_memory": False,
            "detailed_trace_annotation": True,
        })
    original = core.run_vllm_request
    calls = 0

    def traced(llm, *args, **kwargs):
        nonlocal calls
        collect = calls == 1
        calls += 1
        if collect:
            llm.start_profile(profile_prefix="12b-d3", delay_iterations=delay,
                              max_iterations=limit)
        try:
            return original(llm, *args, **kwargs)
        finally:
            if collect:
                llm.stop_profile()

    core.run_vllm_request = traced
    core.main()
    output = Path(option("--output"))
    data = json.loads(output.read_text())
    data["status"] = "diagnostic_only_profiled_not_primary_throughput"
    data["profile"] = {"trace_dir": str(traces),
        "delay_worker_iterations": delay, "active_worker_iterations": limit,
        "scope_must_be_verified_from_trace": True,
        "python_phase_annotations": True, "model_tensor_math_modified": False}
    output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
