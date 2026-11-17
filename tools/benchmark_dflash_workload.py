#!/usr/bin/env python3
"""Experimental DFlash/MTP runner using the existing exact-token benchmark boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import benchmark_wikipedia_workload as core
from hf_cache import default_target_model


def identity(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("llama-cpp", "vllm"), required=True)
    parser.add_argument("--method", choices=("dflash", "mtp"), default="dflash")
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=default_target_model())
    parser.add_argument("--gguf", type=Path, default=Path("build/Linux/llama_cpp/gemma4-12b-mixed-q8-nvfp4.gguf"))
    parser.add_argument("--workload", type=Path, default=Path("benchmarks/prompts/wikipedia-summary-16k.json"))
    parser.add_argument("--draft-tokens", type=core.positive_int, default=15)
    parser.add_argument("--warmups", type=core.nonnegative_int, default=3)
    parser.add_argument("--repetitions", type=core.positive_int, default=10)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-memory-bytes", type=core.positive_int)
    parser.add_argument("--port", type=core.positive_int, default=8097)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    # Match the historical 16K+1135 run, including the five suppressed IDs.
    workload, prompt, generation = core.load_workload(args.workload)
    generation = core.fixed_output_generation(generation, 1135)
    if len(prompt) != 16384:
        parser.error("this experiment requires the existing 16384-token workload")
    context = len(prompt) + generation["max_new_tokens"]
    draft = args.draft.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = None
    log = None
    telemetry_file = args.output.with_suffix(".telemetry.csv").open("x")
    telemetry = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,utilization.gpu",
        "--format=csv", "--loop-ms=200"], stdout=telemetry_file, stderr=subprocess.STDOUT)
    rows = []
    outputs = []
    runtime = {"draft": str(draft), "argv": sys.argv, "environment": {
        k: os.environ.get(k) for k in ("CUDA_HOME", "MAX_JOBS", "TORCHINDUCTOR_COMPILE_THREADS", "FLASHINFER_NVCC_THREADS")}}
    config = {"method": args.method, "draft_tokens": args.draft_tokens, "batch_size": 1,
              "context_tokens_requested": context, "prefix_caching": False,
              "warmups": args.warmups, "measured_repetitions": args.repetitions,
              "primary_statistic": "median", "cpu_offload_gb": 0}
    try:
        if args.engine == "vllm":
            import torch
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
            suppressed = [tokenizer.decode([t], skip_special_tokens=False)
                          for t in generation["suppress_token_ids"]]
            spec = {"method": args.method, "model": str(draft),
                    "num_speculative_tokens": args.draft_tokens}
            runtime["packages"] = core.package_versions(("vllm", "torch", "transformers", "compressed-tensors"))
            runtime["torch_cuda"] = torch.version.cuda
            runtime["device_name"] = torch.cuda.get_device_properties(0).name
            runtime["checkpoint"] = str(args.model.resolve())
            runtime["draft_config"] = json.loads((draft / "config.json").read_text())
            config.update(kv_cache="fp8", gpu_memory_utilization=args.gpu_memory_utilization,
                          kv_cache_memory_bytes=args.kv_cache_memory_bytes,
                          max_num_batched_tokens=1024, speculative_config=spec,
                          trust_remote_code=False, enforce_eager=False)
            llm = LLM(model=str(args.model), tokenizer=str(args.model), trust_remote_code=False,
                      max_model_len=context, gpu_memory_utilization=args.gpu_memory_utilization,
                      cpu_offload_gb=0, enforce_eager=False, enable_prefix_caching=False,
                      enable_chunked_prefill=True, kv_cache_dtype="fp8", max_num_seqs=1,
                      max_num_batched_tokens=1024, disable_log_stats=False, seed=generation["seed"],
                      language_model_only=True, limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
                      mm_processor_cache_gb=0, speculative_config=spec,
                      kv_cache_memory_bytes=args.kv_cache_memory_bytes)

            def run_once():
                row, tokens = core.run_vllm_request(llm, SamplingParams, prompt, generation,
                                                  suppressed, args.draft_tokens)
                # Keep common speculative telemetry without inventing depth histograms.
                counters = row.pop("mtp")
                for key in ("d1_groups", "d2_groups", "d4_groups", "ordinary_fallback_tokens"):
                    counters.pop(key, None)
                row["speculative"] = counters
                return row, tokens
        else:
            executable = core.resolve_central_llama_server(None)
            spec_type = "draft-" + args.method
            command = [str(executable), "--model", str(args.gguf.resolve(strict=True)),
                       "--ctx-size", str(context), "--n-gpu-layers", "all", "--split-mode", "none",
                       "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                       "--parallel", "1", "--cache-ram", "0", "--ctx-checkpoints", "0",
                       "--threads", "8", "--poll", "100", "--prio", "0", "--verbosity", "3",
                       "--batch-size", "2048", "--ubatch-size", "512", "--host", "127.0.0.1",
                       "--port", str(args.port), "--no-webui", "--offline",
                       "--spec-type", spec_type, "--spec-draft-model", str(draft),
                       "--spec-draft-n-max", str(args.draft_tokens), "--spec-draft-n-min", "1",
                       "--spec-draft-ngl", "all", "--override-tensor", "token_embd.weight=CUDA0"]
            runtime.update(command=command, executable=identity(executable),
                           llama_cpp_commit=subprocess.check_output([
                               "git", "-C", str(executable.parents[2]), "rev-parse", "HEAD"], text=True).strip())
            runtime["libraries"] = [identity(p) for p in sorted(executable.parent.glob("lib*.so*"))
                                    if p.is_file() and not p.is_symlink()]
            config.update(kv_cache="q8_0", logical_batch=2048, microbatch=512,
                          threads=8, poll=100, flash_attention=True, gpu_layers="all",
                          token_embedding_placement="CUDA0")
            log = args.output.with_suffix(".server.log").open("x")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            base_url = f"http://127.0.0.1:{args.port}"
            core.wait_for_server(process, base_url)

            def run_once():
                return core.run_llama_request(base_url, prompt, generation, args.draft_tokens,
                                              (spec_type,))

        for index in range(args.warmups + args.repetitions):
            warmup = index < args.warmups
            number = index + 1 if warmup else index - args.warmups + 1
            total = args.warmups if warmup else args.repetitions
            print(f"{'warmup' if warmup else 'measured'} {number}/{total}", flush=True)
            row, tokens = run_once()
            if len(tokens) != 1135:
                raise RuntimeError("incorrect output length")
            if row.get("speculative", {}).get("proposed_tokens", 0) <= 0:
                raise RuntimeError("no observed drafting activity")
            print(json.dumps({"decode_tokens_per_second": row["decode_tokens_per_second"],
                              "output_sha256": row["output_token_sha256"],
                              "speculative": row["speculative"]}), flush=True)
            if not warmup:
                rows.append(row)
                outputs.append(tokens)
                # Retain completed samples even if a subsequent request fails.
                with args.output.with_suffix(".samples.jsonl").open("a") as stream:
                    stream.write(json.dumps({"run": row, "output_token_ids": tokens}) + "\n")
        document = {"schema_version": 1, "status": "development_characterization",
                    "engine": args.engine, "benchmark_source": core.repository_state(),
                    "sources": [identity(Path(__file__)), identity(Path(core.__file__))],
                    "workload": {"path": str(args.workload.resolve()), "prompt_tokens": len(prompt),
                                 "prompt_token_ids_sha256": workload["prompt"]["token_ids_sha256"],
                                 "generation": generation},
                    "runtime": runtime, "configuration": config,
                    "summary": core.summarize_runs(rows), "runs": rows,
                    "representative_output_token_ids": outputs[0],
                    "limitations": ["Historical D2 cohorts are reused at owner request; no fresh D2 controls.",
                                    "Decode uses 1134 intervals after first token, excluding model load and prefill.",
                                    "Only this fixed greedy text workload is characterized; quality parity is not assumed."]}
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "decode_median": document["summary"]["decode_tokens_per_second"]["median"]}), flush=True)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log is not None:
            log.close()
        telemetry.terminate()
        telemetry.wait(timeout=10)
        telemetry_file.close()


if __name__ == "__main__":
    main()
