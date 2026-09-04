#!/usr/bin/env python3
"""Bounded parent/candidate ordinary-decode experiment: one warmup, three runs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from benchmark_wikipedia_workload import (
    load_workload, repository_state, run_gem16, summarize_runs,
)
from qualify_mtp import GpuTelemetrySampler, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--nvfp4", required=True, type=Path)
    parser.add_argument("--trellis35", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    workload_path = Path("benchmarks/prompts/wikipedia-summary-16k.json")
    workload, prompt, generation = load_workload(workload_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = args.output.with_suffix(".prompt-token-ids.txt").resolve()
    with prompt_path.open("x") as output:
        output.write(",".join(map(str, prompt)))
    binaries = {name: getattr(args, name).resolve() for name in ("parent", "candidate")}
    sampling = dict(temperature=1.0, top_k=64, top_p=0.95, repetition_penalty=1.0, seed=0)
    patch = subprocess.check_output(["git", "diff", "--binary"])
    document = {
        "status": "running", "source": repository_state(),
        "tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "workload": str(workload_path), "prompt_tokens": len(prompt),
        "prompt_identity": workload["prompt"]["token_ids_sha256"],
        "generation": generation, "sampling": sampling,
        "warmups": 1, "measurements": 3, "mtp": False,
        "binaries": {name: {"path": str(path), "sha256": file_sha256(path)}
                     for name, path in binaries.items()},
        "models": {},
        "limitations": ["Small diagnostic sample, not release qualification.",
                        "Cold prefix; fresh process per run. Inference excludes model load.",
                        "Different formats are compared to their own exact-output parent, not assumed numerically identical."],
    }
    # Preserve completed runs even when a later run fails.
    def save():
        args.output.write_text(json.dumps(document, indent=2) + "\n")

    try:
        for model_name in ("nvfp4", "trellis35"):
            model = getattr(args, model_name).resolve()
            data = {"path": str(model), "metadata": json.loads((model / "gem16_model.json").read_text()),
                    "runs": {"parent": [], "candidate": []}, "order": []}
            document["models"][model_name] = data
            reference_tokens = None
            for pair in range(4):
                phase = "warmup" if pair == 0 else "measured"
                order = ("parent", "candidate") if pair % 2 == 0 else ("candidate", "parent")
                for mode in order:
                    print(f"{model_name} {phase} {pair}: {mode}", flush=True)
                    telemetry = GpuTelemetrySampler(0.2)
                    telemetry.start()
                    try:
                        run, tokens = run_gem16(binaries[mode], model, prompt_path,
                            len(prompt), generation, None, 0, False, sampling)
                    finally:
                        samples = telemetry.stop()
                    run["gpu_telemetry"] = samples
                    run["phase"] = phase
                    run["pair"] = pair
                    data["runs"][mode].append(run)
                    data["order"].append({"phase": phase, "pair": pair, "mode": mode})
                    if reference_tokens is None:
                        reference_tokens = tokens
                        data["output_token_ids"] = tokens
                    if reference_tokens != tokens:
                        raise RuntimeError(f"{model_name} exact output mismatch")
                    print(f"  {run['decode_tokens_per_second']:.3f} tok/s, {len(tokens)} tokens", flush=True)
                    save()
            data["summary"] = {mode: summarize_runs([r for r in runs if r["phase"] == "measured"])
                               for mode, runs in data["runs"].items()}
            data["ordinary_speedup"] = (
                data["summary"]["candidate"]["decode_tokens_per_second"]["median"] /
                data["summary"]["parent"]["decode_tokens_per_second"]["median"])
            data["exact_output_parity"] = True
        document["status"] = "characterized"
    except Exception as error:
        document["status"] = "failed"
        document["error"] = str(error)
        raise
    finally:
        save()
    print(json.dumps({name: data["ordinary_speedup"] for name, data in document["models"].items()}))


if __name__ == "__main__":
    main()
