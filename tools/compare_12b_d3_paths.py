#!/usr/bin/env python3
"""Compare bounded D3 traces by actual launch correlation and verifier context."""
import argparse
import bisect
import collections
import csv
import gzip
import json
from pathlib import Path
import re

from analyze_12b_d3_paths import union_us


def vllm_groups(events):
    ranges = [e for e in events if e.get("cat") == "user_annotation"]
    execute = sorted((e for e in ranges if e["name"].startswith("execute_")),
                     key=lambda e: e["ts"])
    propose = [e for e in ranges if e["name"] == "gem16_diag.assistant.propose"]
    starts = [e["ts"] for e in execute]
    apis = {e["args"]["correlation"]: e for e in events
            if e.get("cat") in ("cuda_runtime", "cuda_driver")
            and "correlation" in e.get("args", {})}
    groups = [{"index": i, "context_end": int(re.search(r"sq4sk(\d+)", e["name"])[1]),
               "kernels": [], "apis": []} for i, e in enumerate(execute)]
    for e in events:
        if e.get("cat") != "kernel":
            continue
        api = apis[e["args"]["correlation"]]
        i = bisect.bisect_right(starts, api["ts"]) - 1
        if i < 0:
            raise ValueError("GPU kernel precedes first attributed worker iteration")
        ts = api["ts"]
        phase = "head/sampling/control"
        if any(a["ts"] <= ts < a["ts"] + a["dur"] for a in propose):
            phase = "assistant"
        elif execute[i]["ts"] <= ts < execute[i]["ts"] + execute[i]["dur"]:
            phase = "target backbone"
        args = e["args"]
        groups[i]["kernels"].append({"name": e["name"], "short": e["name"],
            "start": e["ts"], "end": e["ts"] + e["dur"], "phase": phase,
            "grid": args["grid"], "block": args["block"],
            "registers": args.get("registers per thread"),
            "shared": args.get("shared memory"), "stream": args.get("stream"),
            "graph": args.get("graph id"), "correlation": args["correlation"]})
    for api in apis.values():
        i = bisect.bisect_right(starts, api["ts"]) - 1
        if i >= 0:
            groups[i]["apis"].append(api)
    return groups


def native_groups(capture, result):
    kernels = capture["kernels"]
    mtp = result["runs"][0]["engine_result"]["mtp"]
    count = mtp["verification_groups"]
    indices = [i for i, k in enumerate(kernels)
               if k["short"] == "InitializeControlledAssistantKernel"]
    if len(indices) != 2 * count:
        raise ValueError("expected exactly one warmup and one retained diagnostic request")
    ids = result["representative_output_token_ids"]
    drafts = mtp["proposed_token_ids"]
    if len(drafts) != count * 3:
        raise ValueError("D3 proposal count mismatch")
    cursor = 1
    accepted_total = 0
    groups = []
    for group, start in enumerate(indices[-count:]):
        finish = next(i for i in range(start, len(kernels))
                      if kernels[i]["short"] == "ContinueMtpChainKernel")
        selected = [dict(k) for k in kernels[start:finish + 1]]
        verify = next(k["start"] for k in selected
                      if k["short"] == "BuildControlledMtpInputsKernel")
        accept = next(k["start"] for k in selected
                      if k["short"] == "AcceptMtpGroupKernel")
        for k in selected:
            k["phase"] = ("assistant" if k["start"] < verify else
                          "target backbone" if k["start"] < accept else
                          "head/sampling/control")
            if "OutputHead" in k["name"]:
                k["phase"] = "head/sampling/control"
        accepted = 0
        for j in range(3):
            if drafts[group * 3 + j] != ids[cursor + j]:
                break
            accepted += 1
        # Q rows start at the committed input; four-row end is exclusive.
        groups.append({"index": group, "context_end": 16384 + cursor + 3,
            "accepted_derived": accepted, "kernels": selected, "apis": []})
        accepted_total += accepted
        cursor += accepted + 1
    if accepted_total != mtp["accepted_tokens"]:
        raise ValueError("derived acceptance does not match this capture's counters")
    return groups


def family(k, engine):
    name = k["name"]
    if k["phase"] == "head/sampling/control":
        return "target head/sampling/control"
    if engine == "native":
        if name == "gem16_d3_local_fp8" or "MergeD3Local4Kernel" in name:
            return "local attention"
        if name == "gem16_d3_global_fp8" or "MergeD3Fast16Kernel" in name:
            return "global attention"
        if "GlobalGqa" in name or ("MergeOnlineDecodeAttention" in name and "512" in name):
            return "global attention"
        if "Attention" in name and ("SplitOnline" in name or "MergeOnline" in name):
            return "local attention"
        if "Fp8MatrixBinding" in name:
            return "FP8 QKV/O"
        if "DownFixedShape" in name:
            return "NVFP4 Down"
        if "Sm120MatrixProjectionKernel" in name:
            return "NVFP4 Gate/Up"
        if "OutputCandidates" in name or "OutputArgmax" in name:
            return "assistant head"
        if "Gemv" in name or "PreProjection" in name:
            return "assistant BF16 projections"
        if "Kv" in name or "KV" in name:
            return "KV/transaction"
        return "norm/quant/rotary/elementwise"
    if name in ("kernel_unified_attention", "reduce_segments"):
        # Actual captured geometries: KVH1 global uses gridY=1 and 41728
        # shared bytes; local KVH8 uses gridY=8 and 29440 shared bytes.
        if name == "kernel_unified_attention":
            return "global attention" if k["grid"][1] == 1 else "local attention"
        return "global attention" if k["shared"] == 0 else "local attention"
    if "MainloopSm120TmaWarpSpecializedBlockScaled" in name:
        if "BlockScaledILi7" in name:
            return "NVFP4 Gate/Up"
        if "BlockScaledILi3" in name:
            return "NVFP4 Down"
        return "NVFP4 matrix (unclassified tactic)"
    if "device_kernel" in name and "e4m3" in name:
        return "FP8 QKV/O"
    if "gemvx" in name or "wmma" in name:
        return "assistant BF16 projections/head"
    return "norm/quant/rotary/elementwise"


def summarize(groups, engine):
    count = len(groups)
    all_k = [k for group in groups for k in group["kernels"]]
    phases = collections.defaultdict(list)
    families = collections.defaultdict(list)
    signatures = collections.defaultdict(list)
    spans = []
    for group in groups:
        events = group["kernels"]
        spans.append(max(k["end"] for k in events) - min(k["start"] for k in events))
        for k in events:
            phases[k["phase"]].append(k)
            families[(k["phase"], family(k, engine))].append(k)
            signatures[(k["phase"], family(k, engine), k["name"],
                        tuple(k["grid"]), tuple(k["block"]),
                        k["registers"], k["shared"])].append(k)
    def cost(events):
        return {"count": len(events), "count_per_group": len(events) / count,
                "sum_ms_per_group": sum(k["end"] - k["start"] for k in events) / count / 1000,
                "union_ms_per_group": union_us(events) / count / 1000}
    result = {"groups": count, "context_end_values": [g["context_end"] for g in groups],
        "group_indices": [g["index"] for g in groups], "all_kernels": cost(all_k),
        "mean_intragroup_span_ms": sum(spans) / count / 1000,
        "mean_intragroup_no_kernel_ms": (sum(spans) - union_us(all_k)) / count / 1000,
        "phases": {p: cost(v) for p, v in phases.items()},
        "families": [{"phase": p, "family": f, **cost(v)}
                     for (p, f), v in sorted(families.items())],
        "dispatches": [{"phase": s[0], "family": s[1], "name": s[2],
                         "grid": s[3], "block": s[4], "registers": s[5],
                         "shared": s[6], **cost(v)}
                        for s, v in sorted(signatures.items(),
                                           key=lambda p: -sum(k["end"] - k["start"] for k in p[1]))]}
    # Only adjacent original groups contribute a measured intergroup interval.
    gaps = []
    for a, b in zip(groups, groups[1:]):
        if b["index"] == a["index"] + 1:
            gaps.append(min(k["start"] for k in b["kernels"]) -
                        max(k["end"] for k in a["kernels"]))
    result["mean_adjacent_intergroup_gap_ms"] = sum(gaps) / len(gaps) / 1000 if gaps else None
    return result


def write_order(path, group, engine):
    start = min(k["start"] for k in group["kernels"])
    with path.open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["order", "phase", "family", "start_us", "duration_us",
                         "stream", "graph", "grid", "block", "registers", "shared", "name"])
        for i, k in enumerate(sorted(group["kernels"], key=lambda k: k["start"])):
            writer.writerow([i, k["phase"], family(k, engine), k["start"] - start,
                k["end"] - k["start"], k["stream"], k["graph"], k["grid"],
                k["block"], k["registers"], k["shared"], k["name"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    raw = json.load(gzip.open(root / "native-kernel-events.json.gz"))
    native = native_groups(raw, json.loads((root / "native-d3-recapture.json").read_text()))
    events = json.load(gzip.open(root / "vllm-bounded-trace.json.gz"))["traceEvents"]
    vllm_all = vllm_groups(events)
    if len(vllm_all) != 16:
        raise ValueError("expected exactly 16 bounded worker iterations")
    # Exclude profiler window edges, with no selection based on kernel timing.
    vllm = vllm_all[2:-2]
    low, high = vllm[0]["context_end"], vllm[-1]["context_end"]
    native = [g for g in native if low <= g["context_end"] <= high]
    if not native:
        raise ValueError("no native verification groups in the same context band")
    result = {"status": "diagnostic_only_not_primary_throughput",
        "selection": {"rule": "exclude first/last two vLLM capture iterations; native groups in same verifier context band",
                      "context_band": [low, high], "same_tensor_values_across_engines": False},
        "native": summarize(native, "native"), "vllm": summarize(vllm, "vllm"),
        "native_source_acceptance_validated": True,
        "vllm_all_kernel_launch_correlations_resolved": True,
        "timing_limits": "Different profilers and numerical/output paths; GPU unions, host APIs and concurrent sums must not be added. No projected primary gain is measured here."}
    for engine, groups in [("native", native), ("vllm", vllm)]:
        write_order(root / (engine + "-representative-order.csv"), groups[len(groups) // 2], engine)
    (root / "d3-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    for engine in ("native", "vllm"):
        x = result[engine]
        print(engine, x["groups"], "groups; kernel union", x["all_kernels"]["union_ms_per_group"],
              "ms/group; intragroup gap", x["mean_intragroup_no_kernel_ms"])
        print([(p["phase"], p["family"], round(p["sum_ms_per_group"], 4)) for p in x["families"]])


if __name__ == "__main__":
    main()
