"""Join native D2 trace costs only to observations from that same capture.

Without a same-capture observer, per-group acceptance/routes remain unknown.
A read-only proposal dump supports derived acceptance/position, not routing.
Timings include profiler effects and never substitute for primary throughput.
"""
import argparse
import bisect
import csv
import json
from pathlib import Path
import statistics

from analyze_decode_cost import sweep
from validate_mtp_groups import validate


def proposal_groups(proposals, engine, prompt_tokens):
    if not proposals.get("captured") or proposals.get("copy_error") != 0:
        raise ValueError("proposal copy did not complete")
    mtp = engine["mtp"]
    if (mtp.get("draft_tokens") != 2 or mtp.get("adaptive") is not False
            or mtp.get("group_execution") != "gpu_chained_fixed_d2_conditional_graph"):
        raise ValueError("proposal derivation requires the native fixed-D2 chain")
    count = mtp["verification_groups"]
    drafts = proposals["proposed_token_ids"]
    expected = {"group_count": count, "proposed_count": mtp["proposed_tokens"],
                "accepted_count": mtp["accepted_tokens"],
                "ordinary_tail_count": mtp["ordinary_fallback_tokens"],
                # The chain excludes the first token sampled by prefill.
                "output_count": len(engine["output_token_ids"]) - 1}
    if any(proposals.get(key) != value for key, value in expected.items()):
        raise ValueError("proposal copy/engine counter mismatch")
    if len(drafts) != count * 2 or len(drafts) != mtp["proposed_tokens"]:
        raise ValueError("cannot establish all-depth-two native groups")
    if mtp["ordinary_fallback_tokens"]:
        raise ValueError("ordinary tail needs separate accounting")
    tokens = engine["output_token_ids"][1:]
    cursor, accepted_total, rows = 0, 0, []
    for i in range(count):
        draft = drafts[i * 2:i * 2 + 2]
        accepted = 0
        while accepted < 2 and cursor + accepted < len(tokens) and draft[accepted] == tokens[cursor + accepted]:
            accepted += 1
        if cursor + accepted >= len(tokens):
            raise ValueError("missing verifier continuation")
        rows.append({"group": i, "position": prompt_tokens + cursor, "depth": 2,
                     "accepted": accepted, "expert_unions_by_layer": None})
        cursor += accepted + 1
        accepted_total += accepted
    if cursor != len(tokens) or accepted_total != mtp["accepted_tokens"]:
        raise ValueError("proposal/output/counter mismatch")
    return rows


def group_facts(engine, prompt_tokens, observation=None, proposals=None):
    if observation is not None and proposals is not None:
        raise ValueError("choose one same-capture observation")
    if observation is not None:
        validate(observation, engine, prompt_tokens)
        return [{"group": g["group"], "position": g["position"], "depth": g["depth"],
                 "accepted": g["accepted"],
                 "expert_unions_by_layer": [len(set(ids)) for ids in g["layer_expert_ids"]]}
                for g in observation["groups"]], {
                    "acceptance_position_depth": "observed in this capture",
                    "expert_unions": "derived from this capture's observed per-row expert IDs"}
    if proposals is not None:
        return proposal_groups(proposals, engine, prompt_tokens), {
            "acceptance_position_depth": "derived from this capture's observed proposals, emitted tokens and native depth-two counters",
            "expert_unions": "unknown; no routing observer attached"}
    return [{"group": i, "position": None, "depth": None, "accepted": None,
             "expert_unions_by_layer": None} for i in range(engine["mtp"]["verification_groups"])], {
                 "acceptance_position_depth": "unknown; no same-capture observation",
                 "expert_unions": "unknown; no routing observer attached"}


def describe(values):
    return {"count": len(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "minimum": min(values), "maximum": max(values)}


def analyze(cost, kernels, facts, provenance):
    starts = [k for k in kernels if k["name"] == "InitializeControlledAssistantKernel"]
    advances = [k for k in kernels if k["name"] == "AdvanceMtpD2ChainKernel"]
    if not len(starts) == len(advances) == len(facts):
        raise ValueError("trace/group boundary count mismatch")
    positions = [k["start"] for k in kernels]
    prefix_end, last = [], -1
    for k in kernels:
        last = max(last, k["end"])
        prefix_end.append(last)
    rows = []
    for i, (start, advance, fact) in enumerate(zip(starts, advances, facts)):
        begin = start["start"]
        end = starts[i + 1]["start"] if i + 1 < len(starts) else cost["window_ns"][1]
        if not begin < advance["end"] <= end:
            raise ValueError("invalid group interval")
        left, right = bisect.bisect_left(positions, begin), bisect.bisect_left(positions, end)
        if left and prefix_end[left - 1] > begin:
            raise ValueError("cross-group overlap needs explicit DAG")
        union = sweep([(k["start"], k["end"], k["category"]) for k in kernels[left:right]], [], begin, end)
        rows.append({**fact, "core_group_ms": (advance["end"] - begin) / 1e6,
                     "group_interval_ms": (end - begin) / 1e6,
                     "kernel_busy_union_ms": union["gpu_busy_union_ms"],
                     "kernel_overlap_ms": union["gpu_overlap_ms"]})
    by_accepted = None
    if all(r["accepted"] is not None for r in rows):
        by_accepted = {}
        for accepted in range(3):
            selected = [r for r in rows if r["accepted"] == accepted]
            by_accepted[str(accepted)] = ({key: describe([r[key] for r in selected])
                                         for key in ("core_group_ms", "group_interval_ms", "position")}
                                        if selected else {"count": 0})
    return {"diagnostic_only": True, "provenance": provenance,
            "scope": "single profiled capture; kernel-only interval union excludes memcpy/memset activities; final group interval includes terminal result boundary; no causal acceptance penalty",
            "by_accepted": by_accepted, "groups": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cost", "kernels", "engine", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    observer = parser.add_mutually_exclusive_group()
    observer.add_argument("--groups", type=Path, help="group observation from this exact capture")
    observer.add_argument("--proposals", type=Path, help="read-only proposals from this exact capture")
    args = parser.parse_args()
    def load(path): return json.loads(path.read_text())
    with args.kernels.open() as stream:
        kernels = list(csv.DictReader(stream))
    for row in kernels:
        for key in ("start", "end"):
            row[key] = int(row[key])
    facts, provenance = group_facts(load(args.engine), args.prompt_tokens,
                                    load(args.groups) if args.groups else None,
                                    load(args.proposals) if args.proposals else None)
    result = analyze(load(args.cost), kernels, facts, provenance)
    result["inputs"] = {name: str(getattr(args, name)) if getattr(args, name) else None
                        for name in ("cost", "kernels", "engine", "groups", "proposals")}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
