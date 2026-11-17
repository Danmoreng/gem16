"""Validate AP1 observed groups against native outputs and aggregate counters."""
from collections import Counter


def validate(observation, engine, prompt_tokens):
    if not observation.get("captured") or observation.get("fault") != 0:
        raise ValueError("group observer did not complete without faults")
    groups = observation["groups"]
    mtp = engine["mtp"]
    if len(groups) != observation["chain_group_count"] or len(groups) != mtp["verification_groups"]:
        raise ValueError("group count mismatch")
    tokens = engine["output_token_ids"]
    prompt = prompt_tokens
    emitted, proposals = [], []
    acceptance, depths, unions = Counter(), Counter(), Counter()
    for index, group in enumerate(groups):
        if group["group"] != index or group["position"] != prompt + len(emitted):
            raise ValueError("group index/position mismatch")
        depth, accepted = group["depth"], group["accepted"]
        if depth != 2 or not 0 <= accepted <= depth:
            raise ValueError("unexpected group depth/acceptance")
        if len(group["proposals"]) != depth or len(group["outputs"]) != accepted + 1:
            raise ValueError("group array extent mismatch")
        if group["output_count"] != len(group["outputs"]):
            raise ValueError("group output count mismatch")
        if group["proposals"][:accepted] != group["outputs"][:accepted]:
            raise ValueError("accepted proposal differs from output")
        if accepted < depth and group["proposals"][accepted] == group["outputs"][accepted]:
            raise ValueError("rejected proposal equals verification")
        emitted.extend(group["outputs"])
        proposals.extend(group["proposals"])
        layers = group["layer_expert_ids"]
        if len(layers) != 30:
            raise ValueError("missing routing layer")
        for ids in layers:
            if len(ids) != 24 or any(type(i) is not int or not 0 <= i < 128 for i in ids):
                raise ValueError("invalid expert IDs")
            if any(len(set(ids[start:start+8])) != 8 for start in (0, 8, 16)):
                raise ValueError("duplicate expert in one top-eight row")
            size = len(set(ids))
            unions[size] += 1
        acceptance[accepted] += 1
        depths[depth] += 1
    tail = mtp["ordinary_fallback_tokens"]
    if emitted != tokens[1:len(emitted)+1] or len(emitted) + tail != len(tokens) - 1:
        raise ValueError("observed output stream mismatch")
    if sum(depths[k]*k for k in depths) != mtp["proposed_tokens"]:
        raise ValueError("proposal count mismatch")
    if sum(acceptance[k]*k for k in acceptance) != mtp["accepted_tokens"]:
        raise ValueError("accepted count mismatch")
    return {
        "observed_groups": len(groups),
        "observed_acceptance_histogram_0_1_2": [acceptance[i] for i in range(3)],
        "observed_depth_histogram": dict(depths),
        "derived_expert_union_histogram": dict(unions),
        "observed_proposals": proposals,
        "output_and_counter_parity": True,
        "scope": "explicit graph observer; union sizes derived from observed per-row expert IDs",
    }
