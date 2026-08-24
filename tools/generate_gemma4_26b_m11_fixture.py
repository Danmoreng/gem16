#!/usr/bin/env python3
"""Generate a bounded ignored M11 input fixture from the accepted M10 sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.gemma4_26b_moe_oracle import (
        GOLDEN, COMPILED, SafeTensorReader, _load_json_file,
    )
except ModuleNotFoundError:  # Direct execution places tools/ on sys.path.
    from gemma4_26b_moe_oracle import (  # type: ignore[no-redef]
        GOLDEN, COMPILED, SafeTensorReader, _load_json_file,
    )


def row(captures: dict, name: str, position: int, key: str) -> list:
    records = captures[name]["rows"]
    return next(record[key] for record in records if int(record["position"]) == position)


def build(layer: int, position: int, golden_path: Path = GOLDEN) -> dict:
    golden = _load_json_file(golden_path)
    golden_label = (str(GOLDEN.relative_to(GOLDEN.parents[3]))
                    if golden_path == GOLDEN else str(golden_path))
    captures = golden["captures"]
    prefix = f"model.language_model.layers.{layer}"
    trusted_prefix = f"layer_{layer}"
    reader = SafeTensorReader(COMPILED)
    pre_norm = reader.bf16(prefix + ".pre_feedforward_layernorm.weight").tolist()
    shared_input = row(captures, trusted_prefix + ".pre_feedforward_layernorm",
                       position, "values_f32")
    hidden = [float(value) / float(scale)
              for value, scale in zip(shared_input, pre_norm)]
    contributions = next(
        group["contributions"]
        for group in captures[trusted_prefix + ".routed_expert_contributions"]
        if int(group["position"]) == position
    )
    return {
        "schema_version": 1,
        "milestone": "M11",
        "source": {
            "m10_acceptance": "artifacts/m10/acceptance.json",
            "golden": golden_label,
            "compiled_artifact": str(COMPILED.relative_to(COMPILED.parents[3])),
            "layer": layer,
            "position": position,
            "residual_reconstruction": "trusted_pre_shared_bf16_divided_by_compiled_pre_shared_norm",
        },
        "hidden_f32": hidden,
        "expected": {
            "router_probabilities": row(captures, trusted_prefix + ".router_probabilities",
                                         position, "values_f32"),
            "top_ids": row(captures, trusted_prefix + ".router_top_ids",
                           position, "values_i64"),
            "top_weights": row(captures, trusted_prefix + ".router_top_weights",
                               position, "values_f32"),
            "shared_output": row(captures, trusted_prefix + ".shared_mlp",
                                 position, "values_f32"),
            "expert_contributions": [record["values_f32"] for record in contributions],
            "routed_sum": row(captures, trusted_prefix + ".routed_experts_sum",
                              position, "values_f32"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--position", type=int, default=0)
    args = parser.parse_args()
    if args.layer not in (0, 5, 6, 29) or args.position < 0:
        raise SystemExit("fixture must use a selected layer and non-negative position")
    document = build(args.layer, args.position, args.golden)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, separators=(",", ":")) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
