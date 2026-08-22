#!/usr/bin/env python3
"""Generate bounded ignored M12 layer-0/layer-5 attention fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.gemma4_26b_moe_oracle import (
        COMPILED, GOLDEN, SafeTensorReader, _load_json_file,
    )
except ModuleNotFoundError:
    from gemma4_26b_moe_oracle import (  # type: ignore[no-redef]
        COMPILED, GOLDEN, SafeTensorReader, _load_json_file,
    )


def row(captures: dict, name: str, position: int) -> list[float]:
    records = captures[name]["rows"]
    return next(record["values_f32"] for record in records
                if int(record["position"]) == position)


def build_case(captures: dict, reader: SafeTensorReader, layer: int) -> dict:
    position = 0
    prefix = f"model.language_model.layers.{layer}"
    trusted = f"layer_{layer}"
    normalized = row(captures, trusted + ".input_layernorm", position)
    norm_weight = reader.bf16(prefix + ".input_layernorm.weight").tolist()
    hidden = [float(value) / float(weight)
              for value, weight in zip(normalized, norm_weight)]
    expected = {
        "q_raw": row(captures, trusted + ".attention.q_proj", position),
        "k_raw": row(captures, trusted + ".attention.k_proj", position),
        "q_normalized": row(captures, trusted + ".attention.q_norm", position),
        "k_normalized": row(captures, trusted + ".attention.k_norm", position),
        "v_normalized": row(captures, trusted + ".attention.v_norm", position),
        "attention": row(captures, trusted + ".attention.context_before_o", position),
        "output_projection": row(captures, trusted + ".attention.o_proj", position),
        "post_attention": row(captures, trusted + ".post_attention_layernorm", position),
    }
    if layer == 0:
        expected["v_raw"] = row(captures, trusted + ".attention.v_proj", position)
    return {"layer": layer, "position": position, "hidden_f32": hidden,
            "expected": expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    golden = _load_json_file(GOLDEN)
    captures = golden["captures"]
    reader = SafeTensorReader(COMPILED)
    document = {
        "schema_version": 1,
        "milestone": "M12",
        "source": {
            "golden": str(GOLDEN.relative_to(GOLDEN.parents[3])),
            "compiled_artifact": str(COMPILED.relative_to(COMPILED.parents[3])),
            "position": 0,
            "residual_reconstruction":
                "trusted_input_layernorm_bf16_divided_by_compiled_norm_weight",
        },
        "cases": [build_case(captures, reader, 0),
                  build_case(captures, reader, 5)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, separators=(",", ":")) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
