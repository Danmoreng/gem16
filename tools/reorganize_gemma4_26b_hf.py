#!/usr/bin/env python3
"""Create the normalized Gemma 4 26B GEM16 Hub layout.

The migration copies immutable blobs inside the Hub repository.  It does not
download, re-hash, re-encode, or upload the multi-gigabyte model payloads.  The
old source revision remains addressable forever; only a new repository commit
gets the normalized paths.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import PurePosixPath
from typing import Any


REPOSITORY = "danmoreng/gemma-4-26B-A4B-it-GEM16"
SOURCE_REVISION = "6de2a057f11332420819f8e6efd08e42d7a03bc7"
MODEL_NAME = "Gemma 4 26B A4B IT"


@dataclass(frozen=True)
class Component:
    component_id: str
    directory: str
    source_directory: str
    source_payload: str
    payload_name: str
    payload_bytes: int
    payload_sha256: str
    role: str
    format_name: str
    quantization: dict[str, Any]
    public: bool
    auxiliary_files: tuple[str, ...]

    @property
    def payload_path(self) -> str:
        return f"{self.directory}/{self.payload_name}"


COMPONENTS = (
    Component(
        component_id="gemma4-26b-a4b-it-trellis35-w4a8",
        directory="components/gemma-4-26b-a4b-it-trellis35-w4a8",
        source_directory="trellis35",
        source_payload="model.gem16",
        payload_name="gemma-4-26b-a4b-it-trellis35-w4a8.gem16",
        payload_bytes=12_204_692_480,
        payload_sha256=(
            "552ace4b3f2e8e20bbc03a9d4b30887bdb6297d6da9f48d54b4a2b4e9fc803c4"
        ),
        role="text-target",
        format_name="gem16-sm120-trellis35-device-image-v2",
        quantization={
            "name": "Trellis35 W4A8",
            "weights": "mixed K3/K4 routed-expert storage; approximately 3.5 bits per weight",
            "activations": "FP8",
            "note": "EXL3-derived offline packing in a GEM16-native direct-load image",
        },
        public=True,
        auxiliary_files=(
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "gem16_model.json",
            "gem16_compilation.json",
            "gem16.lock.json",
        ),
    ),
    Component(
        component_id="gemma4-26b-a4b-it-vision-fp8-e4m3fn",
        directory="components/gemma-4-26b-a4b-it-vision-fp8-e4m3fn",
        source_directory="vision",
        source_payload="vision.gem16",
        payload_name="gemma-4-26b-a4b-it-vision-fp8-e4m3fn.gem16",
        payload_bytes=597_390_648,
        payload_sha256=(
            "805e1fee4ad80dbb76a846c0009d6f57dbb50192e7009ef1633cdc3b5252a536"
        ),
        role="vision-encoder",
        format_name="gem16-sm120-vision-device-image-v1",
        quantization={
            "name": "FP8 E4M3FN rowwise",
            "linear_weights": "FP8 E4M3FN per output row",
            "linear_scales": "BF16 per output row",
            "other_tensors": "BF16",
            "note": "This component is FP8, not Q8.",
        },
        public=True,
        auxiliary_files=(
            "gem16_vision.json",
            "vision_compilation.json",
            "vision.lock.json",
        ),
    ),
    Component(
        component_id="gemma4-26b-a4b-it-assistant-hybrid-nvfp4-fp8-bf16",
        directory="components/gemma-4-26b-a4b-it-assistant-hybrid-nvfp4-fp8-bf16",
        source_directory="assistant",
        source_payload="model-00001-of-00001.safetensors",
        payload_name=(
            "gemma-4-26b-a4b-it-assistant-hybrid-nvfp4-fp8-bf16.safetensors"
        ),
        payload_bytes=258_317_280,
        payload_sha256=(
            "4d3ce2102ad0631d9e7e0586be0b108d5789cbc5b90d21b4c50613979228d927"
        ),
        role="fixed-d2-assistant",
        format_name="sm120-mtp-assistant-hybrid-v1",
        quantization={
            "name": "hybrid NVFP4 / FP8 / BF16",
            "embedding_head_and_experts": "NVFP4 group-16 divisor",
            "attention": "FP8 per output row",
            "other_tensors": "BF16",
        },
        public=True,
        auxiliary_files=(
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "gem16_model.json",
            "gem16_compilation.json",
            "gem16.lock.json",
        ),
    ),
    Component(
        component_id="gemma4-26b-a4b-it-internal-target-hybrid-nvfp4-fp8-bf16",
        directory="internal/gemma-4-26b-a4b-it-target-hybrid-nvfp4-fp8-bf16",
        source_directory="",
        source_payload="model.gem16",
        payload_name=(
            "gemma-4-26b-a4b-it-target-hybrid-nvfp4-fp8-bf16.gem16"
        ),
        payload_bytes=14_696_668_160,
        payload_sha256=(
            "1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72"
        ),
        role="internal-text-target",
        format_name="sm120-text-hybrid-v1",
        quantization={
            "name": "hybrid NVFP4 / FP8 / BF16",
            "embedding_head_and_experts": "NVFP4 group-16 divisor",
            "attention": "FP8 per output row",
            "state_and_control_tensors": "BF16",
        },
        public=False,
        auxiliary_files=(
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "gem16_model.json",
            "gem16_compilation.json",
            "gem16.lock.json",
        ),
    ),
)


ROOT_FILES_TO_REMOVE = (
    "chat_template.jinja",
    "config.json",
    "gem16.lock.json",
    "gem16_compilation.json",
    "gem16_components.json",
    "gem16_model.json",
    "generation_config.json",
    "model.gem16",
    "tokenizer.json",
    "tokenizer_config.json",
)


def source_path(component: Component, name: str) -> str:
    return str(PurePosixPath(component.source_directory, name))


def renamed_auxiliary(name: str) -> str:
    return {
        "gem16_compilation.json": "compilation.json",
        "vision_compilation.json": "compilation.json",
        "gem16_model.json": "runtime.json",
        "gem16_vision.json": "runtime.json",
        "gem16.lock.json": "source.lock.json",
        "vision.lock.json": "source.lock.json",
    }.get(name, name)


def component_document(component: Component) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "component_id": component.component_id,
        "model_name": MODEL_NAME,
        "role": component.role,
        "public": component.public,
        "engine": "GEM16",
        "hardware": "NVIDIA Blackwell SM120",
        "format": component.format_name,
        "quantization": component.quantization,
        "artifact": {
            "path": component.payload_name,
            "bytes": component.payload_bytes,
            "sha256": component.payload_sha256,
        },
        "source_revision": SOURCE_REVISION,
        "runtime_integrity": {
            "load_time_model_hashing": False,
            "offline_lock_verification": True,
        },
    }


def profiles_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "public_profiles": [
            {
                "profile_id": "gemma4-26b-a4b-compact-vision",
                "display_name": "Gemma 4 26B A4B Compact Vision",
                "components": [
                    COMPONENTS[0].component_id,
                    COMPONENTS[1].component_id,
                ],
                "optional_fixed_d2_assistant": COMPONENTS[2].component_id,
                "supports": {"text": True, "image": True, "audio": False},
                "qualification_state": "production_qualified",
            }
        ],
        "internal_profiles": [
            {
                "profile_id": "gemma4-26b-a4b-nvfp4",
                "component": COMPONENTS[3].component_id,
                "optional_fixed_d2_assistant": COMPONENTS[2].component_id,
                "purpose": "regression and rollback",
            }
        ],
    }


def readme() -> str:
    rows = "\n".join(
        f"| `{component.role}` | `{component.payload_path}` | "
        f"{component.quantization['name']} | "
        f"{'public component' if component.public else 'internal regression/rollback'} |"
        for component in COMPONENTS
    )
    return f"""---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
library_name: gem16
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
base_model_relation: quantized
tags:
- gemma
- gem16
- quantized
- fp8
- nvfp4
- sm120
---

# Gemma 4 26B A4B IT — GEM16

This repository contains the GEM16-native components for the **Gemma 4 26B
A4B Compact Vision** profile on NVIDIA Blackwell SM120.  The payload filenames
state the real model and storage/quantization format; there is no anonymous
`model.gem16` payload in this revision.

| Role | Artifact | Quantization / storage | Product status |
|---|---|---|---|
{rows}

The public profile combines the Trellis35 W4A8 text Target with the FP8
E4M3FN Vision component.  The fixed-D2 Assistant is optional.  The larger
hybrid NVFP4 Target remains available under `internal/` for regression and
rollback, but it is not a normal Studio product choice.

Trellis35 is EXL3-derived offline packing in GEM16's native direct-load format;
its routed experts use mixed K3/K4 storage at approximately 3.5 bits per
weight.  Vision uses FP8 E4M3FN linear weights with BF16 row scales and BF16
support tensors—it is not Q8.  The Assistant and internal Target are hybrid
NVFP4/FP8/BF16 artifacts.

`profiles.json` is the machine-readable composition contract.  Each component
has a `component.json` with its exact artifact identity and quantization.  The
`compilation.json` and `source.lock.json` files preserve compiler and immutable
source provenance.

These are compiled GEM16 artifacts, not Transformers, GGUF, or generic
Safetensors checkpoints.  Runtime model hashing is deliberately disabled;
integrity is established during download/install and recorded in the locks.

Engine and usage documentation: https://github.com/Danmoreng/gem16
"""


def notice() -> str:
    return """GEM16 Gemma 4 26B A4B compiled components

This distribution contains modified model artifacts derived from:
  google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
  revision f1e06dc520982d9b9edd76859fdb7ab209449949
and its separately pinned Assistant source recorded in the component metadata.

The upstream Gemma model is provided by Google LLC under the Apache License,
Version 2.0. GEM16 modifications include offline quantization, compilation and
tensor-layout conversion for the GEM16 SM120 runtime. The public Compact
Vision composition and internal regression/rollback component are recorded in
profiles.json; exact artifact and source identities are in each component's
component.json, compilation.json and source.lock.json.

GEM16 source and documentation: https://github.com/Danmoreng/gem16
Upstream terms and notices: https://ai.google.dev/gemma/terms
"""


def normalized_index(source_index: dict[str, Any], component: Component) -> dict[str, Any]:
    old = component.source_payload
    new = component.payload_name
    weight_map = source_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Assistant Safetensors index has no weight_map")
    if any(value != old for value in weight_map.values()):
        raise ValueError("Assistant Safetensors index references an unexpected shard")
    result = dict(source_index)
    result["weight_map"] = {key: new for key in weight_map}
    return result


def build_plan() -> dict[str, Any]:
    copies: list[dict[str, str]] = []
    generated: dict[str, Any] = {
        "README.md": readme(),
        "NOTICE": notice(),
        "profiles.json": profiles_document(),
    }
    for component in COMPONENTS:
        copies.append({
            "source": source_path(component, component.source_payload),
            "destination": component.payload_path,
        })
        for name in component.auxiliary_files:
            copies.append({
                "source": source_path(component, name),
                "destination": f"{component.directory}/{renamed_auxiliary(name)}",
            })
        generated[f"{component.directory}/component.json"] = component_document(component)
    return {
        "repository": REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "copies": copies,
        "generated": generated,
        "delete_folders": ["assistant", "trellis35", "vision"],
        "delete_files": list(ROOT_FILES_TO_REMOVE),
    }


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def validate_remote_files(siblings: list[Any]) -> None:
    """Validate Hub metadata without reading or hashing model payload bytes."""
    by_path = {sibling.rfilename: sibling for sibling in siblings}
    for component in COMPONENTS:
        path = source_path(component, component.source_payload)
        sibling = by_path.get(path)
        if sibling is None:
            raise ValueError(f"missing immutable source payload: {path}")
        if sibling.size != component.payload_bytes:
            raise ValueError(f"source payload size mismatch: {path}")
        lfs = sibling.lfs
        if lfs is None or lfs.sha256 != component.payload_sha256:
            raise ValueError(f"source payload LFS identity mismatch: {path}")
    for item in build_plan()["copies"]:
        if item["source"] not in by_path:
            raise ValueError(f"missing immutable source file: {item['source']}")


def publish(plan: dict[str, Any]) -> str:
    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationCopy,
        CommitOperationDelete,
        HfApi,
        hf_hub_download,
    )

    api = HfApi()
    source_info = api.model_info(
        REPOSITORY, revision=SOURCE_REVISION, files_metadata=True
    )
    if source_info.sha != SOURCE_REVISION:
        raise ValueError("Hub returned an unexpected immutable source revision")
    validate_remote_files(source_info.siblings)

    assistant = COMPONENTS[2]
    index_path = hf_hub_download(
        repo_id=REPOSITORY,
        filename="assistant/model.safetensors.index.json",
        revision=SOURCE_REVISION,
    )
    with open(index_path, encoding="utf-8") as stream:
        source_index = json.load(stream)
    plan["generated"][
        f"{assistant.directory}/model.safetensors.index.json"
    ] = normalized_index(source_index, assistant)

    operations = [
        CommitOperationCopy(
            src_path_in_repo=item["source"],
            path_in_repo=item["destination"],
            src_revision=SOURCE_REVISION,
        )
        for item in plan["copies"]
    ]
    for path, value in plan["generated"].items():
        payload = value.encode("utf-8") if isinstance(value, str) else json_bytes(value)
        operations.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=payload))
    operations.extend(
        CommitOperationDelete(path_in_repo=path, is_folder=True)
        for path in plan["delete_folders"]
    )
    operations.extend(
        CommitOperationDelete(path_in_repo=path, is_folder=False)
        for path in plan["delete_files"]
    )
    result = api.create_commit(
        repo_id=REPOSITORY,
        operations=operations,
        commit_message="Normalize GEM16 component names and product profiles",
        commit_description=(
            "Give every model payload an explicit Gemma 4 26B and quantization "
            "name; expose the Compact Vision composition while retaining the "
            "qualified NVFP4 Target under internal/. Large blobs are copied "
            "server-side from the immutable source revision without re-encoding."
        ),
        parent_commit=SOURCE_REVISION,
    )
    return result.oid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publish",
        action="store_true",
        help="create the normalized Hub commit (default: print the deterministic plan)",
    )
    arguments = parser.parse_args()
    plan = build_plan()
    if arguments.publish:
        print(publish(plan))
    else:
        print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
