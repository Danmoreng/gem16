#!/usr/bin/env python3
"""Audit locked Gemma 4 26B source structure and tokenizer contracts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


MAX_JSON_BYTES = 64 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qat-model", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--unsloth-model", type=Path, required=True)
    parser.add_argument("--qat-lock", type=Path, required=True)
    parser.add_argument("--base-lock", type=Path, required=True)
    parser.add_argument("--unsloth-lock", type=Path, required=True)
    parser.add_argument("--q4-lock", type=Path, required=True)
    parser.add_argument("--software-lock", type=Path, required=True)
    parser.add_argument("--qat-inventory", type=Path, required=True)
    parser.add_argument("--base-inventory", type=Path, required=True)
    parser.add_argument("--unsloth-inventory", type=Path, required=True)
    parser.add_argument("--q4-inventory", type=Path, required=True)
    parser.add_argument("--source-prompts", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        raise ValueError(f"JSON input exceeds {MAX_JSON_BYTES} bytes: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def differing_paths(first: Any, second: Any, prefix: str = "") -> list[str]:
    if isinstance(first, dict) and isinstance(second, dict):
        differences: list[str] = []
        for key in sorted(set(first) | set(second)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in first or key not in second:
                differences.append(path)
            else:
                differences.extend(differing_paths(first[key], second[key], path))
        return differences
    return [] if first == second else [prefix or "<root>"]


def tensor_contract(inventory: dict[str, Any]) -> dict[str, tuple[str, tuple[int, ...], int]]:
    tensors = inventory.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError("Safetensors inventory has no tensor array")
    contract: dict[str, tuple[str, tuple[int, ...], int]] = {}
    for tensor in tensors:
        if not isinstance(tensor, dict):
            raise ValueError("invalid tensor inventory entry")
        name = tensor.get("name")
        shape = tensor.get("shape")
        if not isinstance(name, str) or not name or name in contract:
            raise ValueError(f"missing or duplicate tensor name: {name!r}")
        if not isinstance(shape, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in shape
        ):
            raise ValueError(f"invalid shape for tensor {name}")
        contract[name] = (
            str(tensor.get("dtype")),
            tuple(shape),
            int(tensor.get("bytes")),
        )
    return contract


def summarize_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    tensors = inventory["tensors"]
    dtypes = Counter(str(tensor["dtype"]) for tensor in tensors)
    return {
        "tensor_count": len(tensors),
        "tensor_payload_bytes": sum(int(tensor["bytes"]) for tensor in tensors),
        "tensor_count_by_dtype": dict(sorted(dtypes.items())),
    }


def compare_contracts(
    first: dict[str, tuple[str, tuple[int, ...], int]],
    second: dict[str, tuple[str, tuple[int, ...], int]],
) -> dict[str, Any]:
    common = sorted(set(first) & set(second))
    mismatched = [name for name in common if first[name] != second[name]]
    return {
        "exact": not mismatched and set(first) == set(second),
        "common_tensor_count": len(common),
        "first_only_count": len(set(first) - set(second)),
        "second_only_count": len(set(second) - set(first)),
        "structural_mismatch_count": len(mismatched),
        "first_only_examples": sorted(set(first) - set(second))[:10],
        "second_only_examples": sorted(set(second) - set(first))[:10],
        "structural_mismatch_examples": mismatched[:10],
    }


def unsloth_layout_summary(inventory: dict[str, Any]) -> dict[str, Any]:
    suffix_dtype: Counter[tuple[str, str]] = Counter()
    for tensor in inventory["tensors"]:
        name = str(tensor["name"])
        suffix = next(
            (
                candidate
                for candidate in (
                    "input_global_scale",
                    "weight_global_scale",
                    "weight_packed",
                    "weight_scale",
                    "weight",
                )
                if name.endswith(candidate)
            ),
            "other",
        )
        suffix_dtype[(suffix, str(tensor["dtype"]))] += 1
    return {
        "tensor_count_by_role_and_dtype": {
            f"{suffix}:{dtype}": count
            for (suffix, dtype), count in sorted(suffix_dtype.items())
        }
    }


def expected_q4_metadata(text_config: dict[str, Any]) -> dict[str, Any]:
    layer_types = text_config["layer_types"]
    return {
        "gemma4.block_count": text_config["num_hidden_layers"],
        "gemma4.context_length": text_config["max_position_embeddings"],
        "gemma4.embedding_length": text_config["hidden_size"],
        "gemma4.feed_forward_length": text_config["intermediate_size"],
        "gemma4.attention.head_count": text_config["num_attention_heads"],
        "gemma4.attention.head_count_kv": [
            text_config["num_global_key_value_heads"]
            if layer_type == "full_attention"
            else text_config["num_key_value_heads"]
            for layer_type in layer_types
        ],
        "gemma4.attention.sliding_window": text_config["sliding_window"],
        "gemma4.final_logit_softcapping": text_config["final_logit_softcapping"],
    }


def validate_inventory_lock(inventory: dict[str, Any], lock_path: Path) -> bool:
    source = inventory.get("source")
    return isinstance(source, dict) and source.get("lock_sha256") == file_sha256(lock_path)


def validate_q4_lock(q4_inventory: dict[str, Any], q4_lock: dict[str, Any]) -> bool:
    gguf = q4_inventory.get("gguf")
    if not isinstance(gguf, dict):
        return False
    return any(
        entry.get("path") == Path(str(gguf.get("path"))).name
        and entry.get("sha256") == gguf.get("sha256")
        and entry.get("size") == gguf.get("size_bytes")
        for entry in q4_lock.get("files", [])
        if isinstance(entry, dict)
    )


def source_user(record: dict[str, Any]) -> str:
    if isinstance(record.get("user"), str):
        return record["user"]
    repeat = record.get("user_repeat")
    if not isinstance(repeat, dict) or not isinstance(repeat.get("text"), str):
        raise ValueError(f"record {record.get('id')!r} has no valid user content")
    count = repeat.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or count > 100_000:
        raise ValueError(f"record {record.get('id')!r} has invalid repeat count")
    return repeat["text"] * count


def tokenizer_audit(
    qat_model: Path,
    base_model: Path,
    unsloth_model: Path,
    source_prompts: dict[str, Any],
) -> dict[str, Any]:
    import transformers  # Pinned reference environment only.
    from transformers import AutoTokenizer

    tokenizers = {
        "qat_bf16": AutoTokenizer.from_pretrained(
            qat_model, local_files_only=True, trust_remote_code=False
        ),
        "ordinary_bf16": AutoTokenizer.from_pretrained(
            base_model, local_files_only=True, trust_remote_code=False
        ),
        "unsloth_nvfp4": AutoTokenizer.from_pretrained(
            unsloth_model, local_files_only=True, trust_remote_code=False
        ),
    }
    mismatches: list[dict[str, str]] = []
    records = source_prompts.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("source prompt document has no records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("source prompt record must be an object")
        messages = [{"role": "user", "content": source_user(record)}]
        rendered = {
            name: tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for name, tokenizer in tokenizers.items()
        }
        canonical = rendered["qat_bf16"]
        canonical_ids = tokenizers["qat_bf16"].encode(
            canonical, add_special_tokens=False
        )
        for name in ("ordinary_bf16", "unsloth_nvfp4"):
            if rendered[name] != canonical:
                mismatches.append(
                    {"record": str(record.get("id")), "source": name, "kind": "render"}
                )
                continue
            ids = tokenizers[name].encode(rendered[name], add_special_tokens=False)
            if ids != canonical_ids:
                mismatches.append(
                    {"record": str(record.get("id")), "source": name, "kind": "tokens"}
                )
    return {
        "record_count": len(records),
        "transformers_version": transformers.__version__,
        "trust_remote_code": False,
        "qat_ordinary_unsloth_exact": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def validate_frozen_q4_parity(
    corpus_dir: Path,
    qat_lock: Path,
    base_lock: Path,
    q4_lock: Path,
) -> dict[str, Any]:
    expected = {
        "qat_lock_sha256": file_sha256(qat_lock),
        "base_lock_sha256": file_sha256(base_lock),
        "q4_lock_sha256": file_sha256(q4_lock),
    }
    split_lock_path = corpus_dir / "splits.lock.json"
    split_lock = load_json(split_lock_path)
    valid = split_lock.get("status") == "frozen" and split_lock.get("split_audit", {}).get("status") == "pass"
    record_count = 0
    for entry in split_lock.get("files", []):
        path = corpus_dir / str(entry.get("path"))
        if (
            not path.is_file()
            or path.stat().st_size != entry.get("bytes")
            or file_sha256(path) != entry.get("sha256")
        ):
            valid = False
            continue
        document = load_json(path)
        provenance = document.get("provenance", {})
        if any(provenance.get(key) != value for key, value in expected.items()):
            valid = False
        contract = str(provenance.get("tokenizer_contract", ""))
        if "official Q4_0 exact parity required" not in contract:
            valid = False
        records = document.get("records")
        if not isinstance(records, list):
            valid = False
        else:
            record_count += len(records)
    return {
        "status": "pass" if valid else "fail",
        "record_count": record_count,
        "split_lock_sha256": file_sha256(split_lock_path),
        "method": "pinned llama-tokenize exact token-ID comparison retained by frozen corpus",
    }


def main() -> int:
    args = parse_args()
    qat_lock = load_json(args.qat_lock)
    base_lock = load_json(args.base_lock)
    unsloth_lock = load_json(args.unsloth_lock)
    q4_lock = load_json(args.q4_lock)
    software_lock = load_json(args.software_lock)
    qat_inventory = load_json(args.qat_inventory)
    base_inventory = load_json(args.base_inventory)
    unsloth_inventory = load_json(args.unsloth_inventory)
    q4_inventory = load_json(args.q4_inventory)
    qat_config = load_json(args.qat_model / "config.json")
    base_config = load_json(args.base_model / "config.json")
    unsloth_config = load_json(args.unsloth_model / "config.json")

    qat_contract = tensor_contract(qat_inventory)
    base_contract = tensor_contract(base_inventory)
    unsloth_contract = tensor_contract(unsloth_inventory)
    qat_base = compare_contracts(qat_contract, base_contract)
    qat_unsloth = compare_contracts(qat_contract, unsloth_contract)
    text_configs_equal = (
        qat_config.get("text_config")
        == base_config.get("text_config")
        == unsloth_config.get("text_config")
    )
    expected_metadata = expected_q4_metadata(qat_config["text_config"])
    actual_metadata = q4_inventory.get("selected_metadata", {})
    q4_mismatches = [
        key for key, value in expected_metadata.items() if actual_metadata.get(key) != value
    ]
    tokenizer = tokenizer_audit(
        args.qat_model,
        args.base_model,
        args.unsloth_model,
        load_json(args.source_prompts),
    )
    frozen_q4 = validate_frozen_q4_parity(
        args.corpus_dir, args.qat_lock, args.base_lock, args.q4_lock
    )
    linked_locks = {
        "qat_bf16": validate_inventory_lock(qat_inventory, args.qat_lock),
        "ordinary_bf16": validate_inventory_lock(base_inventory, args.base_lock),
        "unsloth_nvfp4": validate_inventory_lock(unsloth_inventory, args.unsloth_lock),
        "official_q4_0": validate_q4_lock(q4_inventory, q4_lock),
    }
    references = {
        entry.get("name"): entry
        for entry in software_lock.get("references", [])
        if isinstance(entry, dict)
    }
    expected_transformers = references.get("transformers", {}).get("version")
    quantization = unsloth_config.get("quantization_config", {})
    quantization_contract = {
        "quant_method": quantization.get("quant_method"),
        "format": quantization.get("format"),
        "status": quantization.get("quantization_status"),
        "producer_version": quantization.get("version"),
        "config_group_names": sorted(quantization.get("config_groups", {})),
        "kv_cache_num_bits": quantization.get("kv_cache_scheme", {}).get("num_bits"),
    }
    status_checks = [
        all(linked_locks.values()),
        qat_base["exact"],
        text_configs_equal,
        not q4_mismatches,
        tokenizer["qat_ordinary_unsloth_exact"],
        tokenizer["transformers_version"] == expected_transformers,
        frozen_q4["status"] == "pass",
        quantization_contract["quant_method"] == "compressed-tensors",
        quantization_contract["format"] == "mixed-precision",
    ]
    report = {
        "schema_version": 1,
        "status": "pass" if all(status_checks) else "fail",
        "input_sha256": {
            path.as_posix(): file_sha256(path)
            for path in (
                args.qat_lock,
                args.base_lock,
                args.unsloth_lock,
                args.q4_lock,
                args.software_lock,
                args.qat_inventory,
                args.base_inventory,
                args.unsloth_inventory,
                args.q4_inventory,
                args.source_prompts,
                args.corpus_dir / "splits.lock.json",
            )
        },
        "source_revisions": {
            "qat_bf16": qat_lock["revision"],
            "ordinary_bf16": base_lock["revision"],
            "unsloth_nvfp4": unsloth_lock["revision"],
            "official_q4_0": q4_lock["revision"],
        },
        "inventory_lock_identity": linked_locks,
        "architecture": {
            "safetensors_text_config_exact": text_configs_equal,
            "qat_vs_ordinary_config_difference_paths": differing_paths(
                qat_config, base_config
            ),
            "qat_vs_unsloth_config_difference_paths": differing_paths(
                qat_config, unsloth_config
            ),
            "official_q4_0_metadata_exact": not q4_mismatches,
            "official_q4_0_metadata_mismatch_paths": q4_mismatches,
        },
        "inventories": {
            "qat_bf16": summarize_inventory(qat_inventory),
            "ordinary_bf16": summarize_inventory(base_inventory),
            "unsloth_nvfp4": summarize_inventory(unsloth_inventory),
            "official_q4_0": {
                "tensor_count": q4_inventory["gguf"]["tensor_count"],
                "file_bytes": q4_inventory["gguf"]["size_bytes"],
                "tensor_count_and_bytes_by_type": q4_inventory["totals_by_type"],
            },
        },
        "qat_vs_ordinary_bf16_tensor_contract": qat_base,
        "qat_vs_unsloth_raw_name_contract": qat_unsloth,
        "unsloth_quantization": {
            **quantization_contract,
            **unsloth_layout_summary(unsloth_inventory),
        },
        "tokenizer": {
            **tokenizer,
            "official_q4_0": frozen_q4,
        },
        "interpretation": {
            "qat_vs_ordinary": "identical tensor names, dtypes, shapes and byte lengths; source shard hashes differ",
            "unsloth": "same text architecture; producer-specific FP8 attention and NVFP4 MLP/expert layout",
            "official_q4_0": "same architecture metadata; GGUF names and quantized storage are not a Safetensors layout contract",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: status={report['status']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
