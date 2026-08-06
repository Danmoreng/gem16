#!/usr/bin/env python3
"""Freeze disjoint Gemma 4 26B prompts against three locked tokenizers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--qat-model", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--q4-model", type=Path, required=True)
    parser.add_argument("--qat-lock", type=Path, required=True)
    parser.add_argument("--base-lock", type=Path, required=True)
    parser.add_argument("--q4-lock", type=Path, required=True)
    parser.add_argument("--llama-tokenize", type=Path, required=True)
    parser.add_argument("--llama-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token in token_ids:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0 or token > 0xFFFFFFFF:
            raise ValueError(f"invalid token ID: {token!r}")
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def expanded_user(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(record.get("user"), str):
        return record["user"], {"kind": "literal", "text": record["user"]}
    repeat = record.get("user_repeat")
    if not isinstance(repeat, dict) or not isinstance(repeat.get("text"), str):
        raise ValueError(f"record {record.get('id')!r} has no valid user content")
    count = repeat.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or count > 100_000:
        raise ValueError(f"record {record.get('id')!r} has invalid repeat count")
    return repeat["text"] * count, {
        "kind": "repeat",
        "text": repeat["text"],
        "count": count,
    }


def q4_tokenize(executable: Path, model: Path, rendered: str) -> list[int]:
    command = [
        str(executable),
        "--offline",
        "--model",
        str(model),
        "--stdin",
        "--ids",
        "--no-bos",
    ]
    completed = subprocess.run(
        command,
        input=rendered,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"llama-tokenize failed with {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    try:
        tokens = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("llama-tokenize did not return a JSON token array") from error
    if not isinstance(tokens, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in tokens
    ):
        raise RuntimeError("llama-tokenize returned invalid token IDs")
    return tokens


def audit_disjoint(records_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, object]:
    owners: dict[tuple[str, str], str] = {}
    overlaps: list[dict[str, str]] = []
    for split, records in records_by_split.items():
        for record in records:
            for kind, key in (
                ("document", "expanded_user_sha256"),
                ("token_span", "input_token_ids_sha256_u32le"),
            ):
                identity = (kind, str(record[key]))
                previous = owners.get(identity)
                if previous is not None and previous != split:
                    overlaps.append(
                        {
                            "kind": kind,
                            "sha256": identity[1],
                            "first_split": previous,
                            "second_split": split,
                        }
                    )
                owners[identity] = split
    return {
        "status": "pass" if not overlaps else "fail",
        "document_or_token_span_overlap_count": len(overlaps),
        "overlaps": overlaps,
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    from transformers import AutoTokenizer  # Imported only in the pinned reference environment.

    canonical = AutoTokenizer.from_pretrained(
        args.qat_model, local_files_only=True, trust_remote_code=False
    )
    base = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=False
    )
    source = json.loads(args.source.read_text(encoding="utf-8"))
    records = source.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("source prompt document has no records")
    seen_ids: set[str] = set()
    by_split: dict[str, list[dict[str, Any]]] = {
        "calibration": [],
        "development": [],
        "test": [],
    }
    for candidate in records:
        if not isinstance(candidate, dict):
            raise ValueError("source prompt record must be an object")
        record_id = str(candidate.get("id", ""))
        if not record_id or record_id in seen_ids:
            raise ValueError(f"missing or duplicate record ID: {record_id!r}")
        seen_ids.add(record_id)
        split = str(candidate.get("split", ""))
        if split not in by_split:
            raise ValueError(f"invalid split for {record_id!r}: {split!r}")
        user, source_spec = expanded_user(candidate)
        messages = [{"role": "user", "content": user}]
        rendered = canonical.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        base_rendered = base.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if base_rendered != rendered:
            raise RuntimeError(f"QAT/base chat template mismatch for {record_id}")
        qat_ids = canonical.encode(rendered, add_special_tokens=False)
        base_ids = base.encode(rendered, add_special_tokens=False)
        if base_ids != qat_ids:
            raise RuntimeError(f"QAT/base token mismatch for {record_id}")
        q4_ids = q4_tokenize(args.llama_tokenize, args.q4_model, rendered)
        if q4_ids != qat_ids:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(qat_ids, q4_ids))
                    if pair[0] != pair[1]
                ),
                min(len(qat_ids), len(q4_ids)),
            )
            raise RuntimeError(
                f"QAT/Q4 token mismatch for {record_id} at {mismatch}: "
                f"qat_count={len(qat_ids)} q4_count={len(q4_ids)}"
            )
        last = len(qat_ids) - 1
        positions = sorted({0, min(1023, last), min(1024, last), last})
        by_split[split].append(
            {
                "id": record_id,
                "category": candidate.get("category"),
                "source": source_spec,
                "expanded_user_sha256": text_sha256(user),
                "rendered_prompt_sha256": text_sha256(rendered),
                "input_token_ids": qat_ids,
                "input_token_count": len(qat_ids),
                "input_token_ids_sha256_u32le": token_sha256(qat_ids),
                "selected_positions": positions,
                "selected_layers": [0, 5, 6, 29],
            }
        )
    audit = audit_disjoint(by_split)
    if audit["status"] != "pass":
        raise RuntimeError("calibration/development/test split overlap detected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source_prompt_sha256": file_sha256(args.source),
        "qat_lock_sha256": file_sha256(args.qat_lock),
        "base_lock_sha256": file_sha256(args.base_lock),
        "q4_lock_sha256": file_sha256(args.q4_lock),
        "llama_cpp_revision": args.llama_revision,
        "tokenizer_contract": "QAT BF16 canonical; ordinary BF16 and official Q4_0 exact parity required",
    }
    written: list[Path] = []
    for split, split_records in by_split.items():
        document = {
            "schema_version": 1,
            "split": split,
            "policy": (
                "held_out_final_quality_no_quantizer_tuning"
                if split == "test"
                else "quantizer_tuning_permitted" if split == "calibration" else "development_only"
            ),
            "provenance": provenance,
            "records": split_records,
        }
        path = args.output_dir / f"{split}.json"
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    lock = {
        "schema_version": 1,
        "status": "frozen",
        "split_audit": audit,
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in written
        ],
    }
    (args.output_dir / "splits.lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "frozen corpus: "
        + ", ".join(f"{split}={len(records)}" for split, records in by_split.items())
        + "; tokenizer parity PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
