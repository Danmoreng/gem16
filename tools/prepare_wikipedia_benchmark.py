#!/usr/bin/env python3
"""Create an exact-token long-context prompt from a pinned Wikipedia revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import urllib.parse
import urllib.request
from typing import Any, Mapping


DEFAULT_TITLE = "Artificial intelligence"
DEFAULT_REVISION = 1366077412
DEFAULT_REVISION_TIMESTAMP = "2026-07-26T03:46:54Z"
DEFAULT_PROMPT_TOKENS = 16384
DEFAULT_MAX_NEW_TOKENS = 8192
INSTRUCTION = (
    "Fasse den folgenden Wikipedia-Artikel auf Deutsch zusammen. "
    "Nenne die wichtigsten Konzepte, historischen Entwicklungen, "
    "Anwendungsfelder, Chancen und Risiken. Verwende klare Überschriften "
    "und bleibe ausschließlich beim Inhalt des Artikels."
)


class PreparationError(RuntimeError):
    pass


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--revision", type=positive_int, default=DEFAULT_REVISION)
    parser.add_argument(
        "--prompt-tokens", type=positive_int, default=DEFAULT_PROMPT_TOKENS
    )
    parser.add_argument(
        "--max-new-tokens", type=positive_int, default=DEFAULT_MAX_NEW_TOKENS
    )
    return parser.parse_args()


def fetch_extract(title: str, revision: int) -> tuple[str, dict[str, Any]]:
    parameters = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "extracts|revisions",
            "explaintext": 1,
            "exsectionformat": "plain",
            "revids": revision,
            "rvprop": "ids|timestamp",
        }
    )
    url = f"https://en.wikipedia.org/w/api.php?{parameters}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "gem16gb-benchmark/1.0 (https://github.com/Danmoreng/gem16gb)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        document = json.load(response)
    pages = document.get("query", {}).get("pages")
    if not isinstance(pages, list) or len(pages) != 1:
        raise PreparationError("Wikipedia response did not contain exactly one page")
    page = pages[0]
    revisions = page.get("revisions")
    extract = page.get("extract")
    if (
        page.get("title") != title
        or not isinstance(revisions, list)
        or len(revisions) != 1
        or revisions[0].get("revid") != revision
        or not isinstance(extract, str)
        or not extract
    ):
        raise PreparationError("Wikipedia response does not match the pinned revision")
    if (
        revision == DEFAULT_REVISION
        and revisions[0].get("timestamp") != DEFAULT_REVISION_TIMESTAMP
    ):
        raise PreparationError("Wikipedia revision timestamp differs from the pin")
    return extract, {
        "api_url": url,
        "page_id": page.get("pageid"),
        "title": title,
        "revision": revision,
        "revision_timestamp": revisions[0].get("timestamp"),
        "article_url": (
            "https://en.wikipedia.org/w/index.php?"
            + urllib.parse.urlencode({"title": title, "oldid": revision})
        ),
    }


def encode_chat(tokenizer: Any, content: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if not isinstance(encoded, list) or not all(isinstance(token, int) for token in encoded):
        raise PreparationError("tokenizer returned malformed chat token IDs")
    return encoded


def common_suffix(left: list[int], right: list[int]) -> list[int]:
    count = 0
    limit = min(len(left), len(right))
    while count < limit and left[-1 - count] == right[-1 - count]:
        count += 1
    if count == 0:
        raise PreparationError("chat template has no stable generation suffix")
    return left[len(left) - count :]


def token_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    try:
        from transformers import AutoTokenizer

        model = args.model.resolve(strict=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        extract, source = fetch_extract(args.title, args.revision)
        content = (
            f"{INSTRUCTION}\n\n"
            f"Wikipedia-Artikel: {args.title}\n\n"
            f"{extract}"
        )
        full_tokens = encode_chat(tokenizer, content)
        empty_tokens = encode_chat(tokenizer, "")
        suffix = common_suffix(full_tokens, empty_tokens)
        prefix_count = args.prompt_tokens - len(suffix)
        if prefix_count <= 0 or prefix_count >= len(full_tokens) - len(suffix):
            raise PreparationError(
                "article is too short for the requested prompt after preserving the chat suffix"
            )
        prompt_tokens = full_tokens[:prefix_count] + suffix
        if len(prompt_tokens) != args.prompt_tokens:
            raise PreparationError("prepared prompt does not have the requested token count")

        generation = json.loads(
            (model / "generation_config.json").read_text(encoding="utf-8")
        )
        stop_tokens = generation.get("eos_token_id")
        if isinstance(stop_tokens, int):
            stop_tokens = [stop_tokens]
        suppress_tokens = generation.get("suppress_tokens", [])
        if (
            not isinstance(stop_tokens, list)
            or not stop_tokens
            or not all(isinstance(token, int) for token in stop_tokens)
            or not isinstance(suppress_tokens, list)
            or not all(isinstance(token, int) for token in suppress_tokens)
        ):
            raise PreparationError("checkpoint generation controls are malformed")

        document = {
            "schema_version": 1,
            "id": "wikipedia-artificial-intelligence-summary-16k",
            "description": (
                "Pinned Wikipedia summarization workload with an exact shared "
                "Gemma token prompt."
            ),
            "source": {
                **source,
                "license": "CC BY-SA 4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "extract_utf8_sha256": hashlib.sha256(
                    extract.encode("utf-8")
                ).hexdigest(),
                "extract_characters": len(extract),
            },
            "prompt": {
                "language": "German instruction over an English article",
                "instruction": INSTRUCTION,
                "enable_thinking": False,
                "target_tokens": args.prompt_tokens,
                "full_untruncated_tokens": len(full_tokens),
                "preserved_chat_suffix_tokens": len(suffix),
                "token_ids_encoding": "JSON integers; SHA-256 hashes uint32 little-endian",
                "token_ids_sha256": token_sha256(prompt_tokens),
                "token_ids": prompt_tokens,
            },
            "generation": {
                "temperature": 0.0,
                "seed": 0,
                "max_new_tokens": args.max_new_tokens,
                "stop_token_ids": stop_tokens,
                "suppress_token_ids": suppress_tokens,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "revision": args.revision,
                    "extract_characters": len(extract),
                    "full_tokens": len(full_tokens),
                    "prompt_tokens": len(prompt_tokens),
                    "suffix_tokens": len(suffix),
                    "token_ids_sha256": document["prompt"]["token_ids_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (
        PreparationError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
