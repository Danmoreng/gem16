#!/usr/bin/env python3
"""Validate Gemma 4 MTP drafting against Transformers and exact target output.

This bounded correctness harness deliberately uses a one-token BF16-cache
context so the target Layer-46/47 K/V states captured by gem16 are the complete
shared cache consumed by the assistant reference. Performance is out of scope.
"""

from __future__ import annotations

try:
    from tools.hf_cache import default_assistant_model, default_target_model
except ModuleNotFoundError:
    from hf_cache import default_assistant_model, default_target_model

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from compare_states import load_state


TARGET_NORM = "model.language_model.norm.weight"
TARGET_EMBEDDING = "model.language_model.embed_tokens.weight"
SUPPRESSED_ASSISTANT_TOKENS = (258883, 258882)


def run_engine(binary: Path, arguments: list[str]) -> dict:
    completed = subprocess.run(
        [str(binary), *arguments],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def target_embedding_row(model_file: Path, token: int) -> torch.Tensor:
    with safe_open(model_file, framework="pt", device="cpu") as tensors:
        row = tensors.get_slice(TARGET_EMBEDDING)[token : token + 1]
    scale = torch.tensor(math.sqrt(3840), dtype=torch.bfloat16, device="cuda")
    return (row.cuda() * scale).to(torch.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=default_target_model())
    parser.add_argument("--assistant", type=Path, default=default_assistant_model())
    parser.add_argument("--input-token", type=int, default=2)
    parser.add_argument("--draft-tokens", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    target_file = args.target / "model.safetensors"
    if not target_file.is_file():
        raise SystemExit("the pinned single-file target Safetensors is required")

    common = [
        "--model",
        str(args.target),
        "--input-token-ids",
        str(args.input_token),
        "--max-context",
        "8",
        "--kv-cache",
        "bf16",
        "--greedy",
    ]
    with tempfile.TemporaryDirectory(prefix="gem16-mtp-reference-") as temporary:
        state_path = Path(temporary) / "target.state"
        state_result = run_engine(
            args.binary,
            [
                *common,
                "--max-tokens",
                "1",
                "--dump-state",
                str(state_path),
                "--dump-state-position",
                "0",
            ],
        )
        # A target batch over D drafts yields D draft predictions plus one
        # target-only prediction when every draft matches. Request that extra
        # output position so the first proposal group always has D drafts.
        generated_tokens = args.draft_tokens + 2
        ordinary = run_engine(
            args.binary,
            [*common, "--max-tokens", str(generated_tokens)],
        )
        mtp = run_engine(
            args.binary,
            [
                *common,
                "--assistant-model",
                str(args.assistant),
                "--mtp-draft-tokens",
                str(args.draft_tokens),
                "--max-tokens",
                str(generated_tokens),
            ],
        )
        state = load_state(state_path)

    if ordinary["output_token_ids"] != mtp["output_token_ids"]:
        raise SystemExit("MTP output differs from ordinary target greedy output")
    if state_result["output_token_ids"][0] != mtp["output_token_ids"][0]:
        raise SystemExit("diagnostic and production prefill disagree on the first token")
    if mtp["fallbacks"] != 0 or mtp["token_loop_allocations"] is not False:
        raise SystemExit("MTP run used a fallback or reported a token-loop allocation")
    statistics = mtp["mtp"]
    if statistics["target_batches"] == 0:
        raise SystemExit("batched verification did not execute a target batch")
    if statistics["target_forwards"] < len(mtp["output_token_ids"]) - 1:
        raise SystemExit("batched verification evaluated too few target positions")
    if statistics["proposed_tokens"] != (
        statistics["accepted_tokens"] + statistics["rejected_tokens"]
    ):
        raise SystemExit("MTP proposal accounting is inconsistent")

    with safe_open(target_file, framework="pt", device="cpu") as tensors:
        final_norm = tensors.get_tensor(TARGET_NORM).cuda()
    final_residual = torch.tensor(
        state.layers[47].hidden, dtype=torch.float32, device="cuda"
    ).to(torch.bfloat16)
    feedback = torch.nn.functional.rms_norm(
        final_residual, (3840,), final_norm, 1.0e-6
    ).to(torch.bfloat16).unsqueeze(0)

    sliding = tuple(
        torch.tensor(
            getattr(state.layers[46], field), dtype=torch.float32, device="cuda"
        ).to(torch.bfloat16).reshape(1, 8, 1, 256)
        for field in ("key", "value")
    )
    full = tuple(
        torch.tensor(
            getattr(state.layers[47], field), dtype=torch.float32, device="cuda"
        ).to(torch.bfloat16).reshape(1, 1, 1, 512)
        for field in ("key", "value")
    )
    shared_kv = {"sliding_attention": sliding, "full_attention": full}

    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=torch.bfloat16
    ).cuda().eval()
    token = mtp["output_token_ids"][0]
    reference_drafts: list[int] = []
    with torch.inference_mode():
        for _ in range(args.draft_tokens):
            embedding = target_embedding_row(target_file, token)
            combined = torch.cat([embedding, feedback], dim=-1).unsqueeze(0)
            output = assistant(
                inputs_embeds=combined,
                position_ids=torch.tensor([[0]], dtype=torch.long, device="cuda"),
                shared_kv_states=shared_kv,
                use_cache=False,
            )
            logits = output.logits[0, 0].float()
            for suppressed in SUPPRESSED_ASSISTANT_TOKENS:
                logits[suppressed] = -float("inf")
            token = int(logits.argmax())
            reference_drafts.append(token)
            feedback = output.last_hidden_state[0].to(torch.bfloat16)

    gem16_drafts = mtp["mtp"]["proposed_token_ids"][: args.draft_tokens]
    if reference_drafts != gem16_drafts:
        raise SystemExit(
            f"assistant draft mismatch: Transformers={reference_drafts}, "
            f"gem16={gem16_drafts}"
        )

    report = {
        "schema_version": 1,
        "status": "pass",
        "mode": "mtp_correctness",
        "kv_cache_mode": "bf16",
        "input_token": args.input_token,
        "draft_tokens": args.draft_tokens,
        "ordinary_output_token_ids": ordinary["output_token_ids"],
        "mtp_output_token_ids": mtp["output_token_ids"],
        "transformers_draft_token_ids": reference_drafts,
        "gem16_draft_token_ids": gem16_drafts,
        "mtp_statistics": mtp["mtp"],
        "assistant_memory": mtp["assistant"],
        "fallbacks": mtp["fallbacks"],
        "token_loop_allocations": mtp["token_loop_allocations"],
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
