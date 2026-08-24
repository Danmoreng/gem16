#!/usr/bin/env python3
"""Capture compact selected QAT-BF16 Gemma 4 26B reference boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--software-lock", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits-output", type=Path, required=True)
    parser.add_argument("--teacher-forced-logits-output", type=Path)
    parser.add_argument("--offload-folder", type=Path, required=True)
    parser.add_argument("--gpu-memory", default="12GiB")
    parser.add_argument("--cpu-memory", default="38GiB")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def selected_rows(tensor: Any, positions: list[int]) -> dict[str, object]:
    original_shape = [int(value) for value in tensor.shape]
    if tensor.ndim == 1:
        selected = [0]
        row_for_position = lambda _position: tensor
        available_positions = 1
    elif tensor.ndim >= 3 and tensor.shape[0] == 1:
        selected = positions
        row_for_position = lambda position: tensor[0, position].reshape(-1)
        available_positions = int(tensor.shape[1])
    else:
        flattened = tensor.reshape(-1, tensor.shape[-1])
        selected = positions
        row_for_position = lambda position: flattened[position]
        available_positions = int(flattened.shape[0])
    rows = []
    for position in selected:
        if position >= available_positions:
            raise RuntimeError(
                f"capture position {position} exceeds available positions {available_positions}"
            )
        row = row_for_position(position).detach().float().cpu().contiguous()
        rows.append(
            {
                "position": position,
                "sha256_f32le": tensor_sha256(row),
                "values_f32": row.tolist(),
            }
        )
    return {
        "source_shape": original_shape,
        "source_dtype": str(tensor.dtype),
        "rows": rows,
    }


def find_prompt(corpus: dict[str, object], prompt_id: str) -> dict[str, object]:
    records = corpus.get("records")
    if not isinstance(records, list):
        raise ValueError("corpus records must be an array")
    matches = [record for record in records if isinstance(record, dict) and record.get("id") == prompt_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one corpus record for {prompt_id!r}")
    return matches[0]


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1 or args.max_new_tokens > 16:
        raise ValueError("max-new-tokens must be in [1, 16]")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import accelerate
    import torch
    import torch.nn.functional as functional
    import transformers
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    prompt = find_prompt(corpus, args.prompt_id)
    token_ids = prompt.get("input_token_ids")
    if not isinstance(token_ids, list) or not token_ids or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids
    ):
        raise ValueError("prompt token IDs are invalid")
    positions = sorted({0, len(token_ids) - 1})
    selected_layers = prompt.get("selected_layers")
    if selected_layers != [0, 5, 6, 29]:
        raise ValueError("golden prompt must select layers 0, 5, 6 and 29")

    outer_config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    config = Gemma4TextConfig(**outer_config["text_config"])
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    key_mapping = {
        tensor["name"]: "model." + tensor["name"].removeprefix("model.language_model.")
        for tensor in inventory["tensors"]
        if tensor["name"].startswith("model.language_model.")
    }
    args.offload_folder.mkdir(parents=True, exist_ok=True)
    model = Gemma4ForCausalLM.from_pretrained(
        args.model,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        weights_only=True,
        key_mapping=key_mapping,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        offload_folder=args.offload_folder,
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()

    captures: dict[str, object] = {}
    hooks = []
    expert_wrappers = []

    def tensor_hook(label: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captures[label] = selected_rows(tensor, positions)
        return hook

    hooks.append(model.model.embed_tokens.register_forward_hook(tensor_hook("embedding")))
    hooks.append(model.model.norm.register_forward_hook(tensor_hook("final_norm")))

    for layer_index in selected_layers:
        layer = model.model.layers[layer_index]
        prefix = f"layer_{layer_index}"
        hooks.append(layer.register_forward_hook(tensor_hook(f"{prefix}.output")))
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_2",
        ):
            module = getattr(layer, name)
            hooks.append(module.register_forward_hook(tensor_hook(f"{prefix}.{name}")))
        hooks.append(layer.mlp.register_forward_hook(tensor_hook(f"{prefix}.shared_mlp")))
        attention = layer.self_attn
        for name in ("q_proj", "k_proj", "q_norm", "k_norm", "v_norm", "o_proj"):
            module = getattr(attention, name, None)
            if module is not None:
                hooks.append(module.register_forward_hook(tensor_hook(f"{prefix}.attention.{name}")))
        if getattr(attention, "v_proj", None) is not None:
            hooks.append(
                attention.v_proj.register_forward_hook(
                    tensor_hook(f"{prefix}.attention.v_proj")
                )
            )

        def context_pre_hook(_module, inputs, layer_prefix=prefix):
            captures[f"{layer_prefix}.attention.context_before_o"] = selected_rows(
                inputs[0], positions
            )

        hooks.append(attention.o_proj.register_forward_pre_hook(context_pre_hook))

        def router_hook(_module, _inputs, output, layer_prefix=prefix):
            probabilities, top_weights, top_indices = output
            captures[f"{layer_prefix}.router_probabilities"] = selected_rows(
                probabilities, positions
            )
            captures[f"{layer_prefix}.router_top_weights"] = selected_rows(
                top_weights, positions
            )
            flattened = top_indices.reshape(-1, top_indices.shape[-1])
            captures[f"{layer_prefix}.router_top_ids"] = {
                "source_shape": [int(value) for value in top_indices.shape],
                "rows": [
                    {
                        "position": position,
                        "values_i64": [int(value) for value in flattened[position].cpu()],
                    }
                    for position in positions
                ],
            }

        hooks.append(layer.router.register_forward_hook(router_hook))

        experts = layer.experts
        forward_attribute = "_old_forward" if hasattr(experts, "_old_forward") else "forward"
        original_experts_forward = getattr(experts, forward_attribute)

        def experts_forward_with_capture(
            hidden_states,
            top_indices,
            top_weights,
            module=experts,
            original_forward=original_experts_forward,
            layer_prefix=prefix,
        ):
            output = original_forward(hidden_states, top_indices, top_weights)
            flattened_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
            flattened_indices = top_indices.reshape(-1, top_indices.shape[-1])
            flattened_weights = top_weights.reshape(-1, top_weights.shape[-1])
            records = []
            with torch.inference_mode():
                for position in positions:
                    contributions = []
                    current = flattened_hidden[position : position + 1]
                    for rank in range(flattened_indices.shape[-1]):
                        expert_id = int(flattened_indices[position, rank])
                        gate, up = functional.linear(
                            current, module.gate_up_proj[expert_id]
                        ).chunk(2, dim=-1)
                        activated = module.act_fn(gate) * up
                        contribution = functional.linear(
                            activated, module.down_proj[expert_id]
                        ) * flattened_weights[position, rank]
                        row = contribution[0].detach().float().cpu().contiguous()
                        contributions.append(
                            {
                                "rank": rank,
                                "expert_id": expert_id,
                                "router_weight_f32": float(
                                    flattened_weights[position, rank].float().cpu()
                                ),
                                "sha256_f32le": tensor_sha256(row),
                                "values_f32": row.tolist(),
                            }
                        )
                    records.append({"position": position, "contributions": contributions})
            captures[f"{layer_prefix}.routed_expert_contributions"] = records
            return output

        setattr(experts, forward_attribute, experts_forward_with_capture)
        expert_wrappers.append((experts, forward_attribute, original_experts_forward))
        hooks.append(experts.register_forward_hook(tensor_hook(f"{prefix}.routed_experts_sum")))

    input_device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=input_device)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False, return_dict=True)
    final_logits = output.logits[0, -1].detach().float().cpu().contiguous()
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    final_array = final_logits.numpy().astype("<f4", copy=False)
    args.logits_output.write_bytes(final_array.tobytes())
    top_values, top_ids = torch.topk(final_logits, 20)
    del output

    for hook in hooks:
        hook.remove()
    for module, forward_attribute, original_forward in expert_wrappers:
        setattr(module, forward_attribute, original_forward)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=config.pad_token_id,
            eos_token_id=outer_config.get("eos_token_id", config.eos_token_id),
        )
    generated_ids = [int(value) for value in generated[0, len(token_ids) :].cpu()]

    teacher_forced_logits = None
    if args.teacher_forced_logits_output is not None:
        rows = [final_logits]
        with torch.inference_mode():
            for index in range(1, len(generated_ids)):
                prefix = torch.tensor(
                    [generated_ids[:index]], dtype=torch.long,
                    device=input_device,
                )
                forced = model(
                    input_ids=torch.cat((input_ids, prefix), dim=1),
                    use_cache=False,
                    return_dict=True,
                )
                rows.append(forced.logits[0, -1].detach().float().cpu().contiguous())
                del forced
        teacher_forced_array = (
            torch.stack(rows).numpy().astype("<f4", copy=False)
        )
        args.teacher_forced_logits_output.parent.mkdir(parents=True, exist_ok=True)
        args.teacher_forced_logits_output.write_bytes(
            teacher_forced_array.tobytes()
        )
        teacher_forced_logits = {
            "path": args.teacher_forced_logits_output.name,
            "dtype": "float32_le",
            "shape": [len(rows), int(rows[0].shape[0])],
            "bytes": args.teacher_forced_logits_output.stat().st_size,
            "sha256": file_sha256(args.teacher_forced_logits_output),
            "context_token_ids": generated_ids[:-1],
        }

    placements: dict[str, int] = {}
    for destination in model.hf_device_map.values():
        placements[str(destination)] = placements.get(str(destination), 0) + 1
    result = {
        "schema_version": 1,
        "status": "deterministic_reference_candidate",
        "checkpoint": {
            "repository": "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
            "revision": json.loads(args.model_lock.read_text(encoding="utf-8"))[
                "revision"
            ],
            "lock_sha256": file_sha256(args.model_lock),
            "inventory_sha256": file_sha256(args.inventory),
            "source_dtype": "BF16",
        },
        "software": {
            "lock_sha256": file_sha256(args.software_lock),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "accelerate": accelerate.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "attention_implementation": "eager",
            "trust_remote_code": False,
        },
        "execution": {
            "diagnostic_cpu_gpu_disk_dispatch": True,
            "device_map_counts": placements,
            "gpu_memory_limit": args.gpu_memory,
            "cpu_memory_limit": args.cpu_memory,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "offload_file_count": sum(1 for path in args.offload_folder.rglob("*") if path.is_file()),
            "performance_eligible": False,
        },
        "prompt": {
            "id": args.prompt_id,
            "corpus_sha256": file_sha256(args.corpus),
            "input_token_ids": token_ids,
            "input_token_ids_sha256_u32le": prompt["input_token_ids_sha256_u32le"],
            "selected_positions": positions,
            "selected_layers": selected_layers,
        },
        "captures": captures,
        "final_logits": {
            "path": args.logits_output.name,
            "dtype": "float32_le",
            "shape": [int(final_logits.shape[0])],
            "bytes": args.logits_output.stat().st_size,
            "sha256": file_sha256(args.logits_output),
            "top20_token_ids": [int(value) for value in top_ids],
            "top20_logits_f32": [float(value) for value in top_values],
        },
        "generated_token_ids": generated_ids,
    }
    if teacher_forced_logits is not None:
        result["teacher_forced_logits"] = teacher_forced_logits
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: captures={len(captures)} "
        f"logits_sha256={result['final_logits']['sha256']} generated={generated_ids}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
