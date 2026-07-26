#!/usr/bin/env python3
"""Capture Gemma 4 layer outputs and newly produced K/V states from vLLM.

This is an offline correctness reference tool, not a gem16 runtime
dependency. Run it with the pinned vLLM environment from docs/CORRECTNESS.md.
"""

from __future__ import annotations

import argparse
from array import array
import os
from pathlib import Path
import struct
import sys
from typing import Any


MAGIC = b"G16ST001"
HEADER = struct.Struct("<IIQQQII")
LAYER_HEADER = struct.Struct("<IIQ")
HIDDEN_ELEMENTS = 3840
LAYER_COUNT = 48
STAGE_NAMES = (
    "attention_output",
    "post_attention_norm",
    "post_attention_residual",
    "pre_feedforward_norm",
    "gate",
    "up",
    "gelu_product",
    "mlp_output",
    "post_feedforward_norm",
    "hidden",
)
INTERMEDIATE_ELEMENTS = 15360


def token_ids(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("token IDs must be integers") from error
    if not result or any(token < 0 or token >= 262144 for token in result):
        raise argparse.ArgumentTypeError("token IDs must be in [0, 262144)")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-token-ids", type=token_ids, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replay-layer0-down-m1",
        type=Path,
        help="optionally write a raw-float32 Layer-0 Down result replayed with M=1",
    )
    parser.add_argument(
        "--dump-layer0-kv-block0",
        type=Path,
        help=(
            "optionally write vLLM's physical Layer-0 KV block 0 as raw bytes; "
            "the tensor shape, strides, and dtype are printed alongside it"
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "bfloat16", "fp8"),
        default="auto",
    )
    return parser.parse_args()


def as_float_array(tensor: Any) -> array:
    # The intentionally explicit float32/CPU conversion makes the dump
    # independent of whether vLLM stores the state as BF16 or FP32.
    values = tensor.detach().float().cpu().contiguous().reshape(-1)
    result = array("f", values.tolist())
    if sys.byteorder != "little":
        result.byteswap()
    return result


def write_dump(
    path: Path,
    position: int,
    kv_cache_dtype: str,
    attention_contexts: dict[int, array],
    stages: dict[str, dict[int, array]],
    keys: dict[int, array],
    values: dict[int, array],
) -> None:
    for name in STAGE_NAMES:
        if set(stages[name]) != set(range(LAYER_COUNT)):
            missing = sorted(set(range(LAYER_COUNT)) - set(stages[name]))
            raise RuntimeError(f"did not capture stage {name}; missing={missing}")
    if set(attention_contexts) != set(range(LAYER_COUNT)):
        missing = sorted(set(range(LAYER_COUNT)) - set(attention_contexts))
        raise RuntimeError(
            f"did not capture attention contexts; missing={missing}"
        )
    if set(keys) != set(range(LAYER_COUNT)) or set(values) != set(range(LAYER_COUNT)):
        missing_keys = sorted(set(range(LAYER_COUNT)) - set(keys))
        missing_values = sorted(set(range(LAYER_COUNT)) - set(values))
        raise RuntimeError(
            "did not capture every decoder-layer K/V state; "
            f"missing_keys={missing_keys} missing_values={missing_values}"
        )
    total_elements = sum(
        len(attention_contexts[layer])
        + sum(len(stages[name][layer]) for name in STAGE_NAMES)
        + len(keys[layer])
        + len(values[layer])
        for layer in range(LAYER_COUNT)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(MAGIC)
        output.write(
            HEADER.pack(
                5,
                LAYER_COUNT,
                position,
                HIDDEN_ELEMENTS,
                total_elements,
                2,  # vLLM reference
                1 if kv_cache_dtype == "bfloat16" else 0,
            )
        )
        for layer in range(LAYER_COUNT):
            expected_kv = 512 if layer % 6 == 5 else 8 * 256
            expected_attention = 16 * (512 if layer % 6 == 5 else 256)
            if len(attention_contexts[layer]) != expected_attention:
                raise RuntimeError(
                    f"layer {layer}: unexpected attention-context width"
                )
            for name in STAGE_NAMES:
                expected_stage_elements = (
                    INTERMEDIATE_ELEMENTS
                    if name in ("gate", "up", "gelu_product")
                    else HIDDEN_ELEMENTS
                )
                if len(stages[name][layer]) != expected_stage_elements:
                    raise RuntimeError(
                        f"layer {layer}: unexpected width for {name}"
                    )
            if len(keys[layer]) != expected_kv or len(values[layer]) != expected_kv:
                raise RuntimeError(f"layer {layer}: unexpected K/V width")
            output.write(
                LAYER_HEADER.pack(layer, 1 if layer % 6 == 5 else 0, expected_kv)
            )
            output.write(attention_contexts[layer].tobytes())
            for name in STAGE_NAMES:
                output.write(stages[name][layer].tobytes())
            output.write(keys[layer].tobytes())
            output.write(values[layer].tobytes())


def main() -> int:
    args = parse_args()
    if args.max_tokens < 1 or args.position < 0:
        print("error: token limits and position are invalid", file=sys.stderr)
        return 2
    try:
        # Correctness hooks must execute in the process that owns the loaded
        # model. vLLM otherwise starts EngineCore in a child process and keeps
        # only an RPC client on LLMEngine.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        import torch
        from vllm import LLM, SamplingParams

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to vLLM")
        model_path = args.model.resolve(strict=True)
        llm = LLM(
            model=str(model_path),
            tokenizer=str(model_path),
            max_model_len=max(
                64, len(args.input_token_ids) + args.max_tokens
            ),
            gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_offload_gb=0,
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            kv_cache_dtype=args.kv_cache_dtype,
            seed=0,
            limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
        )
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        loaded_model = runner.model
        text_model = loaded_model.language_model.model
        layers = text_model.layers
        if len(layers) != LAYER_COUNT:
            raise RuntimeError(f"expected {LAYER_COUNT} layers, found {len(layers)}")

        current_positions: Any = None
        observed_positions: list[list[int]] = []
        stages: dict[str, dict[int, array]] = {
            name: {} for name in STAGE_NAMES
        }
        attention_contexts: dict[int, array] = {}
        keys: dict[int, array] = {}
        values: dict[int, array] = {}

        handles: list[Any] = []

        def target_row() -> int | None:
            if current_positions is None:
                return None
            matches = torch.nonzero(
                current_positions == args.position, as_tuple=False
            ).reshape(-1)
            return int(matches[0].item()) if matches.numel() else None

        for layer_index, layer in enumerate(layers):
            def layer_pre_hook(
                _module: Any,
                hook_args: tuple[Any, ...],
                hook_kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                nonlocal current_positions
                current_positions = (
                    hook_args[0] if hook_args else hook_kwargs["positions"]
                )
                if index == 0:
                    observed_positions.append(
                        [
                            int(value)
                            for value in current_positions.detach().cpu().tolist()
                        ]
                    )

            def attention_hook(
                _module: Any,
                hook_args: tuple[Any, ...],
                hook_kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                row = target_row()
                if row is None:
                    return
                query = hook_args[0] if hook_args else hook_kwargs["query"]
                key = hook_args[1] if len(hook_args) > 1 else hook_kwargs["key"]
                value = hook_args[2] if len(hook_args) > 2 else hook_kwargs["value"]
                if row >= query.shape[0] or row >= key.shape[0] or row >= value.shape[0]:
                    raise RuntimeError(f"layer {index}: target row exceeds Q/K/V batch")
                keys[index] = as_float_array(key[row])
                values[index] = as_float_array(value[row])

            def layer_hook(
                _module: Any,
                _hook_args: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                row = target_row()
                if row is None:
                    return
                hidden = output[0] if isinstance(output, tuple) else output
                if row >= hidden.shape[0]:
                    raise RuntimeError(f"layer {index}: target row exceeds hidden batch")
                stages["hidden"][index] = as_float_array(hidden[row])

            def stage_hook(
                name: str, index: int
            ) -> Any:
                def hook(
                    _module: Any, _hook_args: tuple[Any, ...], output: Any
                ) -> None:
                    row = target_row()
                    if row is None:
                        return
                    tensor = output[0] if isinstance(output, tuple) else output
                    if row >= tensor.shape[0]:
                        raise RuntimeError(
                            f"layer {index}: target row exceeds {name} batch"
                        )
                    stages[name][index] = as_float_array(tensor[row])

                return hook

            def gate_up_hook(
                _module: Any,
                _hook_args: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                row = target_row()
                if row is None:
                    return
                tensor = output[0] if isinstance(output, tuple) else output
                if row >= tensor.shape[0] or tensor.shape[-1] != 2 * INTERMEDIATE_ELEMENTS:
                    raise RuntimeError(
                        f"layer {index}: unexpected fused Gate/Up output"
                    )
                gate, up = tensor[row].split(INTERMEDIATE_ELEMENTS)
                stages["gate"][index] = as_float_array(gate)
                stages["up"][index] = as_float_array(up)

            def residual_pre_hook(
                _module: Any,
                hook_args: tuple[Any, ...],
                hook_kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                row = target_row()
                if row is None:
                    return
                tensor = hook_args[0] if hook_args else hook_kwargs["input"]
                if row >= tensor.shape[0]:
                    raise RuntimeError(
                        f"layer {index}: target row exceeds residual batch"
                    )
                stages["post_attention_residual"][index] = as_float_array(
                    tensor[row]
                )

            def output_projection_pre_hook(
                _module: Any,
                hook_args: tuple[Any, ...],
                hook_kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                row = target_row()
                if row is None:
                    return
                tensor = hook_args[0] if hook_args else hook_kwargs["input_"]
                if row >= tensor.shape[0]:
                    raise RuntimeError(
                        f"layer {index}: target row exceeds attention context batch"
                    )
                attention_contexts[index] = as_float_array(tensor[row])

            def down_projection_pre_hook(
                _module: Any,
                hook_args: tuple[Any, ...],
                hook_kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                row = target_row()
                if row is None:
                    return
                tensor = hook_args[0] if hook_args else hook_kwargs["input_"]
                if row >= tensor.shape[0] or tensor.shape[-1] != INTERMEDIATE_ELEMENTS:
                    raise RuntimeError(
                        f"layer {index}: unexpected Down-projection input"
                    )
                stages["gelu_product"][index] = as_float_array(tensor[row])

            handles.append(
                layer.register_forward_pre_hook(
                    layer_pre_hook, with_kwargs=True
                )
            )
            handles.append(
                layer.self_attn.attn.register_forward_pre_hook(
                    attention_hook, with_kwargs=True
                )
            )
            handles.append(
                layer.self_attn.register_forward_hook(
                    stage_hook("attention_output", layer_index)
                )
            )
            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    output_projection_pre_hook, with_kwargs=True
                )
            )
            handles.append(
                layer.post_attention_layernorm.register_forward_hook(
                    stage_hook("post_attention_norm", layer_index)
                )
            )
            handles.append(
                layer.pre_feedforward_layernorm.register_forward_pre_hook(
                    residual_pre_hook, with_kwargs=True
                )
            )
            handles.append(
                layer.pre_feedforward_layernorm.register_forward_hook(
                    stage_hook("pre_feedforward_norm", layer_index)
                )
            )
            handles.append(
                layer.mlp.register_forward_hook(
                    stage_hook("mlp_output", layer_index)
                )
            )
            handles.append(
                layer.mlp.gate_up_proj.register_forward_hook(gate_up_hook)
            )
            handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    down_projection_pre_hook, with_kwargs=True
                )
            )
            handles.append(
                layer.post_feedforward_layernorm.register_forward_hook(
                    stage_hook("post_feedforward_norm", layer_index)
                )
            )
            handles.append(layer.register_forward_hook(layer_hook))

        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=0,
        )
        request = llm.generate(
            [{"prompt_token_ids": args.input_token_ids}],
            sampling,
            use_tqdm=False,
        )[0]
        output_ids = list(request.outputs[0].token_ids)
        print(f"observed_positions={observed_positions}")
        print(
            f"captured_hidden={sorted(stages['hidden'])} "
            f"captured_keys={sorted(keys)} captured_values={sorted(values)}"
        )
        for handle in handles:
            handle.remove()
        if args.dump_layer0_kv_block0 is not None:
            if not runner.kv_caches:
                raise RuntimeError("vLLM runner exposes no KV-cache tensors")
            layer0_cache = runner.kv_caches[0]
            if layer0_cache.ndim != 5 or layer0_cache.shape[1] != 2:
                raise RuntimeError(
                    "unexpected Layer-0 KV-cache layout: "
                    f"shape={tuple(layer0_cache.shape)}"
                )
            block0 = layer0_cache[0].contiguous()
            raw_block0 = block0.view(torch.uint8).cpu().numpy().tobytes()
            args.dump_layer0_kv_block0.parent.mkdir(
                parents=True, exist_ok=True
            )
            args.dump_layer0_kv_block0.write_bytes(raw_block0)
            print(
                "layer0_kv_cache="
                f"shape={tuple(layer0_cache.shape)} "
                f"stride={tuple(layer0_cache.stride())} "
                f"dtype={layer0_cache.dtype} "
                f"block0_bytes={len(raw_block0)}"
            )
            print(f"wrote {args.dump_layer0_kv_block0}")
        if args.replay_layer0_down_m1 is not None:
            replay_input = torch.tensor(
                stages["gelu_product"][0],
                dtype=torch.bfloat16,
                device="cuda",
            ).reshape(1, INTERMEDIATE_ELEMENTS)
            replay_output = layers[0].mlp.down_proj(replay_input)
            if isinstance(replay_output, tuple):
                replay_output = replay_output[0]
            replay_values = as_float_array(replay_output[0])
            args.replay_layer0_down_m1.parent.mkdir(parents=True, exist_ok=True)
            args.replay_layer0_down_m1.write_bytes(replay_values.tobytes())
            print(f"wrote {args.replay_layer0_down_m1}")
        write_dump(
            args.output,
            args.position,
            args.kv_cache_dtype,
            attention_contexts,
            stages,
            keys,
            values,
        )
        print(f"output_token_ids={output_ids}")
        print(f"wrote {args.output}")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return 0
    except (OSError, RuntimeError, ValueError, AttributeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
