"""Centralized versioned compiler profile contracts for Gemma 4 26B."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import re
from typing import NamedTuple


@dataclass(frozen=True)
class CompilerProfile:
    name: str
    head_format: str
    milestone: str
    artifact_status: str
    compiler_implementation: str
    header_label: str
    attention: str
    experts: str
    embedding_head: str
    production_quantization_implemented: bool
    allowed_encoders: frozenset[str]
    deferred_reason: str | None = None

    @property
    def quantization(self) -> dict[str, object]:
        return {
            "profile": self.name,
            "attention": self.attention,
            "experts": self.experts,
            "embedding_head": self.embedding_head,
            "production_quantization_implemented": self.production_quantization_implemented,
        }


M04_PROFILE = CompilerProfile(
    name="synthetic-copy-v1",
    head_format="source",
    milestone="M04",
    artifact_status="m04_scaffold_not_runtime_loadable",
    compiler_implementation="gem16_compile_m04_v1",
    header_label="m04-synthetic-copy-scaffold",
    attention="copy-v1-scaffold",
    experts="copy-v1-scaffold",
    embedding_head="source-copy-v1",
    production_quantization_implemented=False,
    allowed_encoders=frozenset({"copy-v1"}),
)

M05_SOURCE_CONTRACT = "gemma4-26b-source-bf16-attention-v1"
M05_VISION_EXCLUSION_REASON = "text-only Gemma 4 26B profile excludes vision tensors"
M05_DEFERRED_REASON = "deferred to M06-M08; absent from M05 attention-only partial artifact"
M05_SOURCE_LOCK_SHA256 = {
    "ordinary_bf16": "1932b06d568ce8c72f3d2013d39ffd15dd61d37732638c2013f2220eb177a620",
    "qat_bf16": "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230",
}
M05_APPROVED_SOURCE_LOCKS = frozenset(M05_SOURCE_LOCK_SHA256.values())

M05_PROFILE = CompilerProfile(
    name="fp8-attention-partial-v1",
    head_format="deferred",
    milestone="M05",
    artifact_status="m05_fp8_attention_partial_not_runtime_loadable",
    compiler_implementation="gem16_compile_m05_native_v1",
    header_label="m05-fp8-attention-partial",
    attention="fp8-per-output-row-v1",
    experts="deferred-to-m06",
    embedding_head="deferred-to-m07",
    production_quantization_implemented=False,
    allowed_encoders=frozenset({"fp8-rowwise-weight-v1", "fp8-rowwise-scale-v1"}),
    deferred_reason=M05_DEFERRED_REASON,
)

M06_SOURCE_CONTRACT = "gemma4-26b-source-bf16-experts-v1"
M06_DEFERRED_REASON = "deferred to M07-M08; absent from M06 expert-only partial artifact"
M06_VISION_EXCLUSION_REASON = "text-only Gemma 4 26B profile excludes vision tensors"
M07_SOURCE_CONTRACT = "gemma4-26b-source-bf16-tied-head-v1"
M07_DEFERRED_REASON = "deferred to M08; absent from M07 tied-head partial artifact"
M07_PROFILE = CompilerProfile(
    name="nvfp4-tied-head-partial-v1",
    head_format="nvfp4",
    milestone="M07",
    artifact_status="m07_nvfp4_tied_head_partial_not_runtime_loadable",
    compiler_implementation="gem16_compile_m07_native_v1",
    header_label="m07-nvfp4-tied-head-partial",
    attention="deferred-to-m08",
    experts="deferred-to-m08",
    embedding_head="nvfp4-group16-divisor-v1",
    production_quantization_implemented=False,
    allowed_encoders=frozenset({
        "nvfp4-packed-v1", "nvfp4-local-scale-v1",
        "nvfp4-weight-divisor-v1", "nvfp4-input-divisor-v1",
    }),
    deferred_reason=M07_DEFERRED_REASON,
)

M08_SOURCE_CONTRACT = "gemma4-26b-source-bf16-complete-hybrid-v1"
M08_PROFILE = CompilerProfile(
    name="sm120-text-hybrid-v1",
    head_format="nvfp4",
    milestone="M08",
    artifact_status="m08_complete_runtime_loadable_experimental",
    compiler_implementation="gem16_compile_m08_hybrid_v1",
    header_label="m08-sm120-text-hybrid-v1",
    attention="fp8-per-output-row-v1",
    experts="nvfp4-group16-divisor-v1",
    embedding_head="nvfp4-group16-divisor-v1",
    production_quantization_implemented=True,
    allowed_encoders=frozenset({
        "copy-v1",
        "constant-bf16-one-v1",
        "fp8-rowwise-weight-v1",
        "fp8-rowwise-scale-v1",
        "nvfp4-packed-v1",
        "nvfp4-local-scale-v1",
        "nvfp4-weight-divisor-v1",
        "nvfp4-input-divisor-v1",
    }),
)

M06_PROFILE = CompilerProfile(
    name="nvfp4-experts-partial-v1",
    head_format="deferred",
    milestone="M06",
    artifact_status="m06_nvfp4_experts_partial_not_runtime_loadable",
    compiler_implementation="gem16_compile_m06_native_v1",
    header_label="m06-nvfp4-experts-partial",
    attention="deferred-to-m08",
    experts="nvfp4-group16-divisor-v1",
    embedding_head="deferred-to-m07",
    production_quantization_implemented=False,
    allowed_encoders=frozenset({
        "nvfp4-packed-v1",
        "nvfp4-local-scale-v1",
        "nvfp4-weight-divisor-v1",
        "nvfp4-input-divisor-v1",
    }),
    deferred_reason=M06_DEFERRED_REASON,
)

M06_QUANTIZER_PARAMETERS = {
    "contract_id": "gem16.nvfp4_bf16_group16",
    "contract_version": 1,
    "source_dtype": "BF16",
    "packed_dtype": "U8",
    "local_scale_dtype": "F8_E4M3",
    "global_scale_dtype": "F32",
    "group_size": 16,
    "value_codec": "E2M1",
    "scale_codec": "E4M3FN",
    "global_scale_role": "divisor",
    "weight_divisor": "bf16_rne((448*6)/tensor_amax)",
    "zero_tensor_divisor": 1.0,
    "input_divisor": 1.0,
    # The binary32 expression is encoded directly; there is no BF16
    # intermediate between block_amax and the E4M3FN encoder.
    "local_scale": "e4m3fn(binary32((block_amax*(1/6))*weight_divisor))",
    "rounding": "nearest_even",
    "saturation": "finite_saturation",
    "signed_zero": "preserve_in_nonzero_blocks_canonicalize_zero_blocks",
    "zero_behavior": "zero_or_underflow_block_emits_zero_scale_and_zero_payload",
    "packing": "low_nibble_first",
    "tensor_granularity": "one_weight_and_input_divisor_per_source_tensor",
}
M06_DEQUANTIZATION_EQUATION = (
    "W_approx = decode_e2m1(weight_packed) * "
    "decode_e4m3fn(weight_scale) / weight_global_divisor"
)

M06_COMPONENT_LAYOUTS = {
    "weight_packed": {
        "encoder": "nvfp4-packed-v1",
        "transformation": "nvfp4-packed",
        "output_dtype": "U8",
        "disk_layout": "canonical_row_major_low_nibble_first",
        "runtime_layout_shared": "sm120_row8_k64",
        "runtime_layout_routed": "expert_major_sm120_row8_k64",
    },
    "weight_scale": {
        "encoder": "nvfp4-local-scale-v1",
        "transformation": "nvfp4-local-scale",
        "output_dtype": "F8_E4M3",
        "disk_layout": "canonical_row_major_group16_e4m3",
        "runtime_layout_shared": "sm120_row8_group16_e4m3",
        "runtime_layout_routed": "expert_major_sm120_row8_group16_e4m3",
    },
    "weight_global_scale": {
        "encoder": "nvfp4-weight-divisor-v1",
        "transformation": "nvfp4-weight-divisor",
        "output_dtype": "F32",
        "disk_layout": "scalar_f32",
        "runtime_layout_shared": "scalar_f32",
        "runtime_layout_routed": "scalar_f32",
    },
    "input_global_scale": {
        "encoder": "nvfp4-input-divisor-v1",
        "transformation": "nvfp4-input-divisor",
        "output_dtype": "F32",
        "disk_layout": "scalar_f32",
        "runtime_layout_shared": "scalar_f32",
        "runtime_layout_routed": "scalar_f32",
    },
}


def m06_component_parameters(component: str) -> dict[str, object]:
    """Return the exact parameter object for one M06 output component."""
    if component not in M06_COMPONENT_LAYOUTS:
        raise ValueError(f"unsupported M06 component: {component}")
    parameters = dict(M06_QUANTIZER_PARAMETERS)
    parameters["component"] = component
    return parameters

# Keep M07's versioned contract independent from the accepted M06 tables. A
# future head-specific revision must not mutate the expert profile by aliasing
# nested dictionaries.
M07_QUANTIZER_PARAMETERS = copy.deepcopy(M06_QUANTIZER_PARAMETERS)
M07_COMPONENT_LAYOUTS = {
    component: copy.deepcopy(layout)
    for component, layout in M06_COMPONENT_LAYOUTS.items()
}
M07_DEQUANTIZATION_EQUATION = M06_DEQUANTIZATION_EQUATION

def m07_component_parameters(component: str) -> dict[str, object]:
    if component not in M07_COMPONENT_LAYOUTS:
        raise ValueError(f"unsupported M07 component: {component}")
    parameters = copy.deepcopy(M07_QUANTIZER_PARAMETERS)
    parameters["component"] = component
    return parameters


class M05AttentionSpec(NamedTuple):
    shape: tuple[int, int]
    role: str


M05_LAYER_COUNT = 30
M05_GLOBAL_LAYERS = frozenset({5, 11, 17, 23, 29})
M05_HIDDEN = 2816
M05_LOCAL_Q = 4096
M05_LOCAL_KV = 2048
M05_GLOBAL_Q = 8192
M05_GLOBAL_KV = 1024


def _make_m05_attention_table() -> dict[str, M05AttentionSpec]:
    table: dict[str, M05AttentionSpec] = {}
    for layer in range(M05_LAYER_COUNT):
        global_layer = layer in M05_GLOBAL_LAYERS
        q = M05_GLOBAL_Q if global_layer else M05_LOCAL_Q
        kv = M05_GLOBAL_KV if global_layer else M05_LOCAL_KV
        prefix = f"model.language_model.layers.{layer}.self_attn"
        for projection, shape in (
            ("q", (q, M05_HIDDEN)),
            ("k", (kv, M05_HIDDEN)),
            ("o", (M05_HIDDEN, q)),
        ):
            table[f"{prefix}.{projection}_proj.weight"] = M05AttentionSpec(
                shape, f"attention_{projection}_projection"
            )
        if not global_layer:
            table[f"{prefix}.v_proj.weight"] = M05AttentionSpec(
                (kv, M05_HIDDEN), "attention_v_projection"
            )
    return dict(sorted(table.items()))


M05_ATTENTION_TABLE = _make_m05_attention_table()
_M05_LAYER_NAME_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_M05_VISION_LAYER_RE = re.compile(
    r"^model\.vision_tower\.encoder\.layers\.(\d+)\.(.+)$"
)
_M05_VISION_LAYER_ROLES = {
    "input_layernorm.weight": "vision_norm",
    "mlp.down_proj.linear.weight": "vision_mlp",
    "mlp.gate_proj.linear.weight": "vision_mlp",
    "mlp.up_proj.linear.weight": "vision_mlp",
    "post_attention_layernorm.weight": "vision_norm",
    "post_feedforward_layernorm.weight": "vision_norm",
    "pre_feedforward_layernorm.weight": "vision_norm",
    "self_attn.k_norm.weight": "vision_norm",
    "self_attn.k_proj.linear.weight": "vision_attention",
    "self_attn.o_proj.linear.weight": "vision_attention",
    "self_attn.q_norm.weight": "vision_norm",
    "self_attn.q_proj.linear.weight": "vision_attention",
    "self_attn.v_proj.linear.weight": "vision_attention",
}
_M05_SOURCE_LAYER_ROLES = {
    "input_layernorm.weight": "input_layer_norm",
    "layer_scalar": "layer_scalar",
    "mlp.down_proj.weight": "shared_mlp_down",
    "mlp.gate_proj.weight": "shared_mlp_gate",
    "mlp.up_proj.weight": "shared_mlp_up",
    "post_attention_layernorm.weight": "post_attention_layer_norm",
    "post_feedforward_layernorm.weight": "post_feed_forward_layer_norm",
    "post_feedforward_layernorm_1.weight": "post_feed_forward_layer_norm_1",
    "post_feedforward_layernorm_2.weight": "post_feed_forward_layer_norm_2",
    "pre_feedforward_layernorm.weight": "pre_feed_forward_layer_norm",
    "pre_feedforward_layernorm_2.weight": "pre_feed_forward_layer_norm_2",
    "router.per_expert_scale": "router_per_expert_scale",
    "router.proj.weight": "router_projection",
    "router.scale": "router_norm_scale",
    "self_attn.k_norm.weight": "attention_k_norm",
    "self_attn.k_proj.weight": "attention_k_projection",
    "self_attn.o_proj.weight": "attention_o_projection",
    "self_attn.q_norm.weight": "attention_q_norm",
    "self_attn.q_proj.weight": "attention_q_projection",
    "self_attn.v_proj.weight": "attention_v_projection",
    "experts.down_proj": "routed_expert_down",
    "experts.gate_up_proj": "routed_expert_gate_up",
}


def classify_m05_source(name: str) -> str:
    """Return the frozen M03 role for a source tensor, or reject unknown input."""
    exact_vision = {
        "model.embed_vision.embedding_projection.weight": "vision_projection",
        "model.vision_tower.patch_embedder.position_embedding_table": "vision_embedding",
        "model.vision_tower.patch_embedder.input_proj.weight": "vision_projection",
        "model.vision_tower.std_bias": "vision_norm",
        "model.vision_tower.std_scale": "vision_norm",
    }
    if name in exact_vision:
        return exact_vision[name]
    vision_match = _M05_VISION_LAYER_RE.fullmatch(name)
    if vision_match is not None:
        layer = int(vision_match.group(1))
        if layer >= 27:
            raise ValueError(f"unknown M05 vision layer: {name}")
        role = _M05_VISION_LAYER_ROLES.get(vision_match.group(2))
        if role is None:
            raise ValueError(f"unknown M05 vision tensor: {name}")
        return role
    if name.startswith("model.vision_tower."):
        raise ValueError(f"unknown M05 vision tensor: {name}")
    if name == "model.language_model.embed_tokens.weight":
        return "tied_embedding_and_output"
    if name == "model.language_model.norm.weight":
        return "final_norm"
    match = _M05_LAYER_NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"unknown M05 source tensor: {name}")
    layer = int(match.group(1))
    if layer < 0 or layer >= M05_LAYER_COUNT:
        raise ValueError(f"M05 source layer is outside 0..29: {name}")
    role = _M05_SOURCE_LAYER_ROLES.get(match.group(2))
    if role is None:
        raise ValueError(f"unknown M05 source tensor: {name}")
    return role


def m06_expected_source_specs() -> dict[str, tuple[str, tuple[int, ...]]]:
    """Return the complete frozen 1,013-name QAT source inventory contract."""
    specs: dict[str, tuple[str, tuple[int, ...]]] = {
        "model.language_model.embed_tokens.weight":
            ("tied_embedding_and_output", (262144, 2816)),
        "model.language_model.norm.weight": ("final_norm", (2816,)),
        "model.embed_vision.embedding_projection.weight":
            ("vision_projection", (2816, 1152)),
        "model.vision_tower.patch_embedder.position_embedding_table":
            ("vision_embedding", (2, 10240, 1152)),
        "model.vision_tower.patch_embedder.input_proj.weight":
            ("vision_projection", (1152, 768)),
        "model.vision_tower.std_bias": ("vision_norm", (1152,)),
        "model.vision_tower.std_scale": ("vision_norm", (1152,)),
    }
    for layer in range(M05_LAYER_COUNT):
        prefix = f"model.language_model.layers.{layer}."
        global_layer = layer in M05_GLOBAL_LAYERS
        q = M05_GLOBAL_Q if global_layer else M05_LOCAL_Q
        kv = M05_GLOBAL_KV if global_layer else M05_LOCAL_KV
        shapes = {
            "input_layernorm.weight": (2816,),
            "layer_scalar": (1,),
            "mlp.down_proj.weight": (2816, 2112),
            "mlp.gate_proj.weight": (2112, 2816),
            "mlp.up_proj.weight": (2112, 2816),
            "post_attention_layernorm.weight": (2816,),
            "post_feedforward_layernorm.weight": (2816,),
            "post_feedforward_layernorm_1.weight": (2816,),
            "post_feedforward_layernorm_2.weight": (2816,),
            "pre_feedforward_layernorm.weight": (2816,),
            "pre_feedforward_layernorm_2.weight": (2816,),
            "router.per_expert_scale": (128,),
            "router.proj.weight": (128, 2816),
            "router.scale": (2816,),
            "self_attn.k_norm.weight": (512 if global_layer else 256,),
            "self_attn.k_proj.weight": (kv, 2816),
            "self_attn.o_proj.weight": (2816, q),
            "self_attn.q_norm.weight": (512 if global_layer else 256,),
            "self_attn.q_proj.weight": (q, 2816),
            "self_attn.v_proj.weight": (kv, 2816),
            "experts.down_proj": (128, 2816, 704),
            "experts.gate_up_proj": (128, 1408, 2816),
        }
        for suffix, shape in shapes.items():
            # Global layers intentionally have no V projection in the locked
            # inventory; this mirrors M05_ATTENTION_TABLE exactly.
            if global_layer and suffix == "self_attn.v_proj.weight":
                continue
            specs[prefix + suffix] = (classify_m05_source(prefix + suffix), shape)
    for layer in range(27):
        prefix = f"model.vision_tower.encoder.layers.{layer}."
        for suffix in _M05_VISION_LAYER_ROLES:
            shape = (1152,)
            if suffix in {"self_attn.k_norm.weight", "self_attn.q_norm.weight"}:
                shape = (72,)
            elif suffix in {
                "self_attn.k_proj.linear.weight", "self_attn.o_proj.linear.weight",
                "self_attn.q_proj.linear.weight", "self_attn.v_proj.linear.weight",
            }:
                shape = (1152, 1152)
            elif suffix in {"mlp.gate_proj.linear.weight", "mlp.up_proj.linear.weight"}:
                shape = (4304, 1152)
            elif suffix == "mlp.down_proj.linear.weight":
                shape = (1152, 4304)
            specs[prefix + suffix] = (classify_m05_source(prefix + suffix), shape)
    return dict(sorted(specs.items()))


PROFILES = {
    profile.name: profile
    for profile in (M04_PROFILE, M05_PROFILE, M06_PROFILE, M07_PROFILE, M08_PROFILE)
}

M05_QUANTIZER_PARAMETERS = {
    "contract_id": "gem16.fp8_attention_rowwise",
    "contract_version": 1,
    "source_dtype": "BF16",
    "output_dtype": "F8_E4M3",
    "codec": "E4M3FN",
    "scale_dtype": "BF16",
    "scale_axis": "output_row",
    "finite_max": 448.0,
    "rounding": "nearest_even",
    "signed_zero": "preserve",
    "zero_row_scale": "1.0",
    "underflow": "minimum_positive_bf16_subnormal",
}
M05_DEQUANTIZATION_EQUATION = (
    "source_approx[n,k] = decode_e4m3fn(weight[n,k]) * "
    "decode_bf16(weight_scale[n,0])"
)


def profile_for(name: str, head_format: str) -> CompilerProfile:
    profile = PROFILES.get(name)
    if profile is None:
        raise ValueError(f"unsupported compiler profile: {name}")
    if profile.head_format != head_format:
        raise ValueError(
            f"profile {name!r} requires head format {profile.head_format!r}, "
            f"got {head_format!r}"
        )
    return profile


__all__ = [
    "CompilerProfile",
    "M04_PROFILE",
    "M05_PROFILE",
    "M06_DEFERRED_REASON",
    "M06_DEQUANTIZATION_EQUATION",
    "M07_DEFERRED_REASON",
    "M07_PROFILE",
    "M07_SOURCE_CONTRACT",
    "M08_PROFILE",
    "M08_SOURCE_CONTRACT",
    "M06_PROFILE",
    "M06_QUANTIZER_PARAMETERS",
    "M06_COMPONENT_LAYOUTS",
    "m06_component_parameters",
    "M07_QUANTIZER_PARAMETERS",
    "M07_COMPONENT_LAYOUTS",
    "M07_DEQUANTIZATION_EQUATION",
    "m07_component_parameters",
    "m06_expected_source_specs",
    "M06_SOURCE_CONTRACT",
    "M05_APPROVED_SOURCE_LOCKS",
    "M05_ATTENTION_TABLE",
    "M05_DEFERRED_REASON",
    "M05_DEQUANTIZATION_EQUATION",
    "M05_GLOBAL_LAYERS",
    "M05_HIDDEN",
    "M05_LAYER_COUNT",
    "M05_LOCAL_Q",
    "M05_LOCAL_KV",
    "M05_GLOBAL_Q",
    "M05_GLOBAL_KV",
    "M05_QUANTIZER_PARAMETERS",
    "M05_SOURCE_CONTRACT",
    "M05_SOURCE_LOCK_SHA256",
    "M05_VISION_EXCLUSION_REASON",
    "M05AttentionSpec",
    "classify_m05_source",
    "PROFILES",
    "profile_for",
]
