"""Centralized versioned compiler profile contracts for Gemma 4 26B."""

from __future__ import annotations

from dataclasses import dataclass
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

PROFILES = {profile.name: profile for profile in (M04_PROFILE, M05_PROFILE)}

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
