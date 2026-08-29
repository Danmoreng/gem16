"""Checked layout and rate-map planning for GEM16-Trellis35 routed experts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

from .common import UINT64_MAX, InvalidPlanError


TRELLIS35_LAYER_COUNT = 30
TRELLIS35_EXPERT_COUNT = 128
TRELLIS35_K3_COUNT = 64
TRELLIS35_K4_COUNT = 64
TRELLIS35_TILE = 16
TRELLIS35_HADAMARD_BLOCK = 128
TRELLIS35_ALIGNMENT = 256
TRELLIS35_DESCRIPTOR_BYTES = 8
TRELLIS35_SIDECAR_ELEMENT_BYTES = 2
UINT32_MAX = (1 << 32) - 1

GATE_UP_INPUT = 2816
GATE_UP_OUTPUT = 1408
GATE_UP_BOUNDARY = 704
DOWN_LOGICAL_INPUT = 704
DOWN_PHYSICAL_INPUT = 768
DOWN_OUTPUT = 2816


def _checked_nonnegative(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidPlanError(f"{description} must be a non-negative integer")
    if value > UINT64_MAX:
        raise InvalidPlanError(f"{description} exceeds uint64")
    return value


def checked_add(left: int, right: int, description: str) -> int:
    left = _checked_nonnegative(left, description)
    right = _checked_nonnegative(right, description)
    if left > UINT64_MAX - right:
        raise InvalidPlanError(f"{description} exceeds uint64")
    return left + right


def checked_mul(*values: int, description: str) -> int:
    result = 1
    for value in values:
        value = _checked_nonnegative(value, description)
        if value != 0 and result > UINT64_MAX // value:
            raise InvalidPlanError(f"{description} exceeds uint64")
        result *= value
    return result


def align_up(value: int, alignment: int, description: str) -> int:
    value = _checked_nonnegative(value, description)
    alignment = _checked_nonnegative(alignment, f"{description} alignment")
    if alignment == 0 or alignment & (alignment - 1):
        raise InvalidPlanError(f"{description} alignment must be a power of two")
    padding = (-value) & (alignment - 1)
    return checked_add(value, padding, description)


def validate_rate_map(rate_map: Sequence[int]) -> tuple[int, ...]:
    """Validate one family/layer rate map in stable expert-index order."""
    if (
        not isinstance(rate_map, Sequence)
        or isinstance(rate_map, (str, bytes))
        or len(rate_map) != TRELLIS35_EXPERT_COUNT
    ):
        raise InvalidPlanError(
            f"Trellis35 rate map must contain {TRELLIS35_EXPERT_COUNT} entries"
        )
    result: list[int] = []
    for expert, rate in enumerate(rate_map):
        if isinstance(rate, bool) or not isinstance(rate, int) or rate not in (3, 4):
            raise InvalidPlanError(
                f"Trellis35 expert {expert} rate must be integer K3 or K4"
            )
        result.append(rate)
    if result.count(3) != TRELLIS35_K3_COUNT or result.count(4) != TRELLIS35_K4_COUNT:
        raise InvalidPlanError("Trellis35 rate map must contain exactly 64 K3 and 64 K4 experts")
    return tuple(result)


def select_rate_map(proxy_benefits: Sequence[float]) -> tuple[int, ...]:
    """Select the 64 highest positive K4 benefits, ties by expert index."""
    if (
        not isinstance(proxy_benefits, Sequence)
        or isinstance(proxy_benefits, (str, bytes))
        or len(proxy_benefits) != TRELLIS35_EXPERT_COUNT
    ):
        raise InvalidPlanError(
            f"Trellis35 proxy benefits must contain {TRELLIS35_EXPERT_COUNT} entries"
        )
    ranked: list[tuple[float, int]] = []
    for expert, raw_benefit in enumerate(proxy_benefits):
        if isinstance(raw_benefit, bool) or not isinstance(raw_benefit, (int, float)):
            raise InvalidPlanError(f"Trellis35 expert {expert} benefit must be numeric")
        benefit = float(raw_benefit)
        if not math.isfinite(benefit):
            raise InvalidPlanError(f"Trellis35 expert {expert} benefit must be finite")
        ranked.append((benefit, expert))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked[:TRELLIS35_K4_COUNT]
    if selected[-1][0] <= 0.0:
        raise InvalidPlanError("Trellis35 requires at least 64 positive K4 proxy benefits")
    k4_experts = {expert for _benefit, expert in selected}
    return validate_rate_map(tuple(4 if expert in k4_experts else 3 for expert in range(128)))


def _payload_bytes(coefficients_per_expert: int, rate_map: Sequence[int], description: str) -> int:
    rates = validate_rate_map(rate_map)
    payload_bits = 0
    for rate in rates:
        payload_bits = checked_add(
            payload_bits,
            checked_mul(coefficients_per_expert, rate, description=f"{description} bits"),
            f"{description} bits",
        )
    if payload_bits % 8:
        raise InvalidPlanError(f"{description} payload is not byte aligned")
    return payload_bits // 8


def _pool_bytes(coefficients_per_expert: int, rate: int, count: int, description: str) -> int:
    bits = checked_mul(coefficients_per_expert, rate, count, description=f"{description} bits")
    if bits % 8:
        raise InvalidPlanError(f"{description} is not byte aligned")
    result = bits // 8
    if result > UINT32_MAX:
        raise InvalidPlanError(f"{description} cannot use U32-local descriptor offsets")
    return result


@dataclass(frozen=True)
class Trellis35Projection:
    name: str
    logical_input: int
    physical_input: int
    output: int
    suh_elements_per_expert: int
    svh_elements_per_expert: int

    @property
    def logical_coefficients_per_expert(self) -> int:
        return checked_mul(
            self.logical_input, self.output,
            description=f"{self.name} logical coefficients",
        )

    @property
    def encoded_coefficients_per_expert(self) -> int:
        return checked_mul(
            self.physical_input, self.output,
            description=f"{self.name} encoded coefficients",
        )


GATE_UP = Trellis35Projection(
    name="gate_up",
    logical_input=GATE_UP_INPUT,
    physical_input=GATE_UP_INPUT,
    output=GATE_UP_OUTPUT,
    suh_elements_per_expert=GATE_UP_INPUT,
    svh_elements_per_expert=GATE_UP_OUTPUT,
)
DOWN = Trellis35Projection(
    name="down",
    logical_input=DOWN_LOGICAL_INPUT,
    physical_input=DOWN_PHYSICAL_INPUT,
    output=DOWN_OUTPUT,
    suh_elements_per_expert=DOWN_PHYSICAL_INPUT,
    svh_elements_per_expert=DOWN_OUTPUT,
)


def _sum_checked(values: Iterable[int], description: str) -> int:
    result = 0
    for value in values:
        result = checked_add(result, value, description)
    return result


def estimate_trellis35_layout(
    *,
    baseline_arena_bytes: int,
    baseline_gate_up_bytes: int,
    baseline_down_bytes: int,
    rate_map: Sequence[int] | None = None,
) -> dict[str, object]:
    """Estimate the exact v1 routed-expert regions and baseline replacement."""
    if rate_map is None:
        # Byte totals are independent of expert identity at the fixed 64/64 budget.
        rate_map = (3,) * TRELLIS35_K3_COUNT + (4,) * TRELLIS35_K4_COUNT
    rates = validate_rate_map(rate_map)
    baseline_arena_bytes = _checked_nonnegative(baseline_arena_bytes, "baseline arena")
    old_gate_up = _checked_nonnegative(baseline_gate_up_bytes, "baseline Gate+Up")
    old_down = _checked_nonnegative(baseline_down_bytes, "baseline Down")
    old_experts = checked_add(old_gate_up, old_down, "baseline routed experts")
    if old_experts > baseline_arena_bytes:
        raise InvalidPlanError("baseline routed experts exceed the baseline arena")

    family_reports: dict[str, dict[str, int]] = {}
    all_region_bytes: list[int] = []
    logical_coefficients = 0
    encoded_coefficients = 0
    payload_bytes = 0
    suh_bytes = 0
    svh_bytes = 0
    descriptor_bytes = 0
    alignment_padding_bytes = 0

    for projection in (GATE_UP, DOWN):
        logical_per_layer = checked_mul(
            TRELLIS35_EXPERT_COUNT,
            projection.logical_coefficients_per_expert,
            description=f"{projection.name} logical coefficients per layer",
        )
        encoded_per_layer = checked_mul(
            TRELLIS35_EXPERT_COUNT,
            projection.encoded_coefficients_per_expert,
            description=f"{projection.name} encoded coefficients per layer",
        )
        payload_per_layer = _payload_bytes(
            projection.encoded_coefficients_per_expert,
            rates,
            f"{projection.name} payload per layer",
        )
        suh_per_layer = checked_mul(
            TRELLIS35_EXPERT_COUNT,
            projection.suh_elements_per_expert,
            TRELLIS35_SIDECAR_ELEMENT_BYTES,
            description=f"{projection.name} SUH per layer",
        )
        svh_per_layer = checked_mul(
            TRELLIS35_EXPERT_COUNT,
            projection.svh_elements_per_expert,
            TRELLIS35_SIDECAR_ELEMENT_BYTES,
            description=f"{projection.name} SVH per layer",
        )
        descriptor_per_layer = checked_mul(
            TRELLIS35_EXPERT_COUNT,
            TRELLIS35_DESCRIPTOR_BYTES,
            description=f"{projection.name} descriptors per layer",
        )

        # Runtime regions are individually 256-byte aligned. K3/K4 pool sizes
        # are computed separately so a future rate-count change cannot hide padding.
        k3_pool = _pool_bytes(
            projection.encoded_coefficients_per_expert,
            3,
            TRELLIS35_K3_COUNT,
            f"{projection.name} K3 pool",
        )
        k4_pool = _pool_bytes(
            projection.encoded_coefficients_per_expert,
            4,
            TRELLIS35_K4_COUNT,
            f"{projection.name} K4 pool",
        )
        raw_regions = (k3_pool, k4_pool, descriptor_per_layer, suh_per_layer, svh_per_layer)
        aligned_regions = tuple(
            align_up(value, TRELLIS35_ALIGNMENT, f"{projection.name} region")
            for value in raw_regions
        )
        padding_per_layer = sum(aligned_regions) - sum(raw_regions)
        total_per_layer = _sum_checked(aligned_regions, f"{projection.name} per-layer bytes")

        family_reports[projection.name] = {
            "logical_input": projection.logical_input,
            "physical_input": projection.physical_input,
            "output": projection.output,
            "logical_coefficients": checked_mul(
                logical_per_layer, TRELLIS35_LAYER_COUNT,
                description=f"{projection.name} logical coefficients total",
            ),
            "encoded_coefficients": checked_mul(
                encoded_per_layer, TRELLIS35_LAYER_COUNT,
                description=f"{projection.name} encoded coefficients total",
            ),
            "k3_payload_bytes_per_layer": k3_pool,
            "k4_payload_bytes_per_layer": k4_pool,
            "trellis_payload_bytes": checked_mul(
                payload_per_layer, TRELLIS35_LAYER_COUNT,
                description=f"{projection.name} payload total",
            ),
            "suh_bytes": checked_mul(
                suh_per_layer, TRELLIS35_LAYER_COUNT,
                description=f"{projection.name} SUH total",
            ),
            "svh_bytes": checked_mul(
                svh_per_layer, TRELLIS35_LAYER_COUNT,
                description=f"{projection.name} SVH total",
            ),
            "rate_descriptor_bytes": checked_mul(
                descriptor_per_layer, TRELLIS35_LAYER_COUNT,
                description=f"{projection.name} descriptors total",
            ),
            "alignment_bytes": checked_mul(
                padding_per_layer, TRELLIS35_LAYER_COUNT,
                description=f"{projection.name} alignment padding total",
            ),
            "total_bytes": checked_mul(
                total_per_layer, TRELLIS35_LAYER_COUNT,
                description=f"{projection.name} total bytes",
            ),
        }
        report = family_reports[projection.name]
        logical_coefficients = checked_add(
            logical_coefficients, int(report["logical_coefficients"]), "logical expert coefficients"
        )
        encoded_coefficients = checked_add(
            encoded_coefficients, int(report["encoded_coefficients"]), "encoded expert coefficients"
        )
        payload_bytes = checked_add(payload_bytes, int(report["trellis_payload_bytes"]), "payload bytes")
        suh_bytes = checked_add(suh_bytes, int(report["suh_bytes"]), "SUH bytes")
        svh_bytes = checked_add(svh_bytes, int(report["svh_bytes"]), "SVH bytes")
        descriptor_bytes = checked_add(
            descriptor_bytes, int(report["rate_descriptor_bytes"]), "descriptor bytes"
        )
        alignment_padding_bytes = checked_add(
            alignment_padding_bytes, int(report["alignment_bytes"]), "alignment bytes"
        )
        all_region_bytes.append(int(report["total_bytes"]))

    codebook_bytes = 0  # v1 codebooks are kernel-owned and selected by validated ID.
    total_expert_bytes = _sum_checked(
        [*all_region_bytes, codebook_bytes], "Trellis35 expert bytes"
    )
    non_routed_baseline_bytes = baseline_arena_bytes - old_experts
    total_target_arena_bytes = checked_add(
        non_routed_baseline_bytes, total_expert_bytes, "estimated Target arena"
    )
    saving = baseline_arena_bytes - total_target_arena_bytes

    return {
        "logical_expert_coefficients": logical_coefficients,
        "encoded_expert_coefficients": encoded_coefficients,
        "trellis_payload_bytes": payload_bytes,
        "suh_bytes": suh_bytes,
        "svh_bytes": svh_bytes,
        "rate_descriptor_bytes": descriptor_bytes,
        "codebook_bytes": codebook_bytes,
        "alignment_bytes": alignment_padding_bytes,
        "total_expert_bytes": total_expert_bytes,
        "total_target_arena_bytes": total_target_arena_bytes,
        "saving_vs_locked_nvfp4_bytes": saving,
        "payload_bpw_encoded": payload_bytes * 8 / encoded_coefficients,
        "payload_bpw_logical": payload_bytes * 8 / logical_coefficients,
        "effective_expert_bpw": total_expert_bytes * 8 / logical_coefficients,
        "baseline": {
            "target_arena_bytes": baseline_arena_bytes,
            "routed_gate_up_bytes": old_gate_up,
            "routed_down_bytes": old_down,
            "non_routed_bytes": non_routed_baseline_bytes,
        },
        "families": family_reports,
    }
