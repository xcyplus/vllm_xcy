# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structure- and value-aware admission helpers for KV offloading.

The policy only filters stores. It never changes load behavior or attention
computation, so a conservative decision can reduce cache usefulness but cannot
alter model outputs.
"""

import math
from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


_DEFAULT_STORE_TYPES = frozenset(
    {"instruction", "template", "schema", "context", "system", "unknown"}
)
_RANGE_TYPES = (
    ("slot_ranges", "slot"),
    ("instruction_ranges", "instruction"),
    ("schema_ranges", "schema"),
    ("template_ranges", "template"),
    ("context_ranges", "context"),
    ("system_ranges", "system"),
)


def _as_float(config: Mapping[str, Any], name: str, default: float) -> float:
    try:
        return float(config.get(name, default))
    except (TypeError, ValueError):
        logger.warning("%s must be numeric; using %s", name, default)
        return default


def _as_int(config: Mapping[str, Any], name: str, default: int) -> int:
    try:
        return int(config.get(name, default))
    except (TypeError, ValueError):
        logger.warning("%s must be an integer; using %s", name, default)
        return default


@dataclass(frozen=True)
class SlotOffloadConfig:
    """Global structure-aware KV offload configuration."""

    enabled: bool = False
    policy: str = "disabled"
    store_types: frozenset[str] = _DEFAULT_STORE_TYPES
    type_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "instruction": 1.0,
            "schema": 0.9,
            "template": 1.0,
            "context": 0.7,
            "system": 1.0,
            "unknown": 1.0,
        }
    )
    slot_penalty: float = 1.0
    max_slot_ratio: float = 0.5
    min_accesses: int = 2
    hotness_saturation: float = 4.0
    hotness_decay: float = 0.01
    reuse_smoothing: float = 2.0
    prefill_us_per_token: float = 20.0
    gpu_to_cpu_gbps: float = 12.0
    cpu_to_gpu_gbps: float = 12.0
    structure_weight: float = 0.5
    hotness_weight: float = 0.3
    cost_weight: float = 0.2
    base_threshold: float = 0.2
    pressure_scale: float = 0.6
    max_tracker_size: int = 64_000

    @classmethod
    def from_extra_config(cls, extra_config: Mapping[str, Any]) -> "SlotOffloadConfig":
        policy = str(extra_config.get("slot_offload_policy", "disabled")).lower()
        enabled = policy not in ("", "0", "false", "none", "disabled", "off")
        if not enabled:
            return cls()

        raw_store_types = extra_config.get("slot_offload_store_types")
        if raw_store_types is None:
            store_types = _DEFAULT_STORE_TYPES
        elif isinstance(raw_store_types, str):
            store_types = frozenset(
                item.strip().lower()
                for item in raw_store_types.split(",")
                if item.strip()
            )
        elif isinstance(raw_store_types, Sequence):
            store_types = frozenset(str(item).lower() for item in raw_store_types)
        else:
            logger.warning(
                "slot_offload_store_types must be a string or sequence; got %r. "
                "Using defaults.",
                raw_store_types,
            )
            store_types = _DEFAULT_STORE_TYPES

        type_weights = {
            kind: _as_float(
                extra_config,
                f"slot_offload_{kind}_weight",
                default,
            )
            for kind, default in (
                ("instruction", 1.0),
                ("schema", 0.9),
                ("template", 1.0),
                ("context", 0.7),
                ("system", 1.0),
                ("unknown", 1.0),
            )
        }
        return cls(
            enabled=True,
            policy=policy,
            store_types=store_types,
            type_weights=type_weights,
            slot_penalty=max(
                0.0,
                _as_float(extra_config, "slot_offload_slot_penalty", 1.0),
            ),
            max_slot_ratio=min(
                1.0,
                max(
                    0.0,
                    _as_float(extra_config, "slot_offload_max_slot_ratio", 0.5),
                ),
            ),
            min_accesses=max(
                1, _as_int(extra_config, "slot_offload_min_accesses", 2)
            ),
            hotness_saturation=max(
                1.0,
                _as_float(extra_config, "slot_offload_hotness_saturation", 4.0),
            ),
            hotness_decay=max(
                0.0, _as_float(extra_config, "slot_offload_hotness_decay", 0.01)
            ),
            reuse_smoothing=max(
                0.0, _as_float(extra_config, "slot_offload_reuse_smoothing", 2.0)
            ),
            prefill_us_per_token=max(
                0.0,
                _as_float(extra_config, "slot_offload_prefill_us_per_token", 20.0),
            ),
            gpu_to_cpu_gbps=max(
                0.0,
                _as_float(extra_config, "slot_offload_gpu_to_cpu_gbps", 12.0),
            ),
            cpu_to_gpu_gbps=max(
                0.0,
                _as_float(extra_config, "slot_offload_cpu_to_gpu_gbps", 12.0),
            ),
            structure_weight=max(
                0.0,
                _as_float(extra_config, "slot_offload_structure_weight", 0.5),
            ),
            hotness_weight=max(
                0.0, _as_float(extra_config, "slot_offload_hotness_weight", 0.3)
            ),
            cost_weight=max(
                0.0, _as_float(extra_config, "slot_offload_cost_weight", 0.2)
            ),
            base_threshold=min(
                1.0,
                max(
                    0.0,
                    _as_float(extra_config, "slot_offload_base_threshold", 0.2),
                ),
            ),
            pressure_scale=max(
                0.0, _as_float(extra_config, "slot_offload_pressure_scale", 0.6)
            ),
            max_tracker_size=max(
                1, _as_int(extra_config, "slot_offload_max_tracker_size", 64_000)
            ),
        )


@dataclass(frozen=True)
class BlockStructure:
    """Token composition of one offloaded KV block."""

    token_counts: Mapping[str, int]
    block_tokens: int
    has_metadata: bool

    @property
    def slot_ratio(self) -> float:
        return self.token_counts.get("slot", 0) / max(1, self.block_tokens)

    @property
    def shared_ratio(self) -> float:
        return 1.0 - self.slot_ratio

    @property
    def primary_type(self) -> str:
        if not self.token_counts:
            return "unknown"
        return max(self.token_counts, key=self.token_counts.__getitem__)


@dataclass(frozen=True)
class SlotOffloadDecision:
    """Observable result of a block admission decision."""

    should_store: bool
    reason: str
    block_type: str
    shared_ratio: float
    slot_ratio: float
    access_count: int = 0
    structure_score: float = 0.0
    hotness_score: float = 0.0
    cost_score: float = 0.0
    value_score: float = 0.0
    threshold: float = 0.0
    cache_pressure: float = 0.0
    expected_benefit_us: float = 0.0


@dataclass
class _HotnessEntry:
    access_count: int
    decayed_count: float
    last_observation: int


def _get_slot_metadata(
    kv_transfer_params: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not kv_transfer_params:
        return {}
    raw = kv_transfer_params.get("slot_offload")
    if isinstance(raw, Mapping):
        return raw
    return kv_transfer_params


def _valid_ranges(raw_ranges: Any) -> list[tuple[int, int]]:
    if not isinstance(raw_ranges, Sequence) or isinstance(raw_ranges, (str, bytes)):
        return []
    result: list[tuple[int, int]] = []
    for item in raw_ranges:
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
        ):
            continue
        try:
            start, end = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            continue
        if 0 <= start < end:
            result.append((start, end))
    return result


def get_block_structure(
    kv_transfer_params: Mapping[str, Any] | None,
    block_idx: int,
    offloaded_block_size: int,
) -> BlockStructure:
    """Calculate exact type ratios for one offloaded block."""

    metadata = _get_slot_metadata(kv_transfer_params)
    block_types = metadata.get("block_types")
    if isinstance(block_types, Sequence) and not isinstance(block_types, (str, bytes)):
        if block_idx < len(block_types):
            block_type = str(block_types[block_idx]).lower()
            return BlockStructure(
                token_counts={block_type: offloaded_block_size},
                block_tokens=offloaded_block_size,
                has_metadata=True,
            )

    block_start = block_idx * offloaded_block_size
    labels = ["unknown"] * offloaded_block_size
    has_metadata = False
    # Earlier entries have priority when malformed metadata overlaps.
    for field_name, block_type in reversed(_RANGE_TYPES):
        ranges = _valid_ranges(metadata.get(field_name))
        has_metadata |= bool(ranges)
        for start, end in ranges:
            overlap_start = max(start, block_start)
            overlap_end = min(end, block_start + offloaded_block_size)
            for token_idx in range(overlap_start, overlap_end):
                labels[token_idx - block_start] = block_type

    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return BlockStructure(
        token_counts=counts,
        block_tokens=offloaded_block_size,
        has_metadata=has_metadata,
    )


def classify_offload_block(
    kv_transfer_params: Mapping[str, Any] | None,
    block_idx: int,
    offloaded_block_size: int,
) -> str:
    """Return the legacy priority-based type for an offloaded block."""

    metadata = _get_slot_metadata(kv_transfer_params)
    block_types = metadata.get("block_types")
    if isinstance(block_types, Sequence) and not isinstance(block_types, (str, bytes)):
        if block_idx < len(block_types):
            return str(block_types[block_idx]).lower()

    block_start = block_idx * offloaded_block_size
    block_end = block_start + offloaded_block_size
    for field_name, block_type in (
        ("instruction_ranges", "instruction"),
        ("slot_ranges", "slot"),
        ("template_ranges", "template"),
        ("schema_ranges", "schema"),
        ("context_ranges", "context"),
        ("system_ranges", "system"),
    ):
        for start, end in _valid_ranges(metadata.get(field_name)):
            if start < block_end and block_start < end:
                return block_type
    return "unknown"


class SlotOffloadAdmissionPolicy:
    """Stateful admission policy combining structure, reuse and cost."""

    def __init__(self, config: SlotOffloadConfig, kv_bytes_per_block: int = 0):
        self.config = config
        self.kv_bytes_per_block = max(0, kv_bytes_per_block)
        self._observation = 0
        self._hotness: OrderedDict[Hashable, _HotnessEntry] = OrderedDict()

    def _observe(self, key: Hashable) -> _HotnessEntry:
        self._observation += 1
        entry = self._hotness.pop(key, None)
        if entry is None:
            entry = _HotnessEntry(1, 1.0, self._observation)
        else:
            gap = self._observation - entry.last_observation
            entry.decayed_count *= math.exp(-self.config.hotness_decay * gap)
            entry.decayed_count += 1.0
            entry.access_count += 1
            entry.last_observation = self._observation
        self._hotness[key] = entry
        while len(self._hotness) > self.config.max_tracker_size:
            self._hotness.popitem(last=False)
        return entry

    def _structure_score(self, structure: BlockStructure) -> float:
        score = -self.config.slot_penalty * structure.slot_ratio
        for kind, count in structure.token_counts.items():
            if kind == "slot":
                continue
            ratio = count / max(1, structure.block_tokens)
            score += self.config.type_weights.get(kind, 0.0) * ratio
        return min(1.0, max(0.0, score))

    def _cost_score(
        self, access_count: int, offloaded_block_size: int
    ) -> tuple[float, float]:
        config = self.config
        if (
            self.kv_bytes_per_block <= 0
            or config.prefill_us_per_token <= 0
            or config.gpu_to_cpu_gbps <= 0
            or config.cpu_to_gpu_gbps <= 0
        ):
            return 0.0, 0.0

        reuse_probability = access_count / (
            access_count + config.reuse_smoothing
        )
        recompute_us = offloaded_block_size * config.prefill_us_per_token
        store_us = self.kv_bytes_per_block / (config.gpu_to_cpu_gbps * 1_000)
        load_us = self.kv_bytes_per_block / (config.cpu_to_gpu_gbps * 1_000)
        benefit_us = (
            reuse_probability * recompute_us
            - store_us
            - reuse_probability * load_us
        )
        score = min(1.0, max(0.0, benefit_us / recompute_us))
        return score, benefit_us

    def evaluate(
        self,
        *,
        key: Hashable,
        kv_transfer_params: Mapping[str, Any] | None,
        block_idx: int,
        offloaded_block_size: int,
        cache_pressure: float = 0.0,
    ) -> SlotOffloadDecision:
        """Return the STORE/SKIP decision for one completed KV block."""

        structure = get_block_structure(
            kv_transfer_params, block_idx, offloaded_block_size
        )
        pressure = min(1.0, max(0.0, cache_pressure))
        if not self.config.enabled or not structure.has_metadata:
            return SlotOffloadDecision(
                should_store=True,
                reason="disabled" if not self.config.enabled else "no_metadata",
                block_type=structure.primary_type,
                shared_ratio=structure.shared_ratio,
                slot_ratio=structure.slot_ratio,
                cache_pressure=pressure,
            )

        if self.config.policy != "value_aware":
            block_type = classify_offload_block(
                kv_transfer_params, block_idx, offloaded_block_size
            )
            should_store = block_type in self.config.store_types
            return SlotOffloadDecision(
                should_store=should_store,
                reason="store_type" if should_store else "filtered_type",
                block_type=block_type,
                shared_ratio=structure.shared_ratio,
                slot_ratio=structure.slot_ratio,
                cache_pressure=pressure,
            )

        entry = self._observe(key)
        structure_score = self._structure_score(structure)
        hotness_score = min(
            1.0, entry.decayed_count / self.config.hotness_saturation
        )
        cost_score, expected_benefit_us = self._cost_score(
            entry.access_count, offloaded_block_size
        )
        value_score = (
            self.config.structure_weight * structure_score
            + self.config.hotness_weight * hotness_score
            + self.config.cost_weight * cost_score
        )
        threshold = min(
            1.0,
            self.config.base_threshold + self.config.pressure_scale * pressure,
        )

        if structure.slot_ratio > self.config.max_slot_ratio:
            should_store = False
            reason = "slot_ratio"
        elif entry.access_count < self.config.min_accesses:
            should_store = False
            reason = "cold"
        elif expected_benefit_us < 0:
            should_store = False
            reason = "negative_benefit"
        else:
            should_store = value_score >= threshold
            reason = "admit" if should_store else "below_threshold"

        return SlotOffloadDecision(
            should_store=should_store,
            reason=reason,
            block_type=structure.primary_type,
            shared_ratio=structure.shared_ratio,
            slot_ratio=structure.slot_ratio,
            access_count=entry.access_count,
            structure_score=structure_score,
            hotness_score=hotness_score,
            cost_score=cost_score,
            value_score=value_score,
            threshold=threshold,
            cache_pressure=pressure,
            expected_benefit_us=expected_benefit_us,
        )


def should_store_offload_block(
    config: SlotOffloadConfig,
    kv_transfer_params: Mapping[str, Any] | None,
    block_idx: int,
    offloaded_block_size: int,
) -> bool:
    """Compatibility helper for the original stateless type policy."""

    if not config.enabled:
        return True
    block_type = classify_offload_block(
        kv_transfer_params=kv_transfer_params,
        block_idx=block_idx,
        offloaded_block_size=offloaded_block_size,
    )
    return block_type in config.store_types
