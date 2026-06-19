# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slot-aware admission helpers for KV offloading.

The policy is intentionally conservative: it only filters stores when the
request explicitly supplies slot metadata. It never changes load behavior or
attention computation, so a poor classification can reduce cache usefulness but
cannot alter model outputs.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


_DEFAULT_STORE_TYPES = frozenset(
    {"instruction", "template", "schema", "context", "system", "unknown"}
)


@dataclass(frozen=True)
class SlotOffloadConfig:
    """Global slot-aware KV offload configuration."""

    enabled: bool = False
    store_types: frozenset[str] = _DEFAULT_STORE_TYPES

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

        return cls(enabled=True, store_types=store_types)


def _get_slot_metadata(kv_transfer_params: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not kv_transfer_params:
        return {}
    raw = kv_transfer_params.get("slot_offload")
    if isinstance(raw, Mapping):
        return raw
    return kv_transfer_params


def _range_overlaps_block(
    start: int,
    end: int,
    block_start: int,
    block_end: int,
) -> bool:
    return start < block_end and block_start < end


def _type_from_ranges(
    metadata: Mapping[str, Any],
    block_start: int,
    block_end: int,
) -> str | None:
    for field_name, block_type in (
        ("instruction_ranges", "instruction"),
        ("slot_ranges", "slot"),
        ("template_ranges", "template"),
        ("schema_ranges", "schema"),
        ("context_ranges", "context"),
        ("system_ranges", "system"),
    ):
        ranges = metadata.get(field_name)
        if not isinstance(ranges, Sequence):
            continue
        for item in ranges:
            if (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and len(item) == 2
            ):
                try:
                    start = int(item[0])
                    end = int(item[1])
                except (TypeError, ValueError):
                    continue
                if _range_overlaps_block(start, end, block_start, block_end):
                    return block_type
    return None


def classify_offload_block(
    kv_transfer_params: Mapping[str, Any] | None,
    block_idx: int,
    offloaded_block_size: int,
) -> str:
    """Classify an offloaded block from request-supplied metadata.

    Supported request metadata forms:

    - ``{"slot_offload": {"block_types": ["template", "slot", ...]}}``
    - ``{"slot_offload": {"slot_ranges": [[start, end]], ...}}``

    Ranges are token offsets in the request prompt. If no metadata matches, the
    block is ``unknown`` and is stored by default.
    """

    metadata = _get_slot_metadata(kv_transfer_params)
    block_types = metadata.get("block_types")
    if isinstance(block_types, Sequence) and not isinstance(block_types, (str, bytes)):
        if block_idx < len(block_types):
            return str(block_types[block_idx]).lower()

    block_start = block_idx * offloaded_block_size
    block_end = block_start + offloaded_block_size
    block_type = _type_from_ranges(metadata, block_start, block_end)
    return block_type or "unknown"


def should_store_offload_block(
    config: SlotOffloadConfig,
    kv_transfer_params: Mapping[str, Any] | None,
    block_idx: int,
    offloaded_block_size: int,
) -> bool:
    if not config.enabled:
        return True

    block_type = classify_offload_block(
        kv_transfer_params=kv_transfer_params,
        block_idx=block_idx,
        offloaded_block_size=offloaded_block_size,
    )
    return block_type in config.store_types
