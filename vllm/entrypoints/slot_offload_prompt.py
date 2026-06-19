# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for building structured prompts with SlotOffload metadata.

Applications often assemble prompts from typed fragments such as an instruction,
field names, user-specific values, and output schemas. These helpers preserve
that structure as token ranges that the KV offloading connector can use for
slot-aware store admission.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


SlotKind = str


@dataclass(frozen=True)
class PromptPart:
    text: str
    kind: SlotKind


def _encode_len(tokenizer: object, text: str) -> int:
    encode = getattr(tokenizer, "encode")
    try:
        return len(encode(text, add_special_tokens=False))
    except TypeError:
        return len(encode(text))


def build_slot_offload_prompt(
    parts: Sequence[PromptPart | tuple[str, SlotKind]],
    tokenizer: object,
) -> tuple[str, dict[str, list[list[int]]]]:
    """Build a prompt and token-range metadata for SlotOffload.

    Args:
        parts: Ordered prompt fragments. Supported kinds include
            ``instruction``, ``schema``, ``slot``, ``context``, ``template`` and
            ``system``. Unknown kinds are still emitted as ``<kind>_ranges``.
        tokenizer: Any tokenizer object with an ``encode`` method.

    Returns:
        A tuple of ``(prompt, metadata)``. ``metadata`` can be placed under
        ``kv_transfer_params["slot_offload"]``.
    """

    prompt_chunks: list[str] = []
    ranges: dict[str, list[list[int]]] = {}
    cursor = 0

    for raw_part in parts:
        if isinstance(raw_part, PromptPart):
            text, kind = raw_part.text, raw_part.kind
        else:
            text, kind = raw_part
        kind = str(kind).lower()
        token_len = _encode_len(tokenizer, text)
        if token_len:
            ranges.setdefault(f"{kind}_ranges", []).append(
                [cursor, cursor + token_len]
            )
        prompt_chunks.append(text)
        cursor += token_len

    return "".join(prompt_chunks), ranges


def build_schema_slot_prompt(
    *,
    instruction: str,
    fields: Iterable[tuple[str, str]],
    output_schema: str,
    tokenizer: object,
    title: str = "输入信息",
) -> tuple[str, dict[str, list[list[int]]]]:
    """Build a common form-style prompt with schema and slot ranges.

    Field names and separators are tagged as ``schema``; field values are tagged
    as ``slot``. This covers workloads like medical reports, customer tickets,
    grading forms, and financial risk forms.
    """

    parts: list[PromptPart] = [
        PromptPart(instruction.rstrip() + "\n\n", "instruction"),
        PromptPart(title.rstrip() + ":\n", "schema"),
    ]
    for name, value in fields:
        parts.append(PromptPart(f"{name}: ", "schema"))
        parts.append(PromptPart(str(value).rstrip() + "\n", "slot"))
    parts.extend(
        [
            PromptPart("\n", "schema"),
            PromptPart(output_schema.rstrip() + "\n", "schema"),
        ]
    )
    return build_slot_offload_prompt(parts, tokenizer)


def build_block_types_from_ranges(
    metadata: dict[str, list[list[int]]],
    offloaded_block_size: int,
    num_prompt_tokens: int,
) -> list[str]:
    """Convert token ranges to coarse block types for easier experiments."""

    def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return a_start < b_end and b_start < a_end

    priority = (
        ("slot_ranges", "slot"),
        ("context_ranges", "context"),
        ("schema_ranges", "schema"),
        ("instruction_ranges", "instruction"),
        ("template_ranges", "template"),
        ("system_ranges", "system"),
    )
    num_blocks = (num_prompt_tokens + offloaded_block_size - 1) // offloaded_block_size
    block_types: list[str] = []
    for block_idx in range(num_blocks):
        block_start = block_idx * offloaded_block_size
        block_end = min(block_start + offloaded_block_size, num_prompt_tokens)
        block_type = "unknown"
        for range_name, kind in priority:
            for start, end in metadata.get(range_name, []):
                if overlaps(start, end, block_start, block_end):
                    block_type = kind
                    break
            if block_type != "unknown":
                break
        block_types.append(block_type)
    return block_types
