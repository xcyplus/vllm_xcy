# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.slot_offload_prompt import (
    PromptPart,
    build_block_types_from_ranges,
    build_schema_slot_prompt,
    build_slot_offload_prompt,
)


class CharTokenizer:

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text)))


def test_build_slot_offload_prompt_ranges():
    prompt, metadata = build_slot_offload_prompt(
        [
            PromptPart("请分析\n", "instruction"),
            PromptPart("年龄: ", "schema"),
            PromptPart("23\n", "slot"),
        ],
        CharTokenizer(),
    )

    assert prompt == "请分析\n年龄: 23\n"
    assert metadata == {
        "instruction_ranges": [[0, 4]],
        "schema_ranges": [[4, 8]],
        "slot_ranges": [[8, 11]],
    }


def test_build_schema_slot_prompt_and_block_types():
    prompt, metadata = build_schema_slot_prompt(
        instruction="请根据以下体检报告分析健康风险。",
        fields=[("年龄", "23"), ("血糖", "6.8")],
        output_schema="请按以下格式输出:\n风险等级:\n建议:",
        tokenizer=CharTokenizer(),
        title="体检报告",
    )

    assert "年龄: 23" in prompt
    assert "血糖: 6.8" in prompt
    assert metadata["instruction_ranges"]
    assert metadata["schema_ranges"]
    assert metadata["slot_ranges"]

    block_types = build_block_types_from_ranges(
        metadata=metadata,
        offloaded_block_size=8,
        num_prompt_tokens=len(prompt),
    )
    assert "slot" in block_types
    assert "schema" in block_types or "instruction" in block_types
