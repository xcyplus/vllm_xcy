# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and inspect a cache-friendly SlotOffload request.

This script does not start a vLLM engine. It tokenizes a structured prompt,
prints each offloaded block and its admission decision, then writes an OpenAI
Completions-compatible request body that can be sent to a running vLLM server.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.slot_policy import (
    SlotOffloadConfig,
    classify_offload_block,
    should_store_offload_block,
)
from vllm.entrypoints.slot_offload_prompt import PromptPart, build_slot_offload_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("slot_request.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Shared content is deliberately placed before personalized values. vLLM's
    # prefix cache can then reuse the shared blocks until the first slot block.
    parts = [
        PromptPart("你是健康管理助手。请根据用户体检数据分析健康风险。\n\n", "instruction"),
        PromptPart(
            "输入字段定义：姓名、年龄、血糖、血压。\n"
            "输出格式：风险等级、主要原因、健康建议。\n\n",
            "schema",
        ),
        PromptPart("用户实例：\n", "schema"),
        PromptPart("姓名=张三；年龄=23；血糖=6.8；血压=140/90。\n", "slot"),
    ]
    prompt, metadata = build_slot_offload_prompt(parts, tokenizer)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    config = SlotOffloadConfig.from_extra_config(
        {
            "slot_offload_policy": "prefer_template",
            "slot_offload_store_types": [
                "instruction",
                "schema",
                "context",
                "system",
                "unknown",
            ],
        }
    )

    print(f"Prompt tokens: {len(token_ids)}; offload block size: {args.block_size}")
    for block_idx, start in enumerate(range(0, len(token_ids), args.block_size)):
        end = min(start + args.block_size, len(token_ids))
        block_type = classify_offload_block(metadata, block_idx, args.block_size)
        store = should_store_offload_block(
            config, metadata, block_idx, args.block_size
        )
        text = tokenizer.decode(token_ids[start:end])
        print(
            f"block={block_idx:02d} tokens=[{start:03d},{end:03d}) "
            f"type={block_type:11s} decision={'STORE' if store else 'SKIP ':5s} "
            f"text={text!r}"
        )

    request_body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0,
        "kv_transfer_params": {"slot_offload": metadata},
    }
    args.output.write_text(
        json.dumps(request_body, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRequest body written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
