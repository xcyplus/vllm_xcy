# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark structured-prompt KV offloading against a running vLLM server."""

import argparse
import csv
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from vllm.entrypoints.slot_offload_prompt import (
    PromptPart,
    build_slot_offload_prompt,
)


METRIC_NAMES = (
    "vllm:kv_offload_store_bytes_total",
    "vllm:kv_offload_load_bytes_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode", choices=("baseline", "slot", "value"), required=True
    )
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def read_metrics(endpoint: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{endpoint}/metrics", timeout=30) as response:
        text = response.read().decode("utf-8")

    metrics = {name: 0.0 for name in METRIC_NAMES}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        if metric_name in metrics:
            metrics[metric_name] += float(line.rsplit(" ", 1)[-1])
    return metrics


def build_workload(
    tokenizer: object,
    groups: int,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for repeat_idx in range(repeats):
        group_order = list(range(groups))
        random.Random(seed + repeat_idx).shuffle(group_order)
        for group_idx in group_order:
            # Shared instruction and schema are placed before all changing values
            # so vLLM can reuse them as a contiguous prefix.
            parts = [
                PromptPart(
                    "You are a structured risk-analysis assistant. "
                    f"Use business policy template {group_idx:03d}.\n",
                    "instruction",
                ),
                PromptPart(
                    "Input fields: customer_id, age, monthly_income, debt, "
                    "credit_score, overdue_count, requested_amount.\n"
                    "Output fields: risk_level, reasons, approval, advice.\n"
                    "Analyze only the following instance values.\n",
                    "schema",
                ),
                PromptPart(
                    "instance_values: "
                    f"customer_id=C{group_idx:03d}-{repeat_idx:04d}; "
                    f"age={20 + (repeat_idx * 7 + group_idx) % 55}; "
                    f"monthly_income={5000 + repeat_idx * 317 + group_idx * 41}; "
                    f"debt={10000 + repeat_idx * 997 + group_idx * 113}; "
                    f"credit_score={520 + (repeat_idx * 17 + group_idx) % 260}; "
                    f"overdue_count={(repeat_idx + group_idx) % 6}; "
                    "requested_amount="
                    f"{50000 + repeat_idx * 1231 + group_idx * 271}.\n",
                    "slot",
                ),
            ]
            prompt, metadata = build_slot_offload_prompt(parts, tokenizer)
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            requests.append(
                {
                    "group": group_idx,
                    "repeat": repeat_idx,
                    "prompt": prompt,
                    "metadata": metadata,
                    "prompt_tokens": len(token_ids),
                }
            )
    return requests


def send_streaming_request(
    endpoint: str,
    model: str,
    item: dict[str, Any],
    max_tokens: int,
) -> tuple[float, float]:
    body = {
        "model": model,
        "prompt": item["prompt"],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "kv_transfer_params": {"slot_offload": item["metadata"]},
    }
    request = urllib.request.Request(
        f"{endpoint}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_token_at: float | None = None
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line[6:])
            choices = payload.get("choices") or []
            if choices and choices[0].get("text") and first_token_at is None:
                first_token_at = time.perf_counter()
    end = time.perf_counter()
    if first_token_at is None:
        first_token_at = end
    return (first_token_at - start) * 1000, (end - start) * 1000


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    workload = build_workload(tokenizer, args.groups, args.repeats, args.seed)

    metrics_before = read_metrics(args.endpoint)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(workload):
        ttft_ms, e2e_ms = send_streaming_request(
            args.endpoint, args.model, item, args.max_tokens
        )
        row = {
            "mode": args.mode,
            "request_index": index,
            "group": item["group"],
            "repeat": item["repeat"],
            "prompt_tokens": item["prompt_tokens"],
            "ttft_ms": round(ttft_ms, 3),
            "e2e_ms": round(e2e_ms, 3),
        }
        rows.append(row)
        print(
            f"[{args.mode}] {index + 1}/{len(workload)} "
            f"group={item['group']} ttft={ttft_ms:.2f}ms e2e={e2e_ms:.2f}ms"
        )

    metrics_after = read_metrics(args.endpoint)
    metric_deltas = {
        name: metrics_after[name] - metrics_before[name] for name in METRIC_NAMES
    }
    ttfts = [float(row["ttft_ms"]) for row in rows]
    e2es = [float(row["e2e_ms"]) for row in rows]
    summary = {
        "mode": args.mode,
        "requests": len(rows),
        "groups": args.groups,
        "repeats": args.repeats,
        "prompt_tokens_total": sum(int(row["prompt_tokens"]) for row in rows),
        "ttft_ms_mean": statistics.fmean(ttfts),
        "ttft_ms_p50": statistics.median(ttfts),
        "ttft_ms_p95": percentile(ttfts, 0.95),
        "e2e_ms_mean": statistics.fmean(e2es),
        "e2e_ms_p50": statistics.median(e2es),
        "e2e_ms_p95": percentile(e2es, 0.95),
        "metric_deltas": metric_deltas,
    }

    csv_path = args.output_dir / "requests.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
