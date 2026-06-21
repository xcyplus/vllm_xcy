# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate SlotOffload experiment suites into CSV and Markdown tables."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


STORE_BYTES = "vllm:kv_offload_store_bytes_total"
LOAD_BYTES = "vllm:kv_offload_load_bytes_total"
STORE_TIME = "vllm:kv_offload_store_time_total"
LOAD_TIME = "vllm:kv_offload_load_time_total"
EXTERNAL_HITS = "vllm:external_prefix_cache_hits_total"
STRATEGY_ORDER = {
    "gpu_only": 0,
    "native": 1,
    "threshold": 2,
    "binary": 3,
    "value_structure": 4,
    "value_hotness": 5,
    "value_cost": 6,
    "value_min1": 7,
    "value_min2": 8,
    "value_min3": 9,
    "value": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def load_runs(result_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(result_root.rglob("summary.json")):
        config_path = summary_path.with_name("run_config.json")
        if not config_path.exists():
            continue
        summary = read_json(summary_path)
        config = read_json(config_path)
        metrics = summary.get("metric_deltas", {})
        store_bytes = float(metrics.get(STORE_BYTES, 0.0))
        load_bytes = float(metrics.get(LOAD_BYTES, 0.0))
        store_time = float(metrics.get(STORE_TIME, 0.0))
        load_time = float(metrics.get(LOAD_TIME, 0.0))
        external_hits = float(metrics.get(EXTERNAL_HITS, 0.0))
        workload_stats = summary.get("workload_stats", {})
        runs.append(
            {
                **config,
                "result_dir": str(summary_path.parent),
                "gpu_capacity_tokens": summary.get("gpu_capacity_tokens") or 0,
                "cpu_capacity_tokens": summary.get("cpu_capacity_tokens") or 0,
                "rho_gpu": summary.get("rho_gpu") or 0.0,
                "rho_cpu": summary.get("rho_cpu") or 0.0,
                "template_working_set_tokens": workload_stats.get(
                    "template_working_set_tokens", 0
                ),
                "shared_prefix_tokens_mean": workload_stats.get(
                    "shared_prefix_tokens_mean", 0.0
                ),
                "frequency_entropy_bits": workload_stats.get(
                    "template_frequency_entropy_bits", 0.0
                ),
                "reuse_distance_mean": workload_stats.get(
                    "reuse_distance_mean", 0.0
                ),
                "ttft_mean_ms": float(summary["ttft_ms_mean"]),
                "ttft_p95_ms": float(summary["ttft_ms_p95"]),
                "ttft_p99_ms": float(summary["ttft_ms_p99"]),
                "requests_per_second": float(summary["requests_per_second"]),
                "store_mb": store_bytes / 1_048_576,
                "load_mb": load_bytes / 1_048_576,
                "store_time_s": store_time,
                "load_time_s": load_time,
                "store_bandwidth_gbps": safe_ratio(store_bytes, store_time) / 1e9,
                "load_bandwidth_gbps": safe_ratio(load_bytes, load_time) / 1e9,
                "external_hit_tokens": external_hits,
                "hit_tokens_per_store_mb": safe_ratio(
                    external_hits, store_bytes / 1_048_576
                ),
            }
        )
    return runs


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimensions = (
        "model",
        "scenario_set",
        "distribution",
        "requests",
        "concurrency",
        "groups",
        "gpu_kv_bytes",
        "cpu_bytes",
        "max_tokens",
        "min_accesses",
        "strategy",
    )
    metrics = (
        "ttft_mean_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "requests_per_second",
        "store_mb",
        "load_mb",
        "external_hit_tokens",
        "hit_tokens_per_store_mb",
        "gpu_capacity_tokens",
        "cpu_capacity_tokens",
        "rho_gpu",
        "rho_cpu",
        "template_working_set_tokens",
        "shared_prefix_tokens_mean",
        "frequency_entropy_bits",
        "reuse_distance_mean",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[tuple(run[key] for key in dimensions)].append(run)

    aggregates: list[dict[str, Any]] = []
    for key, group in grouped.items():
        row = dict(zip(dimensions, key))
        row["run_count"] = len(group)
        for metric in metrics:
            values = [float(run[metric]) for run in group]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_stdev"] = stdev(values)
        aggregates.append(row)

    workload_dimensions = tuple(
        key for key in dimensions if key not in ("min_accesses", "strategy")
    )
    by_workload: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        by_workload[tuple(row[key] for key in workload_dimensions)].append(row)
    for rows in by_workload.values():
        native = next((row for row in rows if row["strategy"] == "native"), None)
        for row in rows:
            if native is None:
                row["ttft_reduction_vs_native_pct"] = 0.0
                row["store_reduction_vs_native_pct"] = 0.0
                row["hit_retention_vs_native_pct"] = 0.0
                continue
            row["ttft_reduction_vs_native_pct"] = 100 * (
                1
                - safe_ratio(
                    row["ttft_mean_ms_mean"], native["ttft_mean_ms_mean"]
                )
            )
            row["store_reduction_vs_native_pct"] = 100 * (
                1 - safe_ratio(row["store_mb_mean"], native["store_mb_mean"])
            )
            row["hit_retention_vs_native_pct"] = 100 * safe_ratio(
                row["external_hit_tokens_mean"],
                native["external_hit_tokens_mean"],
            )
    return sorted(
        aggregates,
        key=lambda row: (
            str(row["scenario_set"]),
            str(row["distribution"]),
            int(row["cpu_bytes"]),
            int(row["concurrency"]),
            STRATEGY_ORDER.get(str(row["strategy"]), 99),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SlotOffload Experiment Summary",
        "",
        "| Scenario | Distribution | CPU MiB | Conc. | Strategy | "
        "rho G | rho C | Reuse D | TTFT mean (ms) | P95 (ms) | P99 (ms) | "
        "Req/s | Store MiB | Load MiB | Hit tokens | TTFT vs native | "
        "Store vs native | Hit retention |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_set']} | {row['distribution']} | "
            f"{int(row['cpu_bytes']) / 1_048_576:.0f} | "
            f"{row['concurrency']} | {row['strategy']} | "
            f"{row['rho_gpu_mean']:.3f} | {row['rho_cpu_mean']:.3f} | "
            f"{row['reuse_distance_mean_mean']:.2f} | "
            f"{row['ttft_mean_ms_mean']:.3f} | "
            f"{row['ttft_p95_ms_mean']:.3f} | "
            f"{row['ttft_p99_ms_mean']:.3f} | "
            f"{row['requests_per_second_mean']:.2f} | "
            f"{row['store_mb_mean']:.2f} | {row['load_mb_mean']:.2f} | "
            f"{row['external_hit_tokens_mean']:.0f} | "
            f"{row['ttft_reduction_vs_native_pct']:+.2f}% | "
            f"{row['store_reduction_vs_native_pct']:+.2f}% | "
            f"{row['hit_retention_vs_native_pct']:.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = load_runs(args.result_root)
    if not runs:
        raise SystemExit(f"No completed runs found under {args.result_root}")
    aggregates = aggregate_runs(runs)
    write_csv(args.result_root / "runs.csv", runs)
    write_csv(args.result_root / "aggregate.csv", aggregates)
    write_markdown(args.result_root / "summary.md", aggregates)
    print(f"Aggregated {len(runs)} runs into {len(aggregates)} rows")
    print(f"Summary: {args.result_root / 'summary.md'}")


if __name__ == "__main__":
    main()
