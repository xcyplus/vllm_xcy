# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.summarize_slot_offload_experiments import aggregate_runs


def make_run(strategy: str, seed: int, ttft: float, store_mb: float) -> dict:
    return {
        "model": "model",
        "scenario_set": "finance",
        "prompt_style": "structured",
        "distribution": "one_hit",
        "requests": 100,
        "concurrency": 1,
        "groups": 8,
        "gpu_kv_bytes": 16,
        "cpu_bytes": 64,
        "max_tokens": 1,
        "min_accesses": 1,
        "strategy": strategy,
        "seed": seed,
        "ttft_mean_ms": ttft,
        "ttft_p95_ms": ttft + 1,
        "ttft_p99_ms": ttft + 2,
        "requests_per_second": 1000 / ttft,
        "store_mb": store_mb,
        "load_mb": 5.0,
        "external_hit_tokens": 100.0,
        "hit_tokens_per_store_mb": 100 / store_mb,
        "gpu_capacity_tokens": 1024,
        "cpu_capacity_tokens": 4096,
        "rho_gpu": 0.25,
        "rho_cpu": 1.0,
        "template_working_set_tokens": 4096,
        "shared_prefix_tokens_mean": 512,
        "frequency_entropy_bits": 3.0,
        "reuse_distance_mean": 8.0,
    }


def test_aggregate_runs_computes_seed_statistics_and_native_changes():
    runs = [
        make_run("native", 1, 10.0, 10.0),
        make_run("native", 2, 12.0, 10.0),
        make_run("value", 1, 8.0, 2.0),
        make_run("value", 2, 10.0, 2.0),
    ]

    rows = aggregate_runs(runs)
    value = next(row for row in rows if row["strategy"] == "value")

    assert value["run_count"] == 2
    assert value["ttft_mean_ms_mean"] == 9.0
    assert value["ttft_mean_ms_stdev"] == pytest.approx(2**0.5)
    expected_ttft_reduction = 100 * (1 - 9 / 11)
    assert value["ttft_reduction_vs_native_pct"] == pytest.approx(
        expected_ttft_reduction
    )
    assert value["store_reduction_vs_native_pct"] == 80.0
    assert value["hit_retention_vs_native_pct"] == 100.0
