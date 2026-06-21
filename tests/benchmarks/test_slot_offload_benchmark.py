# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter

import pytest

from benchmarks.slot_offload_benchmark import (
    SCENARIO_TEMPLATES,
    analyze_workload,
    build_group_schedule,
    build_workload,
)


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text)))


def test_workload_uses_distinct_realistic_scenarios():
    workload = build_workload(CharTokenizer(), groups=3, repeats=2, seed=2026)

    assert len(workload) == 6
    assert {item["scenario"] for item in workload} == {
        "personal_loan",
        "mortgage_review",
        "credit_card_limit",
    }
    assert all("business policy template" not in item["prompt"] for item in workload)
    assert all("You are" not in item["prompt"] for item in workload)
    assert all("Input fields" not in item["prompt"] for item in workload)
    assert all("实例数据：" in item["prompt"] for item in workload)

    first_scenario = [item for item in workload if item["group"] == 0]
    assert len(first_scenario) == 2
    assert first_scenario[0]["prompt"] != first_scenario[1]["prompt"]
    assert (
        first_scenario[0]["metadata"]["slot_ranges"][0][0]
        == first_scenario[1]["metadata"]["slot_ranges"][0][0]
    )


def test_workload_rejects_more_groups_than_scenarios():
    with pytest.raises(ValueError, match="exceeds"):
        build_workload(
            CharTokenizer(),
            groups=len(SCENARIO_TEMPLATES) + 1,
            repeats=1,
            seed=2026,
        )


def test_finance_scenario_set_stays_in_one_domain():
    workload = build_workload(
        CharTokenizer(),
        groups=4,
        repeats=1,
        seed=2026,
        scenario_set="finance",
    )

    assert {item["scenario"] for item in workload} == {
        "personal_loan",
        "mortgage_review",
        "credit_card_limit",
        "small_business_loan",
    }


def test_one_hit_schedule_contains_cold_templates_once():
    schedule = build_group_schedule(
        groups=10,
        repeats=1,
        seed=2026,
        distribution="one_hit",
        num_requests=100,
        zipf_alpha=1.2,
        one_hit_fraction=0.5,
    )
    counts = Counter(schedule)

    assert all(counts[group] == 1 for group in range(5, 10))
    assert sum(counts[group] for group in range(5)) == 95


def test_zipf_schedule_is_deterministic():
    kwargs = {
        "groups": 8,
        "repeats": 1,
        "seed": 2026,
        "distribution": "zipf",
        "num_requests": 100,
        "zipf_alpha": 1.2,
        "one_hit_fraction": 0.5,
    }

    assert build_group_schedule(**kwargs) == build_group_schedule(**kwargs)


@pytest.mark.parametrize(
    ("distribution", "expected_distance"),
    [
        ("reuse_short", 1.0),
        ("reuse_medium", 4.0),
        ("reuse_long", 8.0),
    ],
)
def test_reuse_distance_traces_hold_frequency_constant(
    distribution: str, expected_distance: float
):
    workload = build_workload(
        CharTokenizer(),
        groups=8,
        repeats=1,
        seed=2026,
        scenario_set="finance",
        distribution=distribution,
        num_requests=80,
    )
    stats = analyze_workload(workload, offload_block_size=16)

    assert set(stats["template_request_counts"].values()) == {10}
    assert stats["reuse_distance_mean"] == expected_distance
    assert stats["template_working_set_tokens"] > 0
