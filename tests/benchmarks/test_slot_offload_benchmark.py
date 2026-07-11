# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter

import pytest

from benchmarks.slot_offload_benchmark import (
    SCENARIO_TEMPLATES,
    analyze_workload,
    build_group_schedule,
    build_workload,
    load_bfcl_records,
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


def test_agent_mcp_workload_contains_tool_schema_and_observation():
    workload = build_workload(
        CharTokenizer(),
        groups=2,
        repeats=2,
        seed=2026,
        scenario_set="agent_mcp",
        prompt_style="agent_mcp",
    )

    assert {item["scenario"] for item in workload} == {
        "personal_loan",
        "transaction_fraud",
    }
    assert all("MCP 工具定义" in item["prompt"] for item in workload)
    assert all("inputSchema" in item["prompt"] for item in workload)
    assert all("当前用户请求" in item["prompt"] for item in workload)
    assert all("工具观测结果" in item["prompt"] for item in workload)
    assert all("observation_ranges" in item["metadata"] for item in workload)

    first_scenario = [item for item in workload if item["group"] == 0]
    assert (
        first_scenario[0]["metadata"]["slot_ranges"][0][0]
        == first_scenario[1]["metadata"]["slot_ranges"][0][0]
    )
    assert first_scenario[0]["prompt"] != first_scenario[1]["prompt"]


def test_bfcl_workload_uses_public_tool_schema_shape(tmp_path):
    data = tmp_path / "bfcl.jsonl"
    data.write_text(
        "\n".join(
            [
                (
                    '{"id":"simple_0","question":[[{"role":"user",'
                    '"content":"Find the weather in Paris."}]],'
                    '"function":[{"name":"get_weather","description":'
                    '"Get weather.","parameters":{"type":"object",'
                    '"properties":{"city":{"type":"string"}}}}]}'
                ),
                (
                    '{"id":"simple_1","question":[[{"role":"user",'
                    '"content":"Book a hotel in Berlin."}]],'
                    '"function":[{"name":"book_hotel","description":'
                    '"Book hotel.","parameters":{"type":"object",'
                    '"properties":{"city":{"type":"string"}}}}]}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    workload = build_workload(
        CharTokenizer(),
        groups=2,
        repeats=2,
        seed=2026,
        scenario_set="bfcl",
        prompt_style="bfcl",
        bfcl_data=data,
    )

    assert len(load_bfcl_records(data)) == 2
    assert len(workload) == 4
    assert all("可用函数定义" in item["prompt"] for item in workload)
    assert all("当前用户请求" in item["prompt"] for item in workload)
    assert all("observation_ranges" in item["metadata"] for item in workload)
    assert {item["scenario"] for item in workload} == {"simple_0", "simple_1"}


def test_bfcl_workload_requires_data_path():
    with pytest.raises(ValueError, match="bfcl-data"):
        build_workload(
            CharTokenizer(),
            groups=1,
            repeats=1,
            seed=2026,
            scenario_set="bfcl",
            prompt_style="bfcl",
        )


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
