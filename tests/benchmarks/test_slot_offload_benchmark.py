# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.slot_offload_benchmark import (
    SCENARIO_TEMPLATES,
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
    assert all("instance_values:" in item["prompt"] for item in workload)

    first_scenario = [item for item in workload if item["group"] == 0]
    assert len(first_scenario) == 2
    assert first_scenario[0]["prompt"] != first_scenario[1]["prompt"]
    assert (
        first_scenario[0]["metadata"]["slot_ranges"][0][0]
        == first_scenario[1]["metadata"]["slot_ranges"][0][0]
    )


def test_workload_rejects_more_groups_than_scenarios():
    with pytest.raises(ValueError, match="available scenarios"):
        build_workload(
            CharTokenizer(),
            groups=len(SCENARIO_TEMPLATES) + 1,
            repeats=1,
            seed=2026,
        )
