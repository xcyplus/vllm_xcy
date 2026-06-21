# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.slot_policy import (
    SlotOffloadAdmissionPolicy,
    SlotOffloadConfig,
    get_block_structure,
)


def _config(**overrides) -> SlotOffloadConfig:
    extra_config = {
        "slot_offload_policy": "value_aware",
        "slot_offload_base_threshold": 0.2,
        "slot_offload_pressure_scale": 0.6,
        "slot_offload_min_accesses": 2,
    }
    extra_config.update(overrides)
    return SlotOffloadConfig.from_extra_config(extra_config)


def test_block_structure_calculates_token_ratios():
    structure = get_block_structure(
        {
            "slot_offload": {
                "instruction_ranges": [[0, 4]],
                "schema_ranges": [[4, 6]],
                "slot_ranges": [[6, 8]],
            }
        },
        block_idx=0,
        offloaded_block_size=8,
    )

    assert structure.token_counts == {
        "instruction": 4,
        "schema": 2,
        "slot": 2,
    }
    assert structure.shared_ratio == 0.75
    assert structure.slot_ratio == 0.25
    assert structure.primary_type == "instruction"


def test_value_aware_policy_delays_cold_shared_block():
    policy = SlotOffloadAdmissionPolicy(_config(), kv_bytes_per_block=196_608)
    params = {"slot_offload": {"block_types": ["instruction"]}}

    first = policy.evaluate(
        key="shared-block",
        kv_transfer_params=params,
        block_idx=0,
        offloaded_block_size=16,
    )
    second = policy.evaluate(
        key="shared-block",
        kv_transfer_params=params,
        block_idx=0,
        offloaded_block_size=16,
    )

    assert not first.should_store
    assert first.reason == "cold"
    assert second.should_store
    assert second.reason == "admit"
    assert second.access_count == 2
    assert second.expected_benefit_us > 0


def test_value_aware_policy_rejects_slot_dominated_block():
    policy = SlotOffloadAdmissionPolicy(_config(), kv_bytes_per_block=196_608)
    params = {
        "slot_offload": {
            "schema_ranges": [[0, 3]],
            "slot_ranges": [[3, 8]],
        }
    }

    policy.evaluate(
        key="personal-block",
        kv_transfer_params=params,
        block_idx=0,
        offloaded_block_size=8,
    )
    decision = policy.evaluate(
        key="personal-block",
        kv_transfer_params=params,
        block_idx=0,
        offloaded_block_size=8,
    )

    assert not decision.should_store
    assert decision.reason == "slot_ratio"
    assert decision.slot_ratio == 0.625


def test_cache_pressure_raises_admission_threshold():
    policy = SlotOffloadAdmissionPolicy(
        _config(slot_offload_cost_weight=0),
        kv_bytes_per_block=0,
    )
    params = {"slot_offload": {"block_types": ["schema"]}}

    policy.evaluate(
        key="shared-block",
        kv_transfer_params=params,
        block_idx=0,
        offloaded_block_size=16,
    )
    low_pressure = policy.evaluate(
        key="shared-block",
        kv_transfer_params=params,
        block_idx=0,
        offloaded_block_size=16,
        cache_pressure=0.0,
    )
    high_pressure = policy.evaluate(
        key="shared-block",
        kv_transfer_params=params,
        block_idx=0,
        offloaded_block_size=16,
        cache_pressure=1.0,
    )

    assert low_pressure.should_store
    assert not high_pressure.should_store
    assert high_pressure.reason == "below_threshold"
    assert high_pressure.threshold > low_pressure.threshold


def test_requests_without_metadata_keep_default_store_behavior():
    policy = SlotOffloadAdmissionPolicy(_config(), kv_bytes_per_block=196_608)

    decision = policy.evaluate(
        key="unannotated",
        kv_transfer_params=None,
        block_idx=0,
        offloaded_block_size=16,
        cache_pressure=1.0,
    )

    assert decision.should_store
    assert decision.reason == "no_metadata"
