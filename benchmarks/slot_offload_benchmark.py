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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ScenarioTemplate:
    """A representative structured-prompt workload scenario."""

    name: str
    role: str
    rules: tuple[str, ...]
    input_fields: tuple[str, ...]
    output_fields: tuple[str, ...]

    def build_instruction(self) -> str:
        rules = "\n".join(
            f"{index}. {rule}" for index, rule in enumerate(self.rules, start=1)
        )
        return (
            f"You are {self.role}. Apply the following policy exactly.\n"
            f"{rules}\n"
            "Do not invent missing facts and explain every decision.\n"
        )

    def build_schema(self) -> str:
        return (
            f"Input fields: {', '.join(self.input_fields)}.\n"
            f"Output fields: {', '.join(self.output_fields)}.\n"
            "Analyze only the following instance values.\n"
        )


SCENARIO_TEMPLATES = (
    ScenarioTemplate(
        name="personal_loan",
        role="a personal-loan risk analyst",
        rules=(
            "Treat credit scores below 600 as high risk.",
            "Flag debt above twelve months of income as excessive.",
            "Escalate applicants with more than two overdue payments.",
        ),
        input_fields=(
            "customer_id",
            "age",
            "monthly_income",
            "debt",
            "credit_score",
            "overdue_count",
            "requested_amount",
        ),
        output_fields=("risk_level", "reasons", "approval", "advice"),
    ),
    ScenarioTemplate(
        name="mortgage_review",
        role="a residential-mortgage reviewer",
        rules=(
            "Require at least two years of continuous employment.",
            "Flag requested amounts above eighty percent of collateral value.",
            "Escalate applications with credit scores below 640.",
        ),
        input_fields=(
            "customer_id",
            "monthly_income",
            "debt",
            "credit_score",
            "employment_years",
            "collateral_value",
            "requested_amount",
        ),
        output_fields=("risk_level", "loan_to_value", "approval", "conditions"),
    ),
    ScenarioTemplate(
        name="credit_card_limit",
        role="a credit-card limit analyst",
        rules=(
            "Reject limit increases when utilization exceeds ninety percent.",
            "Escalate customers with two or more overdue payments.",
            "Cap the recommended limit at three months of income.",
        ),
        input_fields=(
            "customer_id",
            "monthly_income",
            "credit_score",
            "overdue_count",
            "utilization_ratio",
            "requested_limit",
        ),
        output_fields=("risk_level", "recommended_limit", "approval", "reasons"),
    ),
    ScenarioTemplate(
        name="small_business_loan",
        role="a small-business lending analyst",
        rules=(
            "Escalate businesses operating for less than two years.",
            "Reject requests above half of annual revenue.",
            "Flag non-positive annual profit as high risk.",
        ),
        input_fields=(
            "business_id",
            "years_in_business",
            "annual_revenue",
            "annual_profit",
            "existing_debt",
            "requested_amount",
        ),
        output_fields=("risk_level", "cash_flow_assessment", "approval", "reasons"),
    ),
    ScenarioTemplate(
        name="transaction_fraud",
        role="a payment-fraud investigator",
        rules=(
            "Block transactions when anomaly score is at least 0.85.",
            "Escalate high-risk countries combined with untrusted devices.",
            "Review accounts with three or more failed attempts.",
        ),
        input_fields=(
            "transaction_id",
            "transaction_amount",
            "country_risk",
            "device_trust",
            "failed_attempts",
            "anomaly_score",
        ),
        output_fields=("fraud_risk", "action", "reasons", "review_priority"),
    ),
    ScenarioTemplate(
        name="insurance_underwriting",
        role="a health-insurance underwriter",
        rules=(
            "Escalate smokers requesting high coverage.",
            "Flag BMI values above 35 for medical review.",
            "Require review when two or more chronic conditions are reported.",
        ),
        input_fields=(
            "applicant_id",
            "age",
            "bmi",
            "smoker",
            "chronic_conditions",
            "coverage_amount",
        ),
        output_fields=("risk_class", "premium_factor", "decision", "requirements"),
    ),
    ScenarioTemplate(
        name="health_screening",
        role="a preventive-health screening assistant",
        rules=(
            "Flag fasting glucose above 7.0 as elevated.",
            "Flag systolic pressure at or above 140 as hypertensive.",
            "Recommend clinical review when multiple risk indicators coexist.",
        ),
        input_fields=(
            "patient_id",
            "age",
            "bmi",
            "blood_glucose",
            "systolic_bp",
            "diastolic_bp",
        ),
        output_fields=("risk_level", "risk_factors", "recommendations", "urgency"),
    ),
    ScenarioTemplate(
        name="support_ticket_triage",
        role="a customer-support ticket triage assistant",
        rules=(
            "Prioritize security and payment issues over general questions.",
            "Escalate negative sentiment after two previous contacts.",
            "Give premium customers priority when severity is equal.",
        ),
        input_fields=(
            "ticket_id",
            "customer_tier",
            "issue_category",
            "sentiment",
            "wait_hours",
            "previous_contacts",
        ),
        output_fields=("priority", "queue", "escalation", "response_strategy"),
    ),
    ScenarioTemplate(
        name="supplier_risk",
        role="a supplier-risk analyst",
        rules=(
            "Flag on-time delivery rates below ninety percent.",
            "Escalate defect rates above five percent.",
            "Treat dependency ratios above sixty percent as concentration risk.",
        ),
        input_fields=(
            "supplier_id",
            "on_time_rate",
            "defect_rate",
            "financial_score",
            "dependency_ratio",
        ),
        output_fields=("risk_level", "risk_factors", "action", "monitoring_plan"),
    ),
    ScenarioTemplate(
        name="contract_compliance",
        role="a commercial-contract compliance reviewer",
        rules=(
            "Escalate contracts with missing liability or termination clauses.",
            "Require legal review for high-risk jurisdictions.",
            "Apply enhanced review when contract value exceeds one million.",
        ),
        input_fields=(
            "contract_id",
            "contract_value",
            "jurisdiction_risk",
            "missing_clauses",
            "counterparty_risk",
        ),
        output_fields=("compliance_level", "issues", "required_review", "actions"),
    ),
    ScenarioTemplate(
        name="assignment_grading",
        role="an academic-assignment grading assistant",
        rules=(
            "Use rubric score as the primary grade component.",
            "Flag plagiarism scores above 0.30 for manual review.",
            "Apply the stated late penalty without changing rubric feedback.",
        ),
        input_fields=(
            "student_id",
            "rubric_score",
            "citation_count",
            "plagiarism_score",
            "late_days",
        ),
        output_fields=("final_grade", "rubric_feedback", "integrity_flag", "advice"),
    ),
    ScenarioTemplate(
        name="cybersecurity_alert",
        role="a security-operations alert analyst",
        rules=(
            "Treat privileged-account anomalies as critical.",
            "Escalate anomaly scores above 0.80.",
            "Increase priority when more than five hosts are affected.",
        ),
        input_fields=(
            "alert_id",
            "severity",
            "affected_hosts",
            "privilege_level",
            "anomaly_score",
        ),
        output_fields=("incident_priority", "containment", "reasons", "next_steps"),
    ),
    ScenarioTemplate(
        name="employee_attrition",
        role="an employee-retention risk analyst",
        rules=(
            "Flag satisfaction scores below forty as high risk.",
            "Escalate sustained overtime above sixty hours per month.",
            "Treat below-market salary ratios as an additional risk factor.",
        ),
        input_fields=(
            "employee_id",
            "tenure_years",
            "overtime_hours",
            "satisfaction_score",
            "salary_ratio",
        ),
        output_fields=(
            "attrition_risk",
            "risk_factors",
            "retention_actions",
            "priority",
        ),
    ),
    ScenarioTemplate(
        name="logistics_exception",
        role="a logistics-exception coordinator",
        rules=(
            "Escalate temperature breaches for controlled goods.",
            "Treat delays above twenty-four hours as severe.",
            "Prioritize high-value shipments on high-risk routes.",
        ),
        input_fields=(
            "shipment_id",
            "delay_hours",
            "temperature_breach",
            "route_risk",
            "goods_value",
        ),
        output_fields=("severity", "action", "customer_notice", "recovery_plan"),
    ),
    ScenarioTemplate(
        name="content_moderation",
        role="a content-safety moderation assistant",
        rules=(
            "Remove content with severe violence or self-harm risk.",
            "Escalate toxicity scores above 0.85.",
            "Use user reports as supporting evidence, not sole evidence.",
        ),
        input_fields=(
            "content_id",
            "toxicity_score",
            "violence_score",
            "self_harm_score",
            "user_reports",
        ),
        output_fields=("safety_level", "action", "policy_reasons", "review_required"),
    ),
    ScenarioTemplate(
        name="predictive_maintenance",
        role="an industrial predictive-maintenance analyst",
        rules=(
            "Escalate vibration readings above 8.0 millimeters per second.",
            "Flag operating temperatures above ninety degrees Celsius.",
            "Prioritize assets with repeated faults and long operating hours.",
        ),
        input_fields=(
            "asset_id",
            "vibration",
            "temperature",
            "operating_hours",
            "fault_count",
        ),
        output_fields=("failure_risk", "maintenance_priority", "reasons", "actions"),
    ),
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


def build_instance_values(group_idx: int, repeat_idx: int) -> dict[str, str]:
    """Build deterministic values covering all representative scenarios."""

    serial = group_idx * 100 + repeat_idx
    return {
        "customer_id": f"C{group_idx:03d}-{repeat_idx:04d}",
        "age": str(20 + (repeat_idx * 7 + group_idx) % 55),
        "monthly_income": str(5_000 + repeat_idx * 317 + group_idx * 41),
        "debt": str(10_000 + repeat_idx * 997 + group_idx * 113),
        "credit_score": str(520 + (repeat_idx * 17 + group_idx) % 260),
        "overdue_count": str((repeat_idx + group_idx) % 6),
        "requested_amount": str(50_000 + repeat_idx * 1_231 + group_idx * 271),
        "employment_years": str(1 + (repeat_idx + group_idx) % 12),
        "collateral_value": str(80_000 + serial * 2_137),
        "utilization_ratio": f"{0.35 + (serial % 61) / 100:.2f}",
        "requested_limit": str(5_000 + serial * 173),
        "business_id": f"B{group_idx:03d}-{repeat_idx:04d}",
        "years_in_business": str(1 + serial % 15),
        "annual_revenue": str(200_000 + serial * 7_919),
        "annual_profit": str(-10_000 + (serial % 20) * 8_500),
        "existing_debt": str(30_000 + serial * 1_337),
        "transaction_id": f"TX{group_idx:03d}-{repeat_idx:04d}",
        "transaction_amount": f"{50 + serial * 37.25:.2f}",
        "country_risk": ("low", "medium", "high")[serial % 3],
        "device_trust": ("trusted", "unknown", "untrusted")[serial % 3],
        "failed_attempts": str(serial % 7),
        "anomaly_score": f"{0.35 + (serial % 64) / 100:.2f}",
        "applicant_id": f"A{group_idx:03d}-{repeat_idx:04d}",
        "bmi": f"{18.0 + (serial % 210) / 10:.1f}",
        "smoker": "yes" if serial % 4 == 0 else "no",
        "chronic_conditions": str(serial % 4),
        "coverage_amount": str(100_000 + serial * 3_721),
        "patient_id": f"P{group_idx:03d}-{repeat_idx:04d}",
        "blood_glucose": f"{4.5 + (serial % 45) / 10:.1f}",
        "systolic_bp": str(105 + serial % 55),
        "diastolic_bp": str(65 + serial % 35),
        "ticket_id": f"T{group_idx:03d}-{repeat_idx:04d}",
        "customer_tier": ("standard", "gold", "premium")[serial % 3],
        "issue_category": ("general", "payment", "security")[serial % 3],
        "sentiment": ("positive", "neutral", "negative")[serial % 3],
        "wait_hours": str(1 + serial % 48),
        "previous_contacts": str(serial % 5),
        "supplier_id": f"S{group_idx:03d}-{repeat_idx:04d}",
        "on_time_rate": f"{0.78 + (serial % 23) / 100:.2f}",
        "defect_rate": f"{0.01 + (serial % 9) / 100:.2f}",
        "financial_score": str(40 + serial % 61),
        "dependency_ratio": f"{0.20 + (serial % 66) / 100:.2f}",
        "contract_id": f"CT{group_idx:03d}-{repeat_idx:04d}",
        "contract_value": str(100_000 + serial * 51_337),
        "jurisdiction_risk": ("low", "medium", "high")[serial % 3],
        "missing_clauses": str(serial % 4),
        "counterparty_risk": ("low", "medium", "high")[serial % 3],
        "student_id": f"ST{group_idx:03d}-{repeat_idx:04d}",
        "rubric_score": str(55 + serial % 46),
        "citation_count": str(serial % 12),
        "plagiarism_score": f"{(serial % 51) / 100:.2f}",
        "late_days": str(serial % 6),
        "alert_id": f"AL{group_idx:03d}-{repeat_idx:04d}",
        "severity": ("low", "medium", "high", "critical")[serial % 4],
        "affected_hosts": str(1 + serial % 12),
        "privilege_level": ("user", "admin", "system")[serial % 3],
        "employee_id": f"E{group_idx:03d}-{repeat_idx:04d}",
        "tenure_years": str(1 + serial % 20),
        "overtime_hours": str(10 + serial % 71),
        "satisfaction_score": str(20 + serial % 81),
        "salary_ratio": f"{0.65 + (serial % 61) / 100:.2f}",
        "shipment_id": f"SH{group_idx:03d}-{repeat_idx:04d}",
        "delay_hours": str(serial % 49),
        "temperature_breach": "yes" if serial % 5 == 0 else "no",
        "route_risk": ("low", "medium", "high")[serial % 3],
        "goods_value": str(10_000 + serial * 2_119),
        "content_id": f"CO{group_idx:03d}-{repeat_idx:04d}",
        "toxicity_score": f"{0.10 + (serial % 90) / 100:.2f}",
        "violence_score": f"{0.05 + (serial % 75) / 100:.2f}",
        "self_harm_score": f"{0.02 + (serial % 60) / 100:.2f}",
        "user_reports": str(serial % 30),
        "asset_id": f"AS{group_idx:03d}-{repeat_idx:04d}",
        "vibration": f"{2.0 + (serial % 85) / 10:.1f}",
        "temperature": str(55 + serial % 51),
        "operating_hours": str(500 + serial * 137),
        "fault_count": str(serial % 8),
    }


def build_workload(
    tokenizer: object,
    groups: int,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    if groups > len(SCENARIO_TEMPLATES):
        raise ValueError(
            f"groups={groups} exceeds the {len(SCENARIO_TEMPLATES)} "
            "available scenarios"
        )

    requests: list[dict[str, Any]] = []
    for repeat_idx in range(repeats):
        group_order = list(range(groups))
        random.Random(seed + repeat_idx).shuffle(group_order)
        for group_idx in group_order:
            scenario = SCENARIO_TEMPLATES[group_idx]
            values = build_instance_values(group_idx, repeat_idx)
            instance = "; ".join(
                f"{field_name}={values[field_name]}"
                for field_name in scenario.input_fields
            )
            # Shared instruction and schema are placed before all changing values
            # so vLLM can reuse them as a contiguous prefix.
            parts = [
                PromptPart(scenario.build_instruction(), "instruction"),
                PromptPart(scenario.build_schema(), "schema"),
                PromptPart(f"instance_values: {instance}.\n", "slot"),
            ]
            prompt, metadata = build_slot_offload_prompt(parts, tokenizer)
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            requests.append(
                {
                    "group": group_idx,
                    "scenario": scenario.name,
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
            "scenario": item["scenario"],
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
        "scenarios": [scenario.name for scenario in SCENARIO_TEMPLATES[: args.groups]],
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
