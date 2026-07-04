# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark structured-prompt KV offloading against a running vLLM server."""

import argparse
import csv
import json
import math
import random
import statistics
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "vllm:kv_offload_store_time_total",
    "vllm:kv_offload_load_time_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
)

FIELD_LABELS = {
    "customer_id": "客户编号",
    "age": "年龄",
    "monthly_income": "月收入",
    "debt": "负债",
    "credit_score": "信用评分",
    "overdue_count": "逾期次数",
    "requested_amount": "申请金额",
    "employment_years": "连续就业年限",
    "collateral_value": "抵押物价值",
    "utilization_ratio": "额度使用率",
    "requested_limit": "申请额度",
    "business_id": "企业编号",
    "years_in_business": "经营年限",
    "annual_revenue": "年营业收入",
    "annual_profit": "年度利润",
    "existing_debt": "现有负债",
    "transaction_id": "交易编号",
    "transaction_amount": "交易金额",
    "country_risk": "国家风险等级",
    "device_trust": "设备可信度",
    "failed_attempts": "失败尝试次数",
    "anomaly_score": "异常评分",
    "applicant_id": "投保人编号",
    "bmi": "体质指数",
    "smoker": "是否吸烟",
    "chronic_conditions": "慢性病数量",
    "coverage_amount": "保险金额",
    "patient_id": "患者编号",
    "blood_glucose": "空腹血糖",
    "systolic_bp": "收缩压",
    "diastolic_bp": "舒张压",
    "ticket_id": "工单编号",
    "customer_tier": "客户等级",
    "issue_category": "问题类别",
    "sentiment": "情绪倾向",
    "wait_hours": "等待时长",
    "previous_contacts": "历史联系次数",
    "supplier_id": "供应商编号",
    "on_time_rate": "准时交付率",
    "defect_rate": "缺陷率",
    "financial_score": "财务评分",
    "dependency_ratio": "供应依赖比例",
    "contract_id": "合同编号",
    "contract_value": "合同金额",
    "jurisdiction_risk": "司法辖区风险",
    "missing_clauses": "缺失条款数量",
    "counterparty_risk": "交易对手风险",
    "student_id": "学生编号",
    "rubric_score": "评分量表得分",
    "citation_count": "引用数量",
    "plagiarism_score": "抄袭风险评分",
    "late_days": "迟交天数",
    "alert_id": "告警编号",
    "severity": "严重等级",
    "affected_hosts": "受影响主机数",
    "privilege_level": "权限等级",
    "employee_id": "员工编号",
    "tenure_years": "任职年限",
    "overtime_hours": "月加班时长",
    "satisfaction_score": "满意度评分",
    "salary_ratio": "薪酬市场比率",
    "shipment_id": "运单编号",
    "delay_hours": "延误时长",
    "temperature_breach": "是否温控异常",
    "route_risk": "路线风险",
    "goods_value": "货物价值",
    "content_id": "内容编号",
    "toxicity_score": "毒性评分",
    "violence_score": "暴力评分",
    "self_harm_score": "自伤风险评分",
    "user_reports": "用户举报数",
    "asset_id": "设备编号",
    "vibration": "振动速度",
    "temperature": "运行温度",
    "operating_hours": "累计运行时长",
    "fault_count": "历史故障次数",
}


@dataclass(frozen=True)
class ToolSpec:
    """A synthetic MCP-style tool exposed to an agent prompt."""

    name: str
    description: str
    parameters: tuple[str, ...]


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
            f"你是{self.role}。请严格执行以下规则：\n"
            f"{rules}\n"
            "不得虚构缺失信息，并说明每项判断的依据。\n"
        )

    def build_schema(self) -> str:
        input_fields = "、".join(FIELD_LABELS[field] for field in self.input_fields)
        return (
            f"输入字段：{input_fields}。\n"
            f"输出字段：{'、'.join(self.output_fields)}。\n"
            "仅根据下方实例数据进行分析。\n"
        )

    def build_agent_instruction(self) -> str:
        rules = "\n".join(
            f"{index}. {rule}" for index, rule in enumerate(self.rules, start=1)
        )
        return (
            f"你是{self.role} Agent，运行在支持 MCP 工具调用的推理服务中。\n"
            "目标：读取当前用户请求，必要时参考工具观测结果，并生成可审计的"
            "结构化决策。\n"
            "执行规则：\n"
            f"{rules}\n"
            "不得虚构工具未返回的信息；不得把用户私有字段写入规则解释；"
            "最终输出必须符合下方 JSON 输出协议。\n"
        )


SCENARIO_TEMPLATES = (
    ScenarioTemplate(
        name="personal_loan",
        role="个人消费贷款风险分析师",
        rules=(
            "信用评分低于600分时判定为高风险。",
            "负债超过十二个月收入时标记为负债过高。",
            "逾期次数超过两次时转交人工复核。",
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
        output_fields=("风险等级", "主要原因", "审批结果", "风险建议"),
    ),
    ScenarioTemplate(
        name="mortgage_review",
        role="住房抵押贷款审核员",
        rules=(
            "申请人连续就业时间不得少于两年。",
            "申请金额超过抵押物价值百分之八十时标记为高贷款价值比。",
            "信用评分低于640分时转交人工复核。",
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
        output_fields=("风险等级", "贷款价值比", "审批结果", "附加条件"),
    ),
    ScenarioTemplate(
        name="credit_card_limit",
        role="信用卡额度分析师",
        rules=(
            "额度使用率超过百分之九十时拒绝提额。",
            "逾期次数达到两次时转交人工复核。",
            "建议额度不得超过三个月收入。",
        ),
        input_fields=(
            "customer_id",
            "monthly_income",
            "credit_score",
            "overdue_count",
            "utilization_ratio",
            "requested_limit",
        ),
        output_fields=("风险等级", "建议额度", "审批结果", "主要原因"),
    ),
    ScenarioTemplate(
        name="small_business_loan",
        role="小微企业贷款分析师",
        rules=(
            "经营时间不足两年的企业需要人工复核。",
            "申请金额超过年营业收入一半时拒绝申请。",
            "年度利润不为正时判定为高风险。",
        ),
        input_fields=(
            "business_id",
            "years_in_business",
            "annual_revenue",
            "annual_profit",
            "existing_debt",
            "requested_amount",
        ),
        output_fields=("风险等级", "现金流评估", "审批结果", "主要原因"),
    ),
    ScenarioTemplate(
        name="transaction_fraud",
        role="支付交易反欺诈分析师",
        rules=(
            "异常评分达到0.85及以上时立即阻断交易。",
            "高风险国家与不可信设备同时出现时转交人工复核。",
            "失败尝试达到三次时要求账户安全审查。",
        ),
        input_fields=(
            "transaction_id",
            "transaction_amount",
            "country_risk",
            "device_trust",
            "failed_attempts",
            "anomaly_score",
        ),
        output_fields=("欺诈风险", "处置动作", "主要原因", "复核优先级"),
    ),
    ScenarioTemplate(
        name="insurance_underwriting",
        role="健康保险核保员",
        rules=(
            "吸烟且申请高额保障时转交人工核保。",
            "体质指数超过35时要求医学审查。",
            "报告两种及以上慢性病时要求进一步审核。",
        ),
        input_fields=(
            "applicant_id",
            "age",
            "bmi",
            "smoker",
            "chronic_conditions",
            "coverage_amount",
        ),
        output_fields=("风险类别", "保费系数", "核保结论", "补充要求"),
    ),
    ScenarioTemplate(
        name="health_screening",
        role="预防性健康筛查助手",
        rules=(
            "空腹血糖超过7.0时标记为血糖偏高。",
            "收缩压达到140及以上时标记为高血压风险。",
            "多个风险指标同时存在时建议临床复查。",
        ),
        input_fields=(
            "patient_id",
            "age",
            "bmi",
            "blood_glucose",
            "systolic_bp",
            "diastolic_bp",
        ),
        output_fields=("风险等级", "风险因素", "健康建议", "紧急程度"),
    ),
    ScenarioTemplate(
        name="support_ticket_triage",
        role="客户服务工单分诊助手",
        rules=(
            "安全和支付问题的优先级高于一般咨询。",
            "负面情绪且历史联系达到两次时升级处理。",
            "严重程度相同时优先处理高级客户。",
        ),
        input_fields=(
            "ticket_id",
            "customer_tier",
            "issue_category",
            "sentiment",
            "wait_hours",
            "previous_contacts",
        ),
        output_fields=("优先级", "处理队列", "是否升级", "响应策略"),
    ),
    ScenarioTemplate(
        name="supplier_risk",
        role="供应商风险分析师",
        rules=(
            "准时交付率低于百分之九十时标记履约风险。",
            "缺陷率超过百分之五时转交质量复核。",
            "供应依赖比例超过百分之六十时标记集中度风险。",
        ),
        input_fields=(
            "supplier_id",
            "on_time_rate",
            "defect_rate",
            "financial_score",
            "dependency_ratio",
        ),
        output_fields=("风险等级", "风险因素", "处置动作", "监控计划"),
    ),
    ScenarioTemplate(
        name="contract_compliance",
        role="商业合同合规审核员",
        rules=(
            "缺失责任或终止条款的合同必须升级审核。",
            "涉及高风险司法辖区时必须进行法务复核。",
            "合同金额超过一百万元时执行增强审查。",
        ),
        input_fields=(
            "contract_id",
            "contract_value",
            "jurisdiction_risk",
            "missing_clauses",
            "counterparty_risk",
        ),
        output_fields=("合规等级", "发现问题", "复核要求", "处置建议"),
    ),
    ScenarioTemplate(
        name="assignment_grading",
        role="课程作业评分助手",
        rules=(
            "以评分量表得分作为成绩的主要依据。",
            "抄袭风险评分超过0.30时转交人工复核。",
            "按迟交天数执行扣分，但不得改变量表反馈。",
        ),
        input_fields=(
            "student_id",
            "rubric_score",
            "citation_count",
            "plagiarism_score",
            "late_days",
        ),
        output_fields=("最终成绩", "量表反馈", "诚信标记", "改进建议"),
    ),
    ScenarioTemplate(
        name="cybersecurity_alert",
        role="安全运营中心告警分析师",
        rules=(
            "特权账户出现异常时判定为严重事件。",
            "异常评分超过0.80时立即升级处置。",
            "受影响主机超过五台时提高事件优先级。",
        ),
        input_fields=(
            "alert_id",
            "severity",
            "affected_hosts",
            "privilege_level",
            "anomaly_score",
        ),
        output_fields=("事件优先级", "隔离措施", "判断依据", "后续步骤"),
    ),
    ScenarioTemplate(
        name="employee_attrition",
        role="员工留任风险分析师",
        rules=(
            "满意度评分低于40分时判定为高流失风险。",
            "月加班时长超过60小时且持续发生时升级关注。",
            "薪酬低于市场水平时增加一项流失风险因素。",
        ),
        input_fields=(
            "employee_id",
            "tenure_years",
            "overtime_hours",
            "satisfaction_score",
            "salary_ratio",
        ),
        output_fields=(
            "流失风险",
            "风险因素",
            "留任措施",
            "干预优先级",
        ),
    ),
    ScenarioTemplate(
        name="logistics_exception",
        role="物流异常协调员",
        rules=(
            "温控货物发生温度异常时立即升级处理。",
            "延误超过二十四小时判定为严重异常。",
            "高风险路线上的高价值货物优先处置。",
        ),
        input_fields=(
            "shipment_id",
            "delay_hours",
            "temperature_breach",
            "route_risk",
            "goods_value",
        ),
        output_fields=("严重程度", "处置动作", "客户通知", "恢复计划"),
    ),
    ScenarioTemplate(
        name="content_moderation",
        role="内容安全审核助手",
        rules=(
            "存在严重暴力或自伤风险的内容应当下架。",
            "毒性评分超过0.85时转交人工复核。",
            "用户举报只能作为辅助证据，不得作为唯一依据。",
        ),
        input_fields=(
            "content_id",
            "toxicity_score",
            "violence_score",
            "self_harm_score",
            "user_reports",
        ),
        output_fields=("安全等级", "处置动作", "规则依据", "是否人工复核"),
    ),
    ScenarioTemplate(
        name="predictive_maintenance",
        role="工业设备预测性维护分析师",
        rules=(
            "振动速度超过每秒8.0毫米时升级维护。",
            "运行温度超过90摄氏度时标记高温风险。",
            "历史故障频繁且运行时间较长的设备优先维护。",
        ),
        input_fields=(
            "asset_id",
            "vibration",
            "temperature",
            "operating_hours",
            "fault_count",
        ),
        output_fields=("故障风险", "维护优先级", "判断依据", "维护措施"),
    ),
)

SCENARIO_SETS = {
    "finance": (0, 1, 2, 3, 4, 5, 8, 9),
    "operations": (7, 8, 9, 11, 12, 13, 15),
    "agent_mcp": (0, 4, 7, 9, 10, 11, 13, 15),
    "mixed": tuple(range(len(SCENARIO_TEMPLATES))),
}

AGENT_TOOLSETS: dict[str, tuple[ToolSpec, ...]] = {
    "personal_loan": (
        ToolSpec("query_credit_profile", "查询客户征信摘要", ("customer_id",)),
        ToolSpec(
            "calculate_debt_ratio",
            "根据收入和负债计算偿债压力",
            ("monthly_income", "debt"),
        ),
        ToolSpec("create_manual_review", "创建人工复核任务", ("customer_id",)),
    ),
    "transaction_fraud": (
        ToolSpec(
            "lookup_device_risk",
            "查询支付设备与登录环境风险",
            ("transaction_id", "device_trust"),
        ),
        ToolSpec(
            "block_transaction",
            "对高风险支付交易执行拦截",
            ("transaction_id", "anomaly_score"),
        ),
    ),
    "support_ticket_triage": (
        ToolSpec("search_ticket_history", "查询客户历史工单", ("ticket_id",)),
        ToolSpec(
            "escalate_ticket",
            "将高优先级工单升级到人工队列",
            ("ticket_id", "issue_category"),
        ),
    ),
    "contract_compliance": (
        ToolSpec("lookup_clause_library", "查询标准合同条款库", ("contract_id",)),
        ToolSpec(
            "request_legal_review",
            "提交法务复核请求",
            ("contract_id", "jurisdiction_risk"),
        ),
    ),
    "assignment_grading": (
        ToolSpec("load_grading_rubric", "读取课程评分量表", ("student_id",)),
        ToolSpec(
            "check_similarity_report",
            "查询作业相似度检测报告",
            ("student_id", "plagiarism_score"),
        ),
    ),
    "cybersecurity_alert": (
        ToolSpec("query_security_logs", "查询安全日志片段", ("alert_id",)),
        ToolSpec(
            "isolate_hosts",
            "隔离受影响主机",
            ("alert_id", "affected_hosts"),
        ),
    ),
    "logistics_exception": (
        ToolSpec("track_shipment", "查询运单实时轨迹", ("shipment_id",)),
        ToolSpec(
            "notify_customer",
            "生成物流异常客户通知",
            ("shipment_id", "delay_hours"),
        ),
    ),
    "predictive_maintenance": (
        ToolSpec("query_sensor_window", "查询设备传感器窗口", ("asset_id",)),
        ToolSpec(
            "schedule_maintenance",
            "创建预测性维护计划",
            ("asset_id", "fault_count"),
        ),
    ),
}


def get_agent_tools(scenario: ScenarioTemplate) -> tuple[ToolSpec, ...]:
    tools = AGENT_TOOLSETS.get(scenario.name)
    if tools:
        return tools
    return (
        ToolSpec(
            f"lookup_{scenario.name}_record",
            f"查询{scenario.role}所需的业务记录",
            scenario.input_fields[:2],
        ),
        ToolSpec(
            f"create_{scenario.name}_review",
            f"创建{scenario.role}人工复核任务",
            scenario.input_fields[:1],
        ),
    )


def build_mcp_tool_schema(scenario: ScenarioTemplate) -> str:
    """Build a stable MCP-like tool schema section for one agent."""

    tools = []
    for tool in get_agent_tools(scenario):
        properties = {
            parameter: {
                "type": "string",
                "description": FIELD_LABELS.get(parameter, parameter),
            }
            for parameter in tool.parameters
        }
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(tool.parameters),
                },
            }
        )

    output_schema = {
        "type": "object",
        "properties": {
            field: {"type": "string"} for field in scenario.output_fields
        },
        "required": list(scenario.output_fields),
    }
    return (
        "MCP 工具定义(JSON Schema)：\n"
        f"{json.dumps({'tools': tools}, ensure_ascii=False, indent=2)}\n"
        "最终输出协议(JSON Schema)：\n"
        f"{json.dumps(output_schema, ensure_ascii=False, indent=2)}\n"
        "工具调用约束：只有当当前请求字段与工具 inputSchema 匹配时才允许"
        "调用；工具参数必须来自当前用户请求或工具观测结果。\n"
    )


def build_agent_slot(
    scenario: ScenarioTemplate,
    values: dict[str, str],
    *,
    group_idx: int,
    repeat_idx: int,
) -> str:
    inputs = {
        FIELD_LABELS[field_name]: values[field_name]
        for field_name in scenario.input_fields
    }
    request = {
        "request_id": f"agent-{group_idx:03d}-{repeat_idx:04d}",
        "agent": scenario.name,
        "user_message": f"请处理这条{scenario.role}请求，并给出结构化结论。",
        "input": inputs,
    }
    return (
        "当前用户请求(JSON)：\n"
        f"{json.dumps(request, ensure_ascii=False, indent=2)}\n"
    )


def build_agent_observation(
    scenario: ScenarioTemplate,
    values: dict[str, str],
    *,
    group_idx: int,
    repeat_idx: int,
) -> str:
    tool = get_agent_tools(scenario)[0]
    evidence = "；".join(
        f"{FIELD_LABELS[field]}={values[field]}" for field in tool.parameters
    )
    observation = {
        "jsonrpc": "2.0",
        "id": f"obs-{group_idx:03d}-{repeat_idx:04d}",
        "result": {
            "tool": tool.name,
            "content": [
                {
                    "type": "text",
                    "text": f"工具返回摘要：{evidence}；未发现字段缺失。",
                }
            ],
            "isError": False,
        },
    }
    return (
        "工具观测结果(JSON-RPC)：\n"
        f"{json.dumps(observation, ensure_ascii=False, indent=2)}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--requests", type=int)
    parser.add_argument(
        "--scenario-set", choices=tuple(SCENARIO_SETS), default="mixed"
    )
    parser.add_argument(
        "--prompt-style",
        choices=("structured", "agent_mcp"),
        default="structured",
    )
    parser.add_argument(
        "--distribution",
        choices=(
            "uniform",
            "hot_cold",
            "zipf",
            "one_hit",
            "reuse_short",
            "reuse_medium",
            "reuse_long",
        ),
        default="uniform",
    )
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--one-hit-fraction", type=float, default=0.5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--offload-block-size", type=int, default=16)
    parser.add_argument("--gpu-capacity-tokens", type=int)
    parser.add_argument("--cpu-capacity-tokens", type=int)
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


def analyze_workload(
    workload: list[dict[str, Any]], offload_block_size: int
) -> dict[str, Any]:
    """Measure template frequency, reuse distance, and cacheable working set."""

    if offload_block_size <= 0:
        raise ValueError("offload_block_size must be positive")
    counts = Counter(int(item["group"]) for item in workload)
    last_seen: dict[int, int] = {}
    reuse_distances: list[int] = []
    shared_tokens_by_group: dict[int, int] = {}
    for request_idx, item in enumerate(workload):
        group = int(item["group"])
        if group in last_seen:
            reuse_distances.append(request_idx - last_seen[group])
        last_seen[group] = request_idx
        slot_ranges = item["metadata"].get("slot_ranges", [])
        shared_tokens = (
            int(slot_ranges[0][0]) if slot_ranges else int(item["prompt_tokens"])
        )
        shared_tokens_by_group[group] = (
            shared_tokens // offload_block_size * offload_block_size
        )

    request_count = len(workload)
    probabilities = [count / request_count for count in counts.values()]
    entropy = -sum(
        probability * math.log2(probability) for probability in probabilities
    )
    working_set_tokens = sum(shared_tokens_by_group.values())
    return {
        "template_request_counts": {
            str(group): counts[group] for group in sorted(counts)
        },
        "template_frequency_entropy_bits": entropy,
        "unique_template_count": len(counts),
        "shared_prefix_tokens_mean": statistics.fmean(
            shared_tokens_by_group.values()
        ),
        "shared_prefix_tokens_by_group": {
            str(group): shared_tokens_by_group[group]
            for group in sorted(shared_tokens_by_group)
        },
        "template_working_set_tokens": working_set_tokens,
        "reuse_distance_mean": (
            statistics.fmean(reuse_distances) if reuse_distances else 0.0
        ),
        "reuse_distance_p50": (
            statistics.median(reuse_distances) if reuse_distances else 0.0
        ),
        "reuse_distance_p95": percentile(
            [float(value) for value in reuse_distances], 0.95
        ),
    }


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
        "country_risk": ("低", "中", "高")[serial % 3],
        "device_trust": ("可信", "未知", "不可信")[serial % 3],
        "failed_attempts": str(serial % 7),
        "anomaly_score": f"{0.35 + (serial % 64) / 100:.2f}",
        "applicant_id": f"A{group_idx:03d}-{repeat_idx:04d}",
        "bmi": f"{18.0 + (serial % 210) / 10:.1f}",
        "smoker": "是" if serial % 4 == 0 else "否",
        "chronic_conditions": str(serial % 4),
        "coverage_amount": str(100_000 + serial * 3_721),
        "patient_id": f"P{group_idx:03d}-{repeat_idx:04d}",
        "blood_glucose": f"{4.5 + (serial % 45) / 10:.1f}",
        "systolic_bp": str(105 + serial % 55),
        "diastolic_bp": str(65 + serial % 35),
        "ticket_id": f"T{group_idx:03d}-{repeat_idx:04d}",
        "customer_tier": ("普通", "金卡", "高级")[serial % 3],
        "issue_category": ("一般咨询", "支付问题", "安全问题")[serial % 3],
        "sentiment": ("正面", "中性", "负面")[serial % 3],
        "wait_hours": str(1 + serial % 48),
        "previous_contacts": str(serial % 5),
        "supplier_id": f"S{group_idx:03d}-{repeat_idx:04d}",
        "on_time_rate": f"{0.78 + (serial % 23) / 100:.2f}",
        "defect_rate": f"{0.01 + (serial % 9) / 100:.2f}",
        "financial_score": str(40 + serial % 61),
        "dependency_ratio": f"{0.20 + (serial % 66) / 100:.2f}",
        "contract_id": f"CT{group_idx:03d}-{repeat_idx:04d}",
        "contract_value": str(100_000 + serial * 51_337),
        "jurisdiction_risk": ("低", "中", "高")[serial % 3],
        "missing_clauses": str(serial % 4),
        "counterparty_risk": ("低", "中", "高")[serial % 3],
        "student_id": f"ST{group_idx:03d}-{repeat_idx:04d}",
        "rubric_score": str(55 + serial % 46),
        "citation_count": str(serial % 12),
        "plagiarism_score": f"{(serial % 51) / 100:.2f}",
        "late_days": str(serial % 6),
        "alert_id": f"AL{group_idx:03d}-{repeat_idx:04d}",
        "severity": ("低", "中", "高", "严重")[serial % 4],
        "affected_hosts": str(1 + serial % 12),
        "privilege_level": ("普通用户", "管理员", "系统")[serial % 3],
        "employee_id": f"E{group_idx:03d}-{repeat_idx:04d}",
        "tenure_years": str(1 + serial % 20),
        "overtime_hours": str(10 + serial % 71),
        "satisfaction_score": str(20 + serial % 81),
        "salary_ratio": f"{0.65 + (serial % 61) / 100:.2f}",
        "shipment_id": f"SH{group_idx:03d}-{repeat_idx:04d}",
        "delay_hours": str(serial % 49),
        "temperature_breach": "是" if serial % 5 == 0 else "否",
        "route_risk": ("低", "中", "高")[serial % 3],
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


def build_group_schedule(
    *,
    groups: int,
    repeats: int,
    seed: int,
    distribution: str,
    num_requests: int | None,
    zipf_alpha: float,
    one_hit_fraction: float,
) -> list[int]:
    """Build a deterministic template-access trace."""

    if groups <= 0 or repeats <= 0:
        raise ValueError("groups and repeats must be positive")
    request_count = num_requests if num_requests is not None else groups * repeats
    if request_count <= 0:
        raise ValueError("requests must be positive")

    rng = random.Random(seed)
    group_ids = list(range(groups))
    if distribution.startswith("reuse_"):
        if request_count % groups:
            raise ValueError(
                "reuse-distance traces require requests divisible by groups"
            )
        occurrences = request_count // groups
        if distribution == "reuse_short":
            return [group for group in group_ids for _ in range(occurrences)]
        if distribution == "reuse_long":
            return group_ids * occurrences
        if distribution != "reuse_medium":
            raise ValueError(f"Unsupported distribution: {distribution}")
        window_size = min(4, groups)
        schedule = []
        for start in range(0, groups, window_size):
            window = group_ids[start : start + window_size]
            schedule.extend(window * occurrences)
        return schedule

    if distribution == "uniform":
        schedule: list[int] = []
        while len(schedule) < request_count:
            cycle = group_ids.copy()
            rng.shuffle(cycle)
            schedule.extend(cycle)
        return schedule[:request_count]

    if distribution == "hot_cold":
        hot_count = max(1, round(groups * 0.2))
        cold_count = groups - hot_count
        weights = [0.8 / hot_count] * hot_count
        if cold_count:
            weights.extend([0.2 / cold_count] * cold_count)
        return rng.choices(group_ids, weights=weights, k=request_count)

    if distribution == "zipf":
        if zipf_alpha <= 0:
            raise ValueError("zipf_alpha must be positive")
        weights = [1 / ((rank + 1) ** zipf_alpha) for rank in group_ids]
        return rng.choices(group_ids, weights=weights, k=request_count)

    if distribution != "one_hit":
        raise ValueError(f"Unsupported distribution: {distribution}")
    if not 0 <= one_hit_fraction < 1:
        raise ValueError("one_hit_fraction must be in [0, 1)")
    cold_count = min(groups - 1, round(groups * one_hit_fraction))
    if request_count < cold_count:
        raise ValueError("requests must cover all one-hit templates")
    cold_groups = group_ids[-cold_count:] if cold_count else []
    hot_groups = group_ids[: groups - cold_count]
    schedule = cold_groups + rng.choices(
        hot_groups, k=request_count - len(cold_groups)
    )
    rng.shuffle(schedule)
    return schedule


def build_workload(
    tokenizer: object,
    groups: int,
    repeats: int,
    seed: int,
    *,
    scenario_set: str = "mixed",
    prompt_style: str = "structured",
    distribution: str = "uniform",
    num_requests: int | None = None,
    zipf_alpha: float = 1.2,
    one_hit_fraction: float = 0.5,
) -> list[dict[str, Any]]:
    scenario_indices = SCENARIO_SETS[scenario_set]
    if groups > len(scenario_indices):
        raise ValueError(
            f"groups={groups} exceeds the {len(scenario_indices)} scenarios "
            f"in set {scenario_set!r}"
        )

    schedule = build_group_schedule(
        groups=groups,
        repeats=repeats,
        seed=seed,
        distribution=distribution,
        num_requests=num_requests,
        zipf_alpha=zipf_alpha,
        one_hit_fraction=one_hit_fraction,
    )
    requests: list[dict[str, Any]] = []
    occurrences = [0] * groups
    for request_idx, group_idx in enumerate(schedule):
        repeat_idx = occurrences[group_idx]
        occurrences[group_idx] += 1
        scenario = SCENARIO_TEMPLATES[scenario_indices[group_idx]]
        values = build_instance_values(group_idx, repeat_idx)
        if prompt_style == "agent_mcp":
            parts = [
                PromptPart(scenario.build_agent_instruction(), "instruction"),
                PromptPart(build_mcp_tool_schema(scenario), "schema"),
                PromptPart(
                    build_agent_slot(
                        scenario,
                        values,
                        group_idx=group_idx,
                        repeat_idx=repeat_idx,
                    ),
                    "slot",
                ),
                PromptPart(
                    build_agent_observation(
                        scenario,
                        values,
                        group_idx=group_idx,
                        repeat_idx=repeat_idx,
                    ),
                    "observation",
                ),
            ]
        else:
            instance = "；".join(
                f"{FIELD_LABELS[field_name]}={values[field_name]}"
                for field_name in scenario.input_fields
            )
            # Shared content precedes all changing values for exact prefix reuse.
            parts = [
                PromptPart(scenario.build_instruction(), "instruction"),
                PromptPart(scenario.build_schema(), "schema"),
                PromptPart(f"实例数据：{instance}。\n", "slot"),
            ]
        prompt, metadata = build_slot_offload_prompt(parts, tokenizer)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        requests.append(
            {
                "request_index": request_idx,
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
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    workload = build_workload(
        tokenizer,
        args.groups,
        args.repeats,
        args.seed,
        scenario_set=args.scenario_set,
        prompt_style=args.prompt_style,
        distribution=args.distribution,
        num_requests=args.requests,
        zipf_alpha=args.zipf_alpha,
        one_hit_fraction=args.one_hit_fraction,
    )
    workload_stats = analyze_workload(workload, args.offload_block_size)

    if args.warmup_requests:
        warmup = build_workload(
            tokenizer,
            args.groups,
            args.repeats,
            args.seed - 1,
            scenario_set=args.scenario_set,
            prompt_style=args.prompt_style,
            distribution="uniform",
            num_requests=args.warmup_requests,
        )
        for item in warmup:
            send_streaming_request(
                args.endpoint, args.model, item, args.max_tokens
            )

    metrics_before = read_metrics(args.endpoint)
    rows: list[dict[str, Any]] = []

    def run_request(item: dict[str, Any]) -> dict[str, Any]:
        ttft_ms, e2e_ms = send_streaming_request(
            args.endpoint, args.model, item, args.max_tokens
        )
        return {
            "mode": args.mode,
            "request_index": item["request_index"],
            "group": item["group"],
            "scenario": item["scenario"],
            "repeat": item["repeat"],
            "prompt_tokens": item["prompt_tokens"],
            "ttft_ms": round(ttft_ms, 3),
            "e2e_ms": round(e2e_ms, 3),
        }

    measurement_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(run_request, item): item for item in workload}
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{args.mode}] {completed}/{len(workload)} "
                f"scenario={row['scenario']} ttft={row['ttft_ms']:.2f}ms "
                f"e2e={row['e2e_ms']:.2f}ms"
            )
    measurement_seconds = time.perf_counter() - measurement_start
    rows.sort(key=lambda row: int(row["request_index"]))

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
        "scenario_set": args.scenario_set,
        "prompt_style": args.prompt_style,
        "scenarios": sorted({str(row["scenario"]) for row in rows}),
        "distribution": args.distribution,
        "zipf_alpha": args.zipf_alpha,
        "one_hit_fraction": args.one_hit_fraction,
        "offload_block_size": args.offload_block_size,
        "gpu_capacity_tokens": args.gpu_capacity_tokens,
        "cpu_capacity_tokens": args.cpu_capacity_tokens,
        "rho_gpu": (
            args.gpu_capacity_tokens
            / workload_stats["template_working_set_tokens"]
            if args.gpu_capacity_tokens is not None
            else None
        ),
        "rho_cpu": (
            args.cpu_capacity_tokens
            / workload_stats["template_working_set_tokens"]
            if args.cpu_capacity_tokens is not None
            else None
        ),
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup_requests,
        "repeats": args.repeats,
        "measurement_seconds": measurement_seconds,
        "requests_per_second": len(rows) / measurement_seconds,
        "prompt_tokens_total": sum(int(row["prompt_tokens"]) for row in rows),
        "ttft_ms_mean": statistics.fmean(ttfts),
        "ttft_ms_p50": statistics.median(ttfts),
        "ttft_ms_p95": percentile(ttfts, 0.95),
        "ttft_ms_p99": percentile(ttfts, 0.99),
        "e2e_ms_mean": statistics.fmean(e2es),
        "e2e_ms_p50": statistics.median(e2es),
        "e2e_ms_p95": percentile(e2es, 0.95),
        "e2e_ms_p99": percentile(e2es, 0.99),
        "metric_deltas": metric_deltas,
        "workload_stats": workload_stats,
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
