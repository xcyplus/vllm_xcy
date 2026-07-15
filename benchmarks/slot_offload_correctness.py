# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check deterministic output equivalence across SlotOffload policies."""

import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.slot_offload_benchmark import build_workload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare native, binary, and value SlotOffload outputs."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--gpu-kv-bytes", required=True)
    parser.add_argument("--cpu-bytes", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenario-set", default="agent_mcp")
    parser.add_argument("--prompt-style", default="agent_mcp")
    parser.add_argument("--distribution", default="zipf")
    parser.add_argument("--bfcl-data", type=Path)
    parser.add_argument("--startup-timeout", type=int, default=300)
    return parser.parse_args()


def build_kv_config(strategy: str, cpu_bytes: int) -> dict[str, Any]:
    extra_config: dict[str, Any] = {
        "cpu_bytes_to_use": cpu_bytes,
        "offload_prompt_only": True,
    }
    if strategy == "binary":
        extra_config.update({
            "slot_offload_policy": "prefer_template",
            "slot_offload_store_types": [
                "instruction",
                "schema",
                "context",
                "system",
                "unknown",
            ],
        })
    elif strategy == "value":
        extra_config.update({
            "slot_offload_policy": "value_aware",
            "slot_offload_min_accesses": 1,
            "slot_offload_max_slot_ratio": 0.5,
            "slot_offload_base_threshold": 0.5,
            "slot_offload_pressure_scale": 0.25,
        })
    elif strategy != "native":
        raise ValueError(f"Unsupported strategy: {strategy}")

    return {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra_config,
    }


def wait_for_server(port: int, proc: subprocess.Popen[Any],
                    timeout_seconds: int, log_path: Path) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early. See {log_path}")
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for vLLM. See {log_path}")


def start_server(args: argparse.Namespace,
                 strategy: str) -> tuple[subprocess.Popen[Any], Any]:
    log_path = args.output_dir / f"{strategy}_server.log"
    log_file = log_path.open("w", encoding="utf-8")
    command = [
        ".venv/bin/python",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--port",
        str(args.port),
        "--enforce-eager",
        "--generation-config",
        "vllm",
        "--enable-prefix-caching",
        "--max-model-len",
        str(args.max_model_len),
        "--kv-cache-memory-bytes",
        args.gpu_kv_bytes,
        "--kv-transfer-config",
        json.dumps(
            build_kv_config(strategy, args.cpu_bytes), ensure_ascii=False),
    ]
    proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    wait_for_server(args.port, proc, args.startup_timeout, log_path)
    return proc, log_file


def stop_server(proc: subprocess.Popen[Any], log_file: Any) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()


def send_completion(args: argparse.Namespace, item: dict[str, Any]) -> str:
    body = {
        "model": args.model,
        "prompt": item["prompt"],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "kv_transfer_params": {"slot_offload": item["metadata"]},
    }
    request = urllib.request.Request(
        f"http://localhost:{args.port}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["choices"][0]["text"])


def run_strategy(args: argparse.Namespace, strategy: str,
                 workload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    print(f"Starting {strategy}", flush=True)
    proc, log_file = start_server(args, strategy)
    outputs: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(workload, start=1):
            outputs.append({
                "request_index": item["request_index"],
                "group": item["group"],
                "scenario": item["scenario"],
                "repeat": item["repeat"],
                "prompt_tokens": item["prompt_tokens"],
                "output": send_completion(args, item),
            })
            print(f"{strategy}: {index}/{len(workload)}", flush=True)
    finally:
        stop_server(proc, log_file)
    path = args.output_dir / f"{strategy}_outputs.json"
    path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return outputs


def compare_outputs(outputs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    native = outputs["native"]
    summary: dict[str, Any] = {
        "requests": len(native),
        "strategies": ["native", "binary", "value"],
    }
    for strategy in ("binary", "value"):
        mismatches = []
        for index, row in enumerate(native):
            if row["output"] != outputs[strategy][index]["output"]:
                mismatches.append({
                    "request_index": row["request_index"],
                    "native_output": row["output"],
                    f"{strategy}_output": outputs[strategy][index]["output"],
                })
        match_count = len(native) - len(mismatches)
        summary[f"{strategy}_exact_matches"] = match_count
        summary[f"{strategy}_exact_match_rate"] = match_count / len(native)
        summary[f"{strategy}_mismatches"] = mismatches
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    workload = build_workload(
        tokenizer,
        args.groups,
        repeats=1,
        seed=args.seed,
        scenario_set=args.scenario_set,
        prompt_style=args.prompt_style,
        distribution=args.distribution,
        num_requests=args.requests,
        bfcl_data=args.bfcl_data,
    )
    (args.output_dir / "workload.json").write_text(
        json.dumps(
            [{
                "request_index": item["request_index"],
                "group": item["group"],
                "scenario": item["scenario"],
                "repeat": item["repeat"],
                "prompt_tokens": item["prompt_tokens"],
            } for item in workload],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    outputs = {
        strategy: run_strategy(args, strategy, workload)
        for strategy in ("native", "binary", "value")
    }
    summary = compare_outputs(outputs)
    summary.update({
        "model": args.model,
        "scenario_set": args.scenario_set,
        "prompt_style": args.prompt_style,
        "distribution": args.distribution,
        "requests": len(workload),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    })
    (args.output_dir / "correctness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
