# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native, binary SlotOffload, and value-aware benchmark summaries."""

import argparse
import json
from pathlib import Path
from typing import Any


STORE_BYTES = "vllm:kv_offload_store_bytes_total"
LOAD_BYTES = "vllm:kv_offload_load_bytes_total"
EXTERNAL_HITS = "vllm:external_prefix_cache_hits_total"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("slot_results"))
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reduction(baseline: float, slot: float) -> float:
    return 0.0 if baseline == 0 else (baseline - slot) / baseline * 100


def main() -> None:
    args = parse_args()
    baseline = load(args.result_root / "baseline" / "summary.json")
    slot = load(args.result_root / "slot" / "summary.json")
    value = load(args.result_root / "value" / "summary.json")
    baseline_metrics = baseline["metric_deltas"]
    slot_metrics = slot["metric_deltas"]
    value_metrics = value["metric_deltas"]

    store_reduction = reduction(
        baseline_metrics[STORE_BYTES], slot_metrics[STORE_BYTES]
    )
    ttft_change = reduction(baseline["ttft_ms_mean"], slot["ttft_ms_mean"])
    value_store_reduction = reduction(
        baseline_metrics[STORE_BYTES], value_metrics[STORE_BYTES]
    )
    value_ttft_change = reduction(
        baseline["ttft_ms_mean"], value["ttft_ms_mean"]
    )

    rows = (
        (
            "Mean TTFT (ms)",
            f"{baseline['ttft_ms_mean']:.3f}",
            f"{slot['ttft_ms_mean']:.3f} ({ttft_change:+.2f}%)",
            f"{value['ttft_ms_mean']:.3f} ({value_ttft_change:+.2f}%)",
        ),
        (
            "P95 TTFT (ms)",
            f"{baseline['ttft_ms_p95']:.3f}",
            f"{slot['ttft_ms_p95']:.3f}",
            f"{value['ttft_ms_p95']:.3f}",
        ),
        (
            "Store bytes",
            f"{baseline_metrics[STORE_BYTES]:.0f}",
            f"{slot_metrics[STORE_BYTES]:.0f} ({store_reduction:.2f}% less)",
            f"{value_metrics[STORE_BYTES]:.0f} "
            f"({value_store_reduction:.2f}% less)",
        ),
        (
            "Load bytes",
            f"{baseline_metrics[LOAD_BYTES]:.0f}",
            f"{slot_metrics[LOAD_BYTES]:.0f}",
            f"{value_metrics[LOAD_BYTES]:.0f}",
        ),
        (
            "External hit tokens",
            f"{baseline_metrics[EXTERNAL_HITS]:.0f}",
            f"{slot_metrics[EXTERNAL_HITS]:.0f}",
            f"{value_metrics[EXTERNAL_HITS]:.0f}",
        ),
    )
    table = "\n".join(f"| {' | '.join(row)} |" for row in rows)
    report = f"""# SlotOffload Benchmark Comparison

| Metric | Native offload | Binary SlotOffload | Value-aware |
| --- | ---: | ---: | ---: |
{table}

The runs use the same deterministic request order and prompt metadata. The
baseline ignores metadata, binary SlotOffload filters by priority type, and the
value-aware policy combines structure, reuse history, transfer benefit, and CPU
cache pressure.
"""
    output = args.result_root / "comparison.md"
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written to: {output}")


if __name__ == "__main__":
    main()
