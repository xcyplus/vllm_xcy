# Slot-Aware KV Offload Demo

This experimental policy admits reusable instruction, schema, context, system,
and unknown prompt blocks to KV offload storage while skipping personalized
slot-value blocks.

## Inspect the Prompt

```bash
python examples/features/slot_offload_demo.py \
  --model <model-name-or-local-path> \
  --block-size 16
```

The script prints each block's token range, decoded text, type, and
`STORE`/`SKIP` decision. It also writes `slot_request.json` for the HTTP demo.

## Start vLLM

```bash
vllm serve <model-name-or-local-path> \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "cpu_bytes_to_use": 10737418240,
      "offload_prompt_only": true,
      "slot_offload_policy": "prefer_template",
      "slot_offload_store_types": [
        "instruction", "schema", "context", "system", "unknown"
      ],
      "slot_offload_log_decisions": true
    }
  }'
```

## Submit the Request

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  --data @slot_request.json
```

The scheduler emits log lines like:

```text
SlotOffload request=... block=2 tokens=[32,48) type=schema decision=STORE ...
SlotOffload request=... block=3 tokens=[48,64) type=slot decision=SKIP ...
```

`STORE` is an admission decision. The offload backend may skip the physical
write if the block is already present. `SKIP` means the block is not copied to
the offload tier by this request; it is not a guarantee that the block remains
resident in GPU memory indefinitely.

For prefix reuse, place shared instructions and schema before personalized slot
values. vLLM can only load a consecutive matching prefix; a slot difference near
the beginning prevents later blocks from being prefix hits.

## Value-Aware Admission

Set `slot_offload_policy` to `value_aware` to combine exact block composition,
reuse history, estimated transfer benefit, and CPU-cache pressure:

```json
{
  "cpu_bytes_to_use": 10737418240,
  "slot_offload_policy": "value_aware",
  "slot_offload_min_accesses": 2,
  "slot_offload_max_slot_ratio": 0.5,
  "slot_offload_prefill_us_per_token": 20.0,
  "slot_offload_gpu_to_cpu_gbps": 12.0,
  "slot_offload_cpu_to_gpu_gbps": 12.0,
  "slot_offload_structure_weight": 0.5,
  "slot_offload_hotness_weight": 0.3,
  "slot_offload_cost_weight": 0.2,
  "slot_offload_base_threshold": 0.2,
  "slot_offload_pressure_scale": 0.6,
  "slot_offload_log_decisions": true
}
```

For a block `b`, the policy computes:

```text
value(b) = 0.5 * structure_score
         + 0.3 * hotness_score
         + 0.2 * cost_score

threshold = base_threshold + pressure_scale * cpu_cache_usage
```

The first `slot_offload_min_accesses - 1` observations are skipped. Blocks with
a slot-token ratio above `slot_offload_max_slot_ratio` are also skipped. The
cost score compares expected prefill recomputation time with GPU-to-CPU store
and CPU-to-GPU load time. Bandwidth is expressed in decimal GB/s and prefill
cost in microseconds per token.

Requests without `slot_offload` metadata retain native STORE behavior. This
keeps value-aware admission opt-in for annotated workloads.

## Baseline Comparison

Stop any server already using port 8000, activate the repository virtual
environment, and run the three modes from the repository root:

```bash
bash examples/features/run_slot_offload_benchmark.sh baseline
bash examples/features/run_slot_offload_benchmark.sh slot
bash examples/features/run_slot_offload_benchmark.sh value
```

All modes use the same deterministic workload, a 16 MiB GPU KV cache, and a
128 MiB CPU offload tier. The baseline uses native offloading and ignores the
request's slot metadata. The slot mode skips slot-value blocks, while value
mode also considers reuse history, estimated cost, and cache pressure.

The default 16 groups are representative scenarios rather than numeric
placeholders: personal and mortgage lending, credit-card limits, small-business
lending, transaction fraud, insurance underwriting, health screening, support
triage, supplier risk, contract compliance, assignment grading, cybersecurity,
employee attrition, logistics, content moderation, and predictive maintenance.
Each scenario has distinct fixed rules and fields; only its instance values
change across repeats.

Compare the completed runs:

```bash
.venv/bin/python benchmarks/compare_slot_offload.py \
  --result-root slot_results
```

Outputs:

```text
slot_results/
  baseline/
    client.log
    requests.csv
    server.log
    summary.json
  slot/
    client.log
    requests.csv
    server.log
    summary.json
  value/
    client.log
    requests.csv
    server.log
    summary.json
  comparison.md
```

`requests.csv` contains per-request TTFT and end-to-end latency.
`summary.json` contains latency aggregates and metric deltas for the run.
`comparison.md` reports native, binary slot, and value-aware results together.
