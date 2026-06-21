#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

MODE="${1:-}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
RESULT_ROOT="${RESULT_ROOT:-slot_results}"

if [[ "$MODE" != "baseline" && "$MODE" != "slot" && "$MODE" != "value" ]]; then
  echo "Usage: $0 {baseline|slot|value}"
  exit 2
fi

if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "Port ${PORT} already has a healthy vLLM server. Stop it before running."
  exit 1
fi

RESULT_DIR="${RESULT_ROOT}/${MODE}"
mkdir -p "$RESULT_DIR"

if [[ "$MODE" == "value" ]]; then
  EXTRA_CONFIG='{
    "cpu_bytes_to_use": 134217728,
    "offload_prompt_only": true,
    "slot_offload_policy": "value_aware",
    "slot_offload_min_accesses": 2,
    "slot_offload_max_slot_ratio": 0.5,
    "slot_offload_base_threshold": 0.2,
    "slot_offload_pressure_scale": 0.6,
    "slot_offload_log_decisions": false
  }'
elif [[ "$MODE" == "slot" ]]; then
  EXTRA_CONFIG='{
    "cpu_bytes_to_use": 134217728,
    "offload_prompt_only": true,
    "slot_offload_policy": "prefer_template",
    "slot_offload_store_types": [
      "instruction", "schema", "context", "system", "unknown"
    ],
    "slot_offload_log_decisions": false
  }'
else
  EXTRA_CONFIG='{
    "cpu_bytes_to_use": 134217728,
    "offload_prompt_only": true
  }'
fi

KV_CONFIG=$(cat <<JSON
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": ${EXTRA_CONFIG}
}
JSON
)

echo "Starting ${MODE} server; logs: ${RESULT_DIR}/server.log"
.venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --port "$PORT" \
  --enforce-eager \
  --generation-config vllm \
  --enable-prefix-caching \
  --max-model-len 512 \
  --kv-cache-memory-bytes 16777216 \
  --kv-transfer-config "$KV_CONFIG" \
  >"${RESULT_DIR}/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "vLLM exited before becoming healthy. See ${RESULT_DIR}/server.log"
    exit 1
  fi
  sleep 1
done

if ! curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "Timed out waiting for vLLM. See ${RESULT_DIR}/server.log"
  exit 1
fi

echo "Server is healthy. Running ${MODE} workload."
.venv/bin/python benchmarks/slot_offload_benchmark.py \
  --endpoint "http://localhost:${PORT}" \
  --model "$MODEL" \
  --mode "$MODE" \
  --groups 16 \
  --repeats 8 \
  --max-tokens 1 \
  --output-dir "$RESULT_DIR" \
  2>&1 | tee "${RESULT_DIR}/client.log"

echo "Finished ${MODE}. Summary: ${RESULT_DIR}/summary.json"
