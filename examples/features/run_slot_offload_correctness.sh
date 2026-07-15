#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

ACTION="${1:-status}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SCENARIO_SET="${SCENARIO_SET:-agent_mcp}"
PROMPT_STYLE="${PROMPT_STYLE:-$SCENARIO_SET}"
DISTRIBUTION="${DISTRIBUTION:-zipf}"
WORKLOAD_GROUPS="${WORKLOAD_GROUPS:-8}"
REQUESTS="${REQUESTS:-32}"
SEED="${SEED:-2026}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_TOKENS="${MAX_TOKENS:-16}"
GPU_KV_BYTES="${GPU_KV_BYTES:-12582912}"
CPU_BYTES_OVERRIDE="${CPU_BYTES_OVERRIDE:-$((GPU_KV_BYTES * 4))}"
PORT="${PORT:-8011}"
VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-True}"
BFCL_DATA="${BFCL_DATA:-}"

model_tag="${MODEL##*/}"
RESULT_DIR="${RESULT_DIR:-slot_correctness_${SCENARIO_SET}_${model_tag}_tok${MAX_TOKENS}}"
LOG_FILE="${LOG_FILE:-${RESULT_DIR}.log}"
PID_FILE="${PID_FILE:-${RESULT_DIR}.pid}"

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

print_config() {
  cat <<EOF
model: ${MODEL}
scenario_set: ${SCENARIO_SET}
prompt_style: ${PROMPT_STYLE}
distribution: ${DISTRIBUTION}
groups: ${WORKLOAD_GROUPS}
requests: ${REQUESTS}
seed: ${SEED}
gpu_kv_bytes: ${GPU_KV_BYTES}
cpu_bytes: ${CPU_BYTES_OVERRIDE}
max_model_len: ${MAX_MODEL_LEN}
max_tokens: ${MAX_TOKENS}
port: ${PORT}
bfcl_data: ${BFCL_DATA}
result_dir: ${RESULT_DIR}
log_file: ${LOG_FILE}
pid_file: ${PID_FILE}
EOF
}

start_check() {
  if is_running; then
    echo "Correctness check is already running with PID $(cat "$PID_FILE")."
    exit 0
  fi

  mkdir -p "$RESULT_DIR"
  print_config >"${RESULT_DIR}/launch_config.txt"

  local bfcl_args=()
  if [[ -n "$BFCL_DATA" ]]; then
    bfcl_args=(--bfcl-data "$BFCL_DATA")
  fi

  echo "Starting SlotOffload correctness check in background:"
  print_config

  env VLLM_USE_MODELSCOPE="$VLLM_USE_MODELSCOPE" \
    nohup .venv/bin/python benchmarks/slot_offload_correctness.py \
      --model "$MODEL" \
      --output-dir "$RESULT_DIR" \
      --port "$PORT" \
      --gpu-kv-bytes "$GPU_KV_BYTES" \
      --cpu-bytes "$CPU_BYTES_OVERRIDE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-tokens "$MAX_TOKENS" \
      --requests "$REQUESTS" \
      --groups "$WORKLOAD_GROUPS" \
      --seed "$SEED" \
      --scenario-set "$SCENARIO_SET" \
      --prompt-style "$PROMPT_STYLE" \
      --distribution "$DISTRIBUTION" \
      "${bfcl_args[@]}" \
      >"$LOG_FILE" 2>&1 &

  echo $! >"$PID_FILE"
  echo "Started PID $(cat "$PID_FILE")."
  echo "Use: bash $0 status"
  echo "Use: bash $0 tail"
  echo "Use: bash $0 summary"
}

status_check() {
  if is_running; then
    echo "status: running"
    echo "pid: $(cat "$PID_FILE")"
  else
    echo "status: stopped"
    if [[ -f "$PID_FILE" ]]; then
      echo "last_pid: $(cat "$PID_FILE")"
    fi
  fi
  echo "result_dir: ${RESULT_DIR}"
  echo "log_file: ${LOG_FILE}"
  if [[ -f "${RESULT_DIR}/correctness_summary.json" ]]; then
    echo "summary: ${RESULT_DIR}/correctness_summary.json"
  fi

  if [[ -f "$LOG_FILE" ]]; then
    echo
    echo "last progress:"
    grep -E "^(Starting|native:|binary:|value:)" "$LOG_FILE" | tail -n 8 || true
    echo
    echo "last log lines:"
    tail -n 20 "$LOG_FILE"
  fi
}

tail_log() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "Log file not found: ${LOG_FILE}"
    exit 1
  fi
  tail -f "$LOG_FILE"
}

stop_check() {
  if ! is_running; then
    echo "No running correctness check found."
    exit 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  echo "Stopping PID ${pid}."
  kill "$pid"
}

print_summary() {
  local summary="${RESULT_DIR}/correctness_summary.json"
  if [[ ! -f "$summary" ]]; then
    echo "Summary not found yet: ${summary}"
    exit 1
  fi
  cat "$summary"
}

case "$ACTION" in
  start) start_check ;;
  status) status_check ;;
  tail) tail_log ;;
  stop) stop_check ;;
  summary) print_summary ;;
  config) print_config ;;
  *)
    echo "Usage: $0 {start|status|tail|stop|summary|config}"
    exit 2
    ;;
esac
