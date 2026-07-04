#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

ACTION="${1:-status}"
PROFILE="${PROFILE:-agent_core}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_TOKENS="${MAX_TOKENS:-1}"
GPU_KV_BYTES="${GPU_KV_BYTES:-12582912}"
CPU_BYTES_OVERRIDE="${CPU_BYTES_OVERRIDE:-$((GPU_KV_BYTES * 4))}"
RESUME="${RESUME:-1}"
VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-True}"

model_tag="${MODEL##*/}"
RESULT_ROOT="${RESULT_ROOT:-slot_${PROFILE}_${model_tag}_len${MAX_MODEL_LEN}}"
LOG_FILE="${LOG_FILE:-${RESULT_ROOT}.log}"
PID_FILE="${PID_FILE:-${RESULT_ROOT}.pid}"

expected_runs() {
  case "$PROFILE" in
    agent_smoke) echo 3 ;;
    agent_core) echo 75 ;;
    smoke) echo 3 ;;
    core) echo 15 ;;
    reuse_distance) echo 36 ;;
    ablation) echo 15 ;;
    sensitivity) echo 18 ;;
    paper) echo 165 ;;
    *) echo 0 ;;
  esac
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

completed_runs() {
  local summary_root="${RESULT_ROOT}/${PROFILE}"
  if [[ -d "$summary_root" ]]; then
    find "$summary_root" -name summary.json | wc -l
  else
    echo 0
  fi
}

print_config() {
  cat <<EOF
profile: ${PROFILE}
model: ${MODEL}
gpu_kv_bytes: ${GPU_KV_BYTES}
cpu_bytes: ${CPU_BYTES_OVERRIDE}
max_model_len: ${MAX_MODEL_LEN}
max_tokens: ${MAX_TOKENS}
result_root: ${RESULT_ROOT}
log_file: ${LOG_FILE}
pid_file: ${PID_FILE}
resume: ${RESUME}
EOF
}

start_experiments() {
  if is_running; then
    echo "Experiment is already running with PID $(cat "$PID_FILE")."
    exit 0
  fi

  mkdir -p "$RESULT_ROOT"
  print_config >"${RESULT_ROOT}/launch_config.txt"

  echo "Starting SlotOffload experiments in background:"
  print_config

  env \
    VLLM_USE_MODELSCOPE="$VLLM_USE_MODELSCOPE" \
    MODEL="$MODEL" \
    GPU_KV_BYTES="$GPU_KV_BYTES" \
    CPU_BYTES_OVERRIDE="$CPU_BYTES_OVERRIDE" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    MAX_TOKENS="$MAX_TOKENS" \
    RESULT_ROOT="$RESULT_ROOT" \
    RESUME="$RESUME" \
    nohup bash examples/features/run_slot_offload_experiment_suite.sh "$PROFILE" \
      >"$LOG_FILE" 2>&1 &

  echo $! >"$PID_FILE"
  echo "Started PID $(cat "$PID_FILE")."
  echo "Use: bash $0 status"
  echo "Use: bash $0 tail"
}

status_experiments() {
  local expected
  local completed
  expected="$(expected_runs)"
  completed="$(completed_runs)"

  if is_running; then
    echo "status: running"
    echo "pid: $(cat "$PID_FILE")"
  else
    echo "status: stopped"
    if [[ -f "$PID_FILE" ]]; then
      echo "last_pid: $(cat "$PID_FILE")"
    fi
  fi
  echo "completed: ${completed}/${expected}"
  echo "result_root: ${RESULT_ROOT}"
  echo "log_file: ${LOG_FILE}"

  if [[ -f "$LOG_FILE" ]]; then
    echo
    echo "last started run:"
    grep -E "^(Running|Starting) " "$LOG_FILE" | tail -n 5 || true
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

stop_experiments() {
  if ! is_running; then
    echo "No running experiment found."
    exit 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  echo "Stopping PID ${pid}."
  kill "$pid"
}

print_summary() {
  local summary="${RESULT_ROOT}/${PROFILE}/summary.md"
  if [[ ! -f "$summary" ]]; then
    echo "Summary not found yet: ${summary}"
    exit 1
  fi
  cat "$summary"
}

case "$ACTION" in
  start) start_experiments ;;
  status) status_experiments ;;
  tail) tail_log ;;
  stop) stop_experiments ;;
  summary) print_summary ;;
  config) print_config ;;
  *)
    echo "Usage: $0 {start|status|tail|stop|summary|config}"
    exit 2
    ;;
esac
