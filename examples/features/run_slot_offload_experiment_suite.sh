#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

PROFILE="${1:-smoke}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
RESULT_ROOT="${RESULT_ROOT:-slot_experiments}"
GPU_KV_BYTES="${GPU_KV_BYTES:-16777216}"
CPU_BYTES_OVERRIDE="${CPU_BYTES_OVERRIDE:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_TOKENS="${MAX_TOKENS:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
PROMPT_STYLE="${PROMPT_STYLE:-}"
BFCL_DATA="${BFCL_DATA:-}"
MIN_ACCESSES="${MIN_ACCESSES:-1}"
MAX_SLOT_RATIO="${MAX_SLOT_RATIO:-0.5}"
BASE_THRESHOLD="${BASE_THRESHOLD:-0.2}"
PRESSURE_SCALE="${PRESSURE_SCALE:-0.6}"
STORE_THRESHOLD="${STORE_THRESHOLD:-2}"
RESUME="${RESUME:-1}"

if [[ "$PROFILE" == "all" ]]; then
  for child_profile in paper reuse_distance ablation sensitivity; do
    bash "$0" "$child_profile"
  done
  exit 0
fi

case "$PROFILE" in
  smoke)
    PROMPT_STYLE="${PROMPT_STYLE:-structured}"
    STRATEGIES=(native binary value)
    SEEDS=(2026)
    WORKLOADS=("finance one_hit 64 1 64")
    ;;
  agent_smoke)
    PROMPT_STYLE="${PROMPT_STYLE:-agent_mcp}"
    STRATEGIES=(native binary value)
    SEEDS=(2026)
    WORKLOADS=("agent_mcp zipf 96 1 67108864")
    ;;
  bfcl_smoke)
    PROMPT_STYLE="${PROMPT_STYLE:-bfcl}"
    STRATEGIES=(native binary value)
    SEEDS=(2026)
    WORKLOADS=("bfcl zipf 96 1 67108864")
    ;;
  bfcl_core)
    PROMPT_STYLE="${PROMPT_STYLE:-bfcl}"
    STRATEGIES=(gpu_only native threshold binary value)
    SEEDS=(2026 2027 2028)
    WORKLOADS=(
      "bfcl uniform 256 1 67108864"
      "bfcl zipf 256 1 67108864"
      "bfcl reuse_long 256 1 67108864"
    )
    ;;
  core)
    PROMPT_STYLE="${PROMPT_STYLE:-structured}"
    STRATEGIES=(gpu_only native threshold binary value)
    SEEDS=(2026)
    WORKLOADS=(
      "finance uniform 256 1 67108864"
      "finance one_hit 256 1 67108864"
      "finance zipf 256 1 67108864"
    )
    ;;
  agent_core)
    PROMPT_STYLE="${PROMPT_STYLE:-agent_mcp}"
    STRATEGIES=(gpu_only native threshold binary value)
    SEEDS=(2026 2027 2028)
    WORKLOADS=(
      "agent_mcp uniform 512 1 67108864"
      "agent_mcp one_hit 512 1 67108864"
      "agent_mcp zipf 512 1 67108864"
      "agent_mcp reuse_short 512 1 67108864"
      "agent_mcp reuse_long 512 1 67108864"
    )
    ;;
  ablation)
    PROMPT_STYLE="${PROMPT_STYLE:-structured}"
    STRATEGIES=(binary value_structure value_hotness value_cost value)
    SEEDS=(2026 2027 2028)
    WORKLOADS=("finance one_hit 1000 1 33554432")
    ;;
  sensitivity)
    PROMPT_STYLE="${PROMPT_STYLE:-structured}"
    STRATEGIES=(value_min1 value_min2 value_min3)
    SEEDS=(2026 2027 2028)
    WORKLOADS=(
      "finance uniform 1000 1 67108864"
      "finance one_hit 1000 1 33554432"
    )
    ;;
  reuse_distance)
    PROMPT_STYLE="${PROMPT_STYLE:-structured}"
    STRATEGIES=(native threshold binary value)
    SEEDS=(2026 2027 2028)
    WORKLOADS=(
      "finance reuse_short 1000 1 33554432"
      "finance reuse_medium 1000 1 33554432"
      "finance reuse_long 1000 1 33554432"
    )
    ;;
  paper)
    PROMPT_STYLE="${PROMPT_STYLE:-structured}"
    STRATEGIES=(gpu_only native threshold binary value)
    SEEDS=(2026 2027 2028)
    WORKLOADS=(
      "finance uniform 1000 1 67108864"
      "finance one_hit 1000 1 67108864"
      "finance zipf 1000 1 67108864"
      "finance one_hit 1000 1 16777216"
      "finance one_hit 1000 1 33554432"
      "finance one_hit 1000 1 134217728"
      "finance zipf 1000 4 67108864"
      "finance zipf 1000 8 67108864"
      "finance zipf 1000 16 67108864"
      "operations zipf 1000 1 67108864"
      "mixed zipf 1000 1 67108864"
    )
    ;;
  *)
    echo "Usage: $0 {smoke|agent_smoke|bfcl_smoke|core|agent_core|bfcl_core|reuse_distance|ablation|sensitivity|paper|all}"
    exit 2
    ;;
esac

if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "Port ${PORT} already has a healthy vLLM server. Stop it first."
  exit 1
fi

mkdir -p "$RESULT_ROOT"
ACTIVE_SERVER_PID=""
CURRENT_GPU_CACHE_TOKENS=0

stop_server() {
  if [[ -n "$ACTIVE_SERVER_PID" ]] \
    && kill -0 "$ACTIVE_SERVER_PID" >/dev/null 2>&1; then
    kill "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
    wait "$ACTIVE_SERVER_PID" 2>/dev/null || true
  fi
  ACTIVE_SERVER_PID=""
}
trap stop_server EXIT INT TERM

groups_for_set() {
  case "$1" in
    finance) echo 8 ;;
    operations) echo 7 ;;
    agent_mcp) echo 8 ;;
    bfcl) echo "${BFCL_GROUPS:-8}" ;;
    mixed) echo 16 ;;
    *) return 1 ;;
  esac
}

build_kv_config() {
  local strategy="$1"
  local cpu_bytes="$2"
  case "$strategy" in
    native)
      cat <<JSON
{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":${cpu_bytes},"offload_prompt_only":true}}
JSON
      ;;
    threshold)
      cat <<JSON
{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":${cpu_bytes},"offload_prompt_only":true,"store_threshold":${STORE_THRESHOLD}}}
JSON
      ;;
    binary)
      cat <<JSON
{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":${cpu_bytes},"offload_prompt_only":true,"slot_offload_policy":"prefer_template","slot_offload_store_types":["instruction","schema","context","system","unknown"]}}
JSON
      ;;
    value)
      cat <<JSON
{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":${cpu_bytes},"offload_prompt_only":true,"slot_offload_policy":"value_aware","slot_offload_min_accesses":${MIN_ACCESSES},"slot_offload_max_slot_ratio":${MAX_SLOT_RATIO},"slot_offload_base_threshold":${BASE_THRESHOLD},"slot_offload_pressure_scale":${PRESSURE_SCALE}}}
JSON
      ;;
    value_structure)
      cat <<JSON
{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":${cpu_bytes},"offload_prompt_only":true,"slot_offload_policy":"value_aware","slot_offload_min_accesses":2,"slot_offload_structure_weight":1.0,"slot_offload_hotness_weight":0.0,"slot_offload_cost_weight":0.0,"slot_offload_pressure_scale":0.0}}
JSON
      ;;
    value_hotness)
      cat <<JSON
{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":${cpu_bytes},"offload_prompt_only":true,"slot_offload_policy":"value_aware","slot_offload_min_accesses":2,"slot_offload_structure_weight":0.625,"slot_offload_hotness_weight":0.375,"slot_offload_cost_weight":0.0,"slot_offload_pressure_scale":0.0}}
JSON
      ;;
    value_cost)
      cat <<JSON
{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":${cpu_bytes},"offload_prompt_only":true,"slot_offload_policy":"value_aware","slot_offload_min_accesses":2,"slot_offload_structure_weight":0.5,"slot_offload_hotness_weight":0.3,"slot_offload_cost_weight":0.2,"slot_offload_pressure_scale":0.0}}
JSON
      ;;
    value_min1|value_min2|value_min3)
      local variant_min="${strategy#value_min}"
      cat <<JSON
{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":${cpu_bytes},"offload_prompt_only":true,"slot_offload_policy":"value_aware","slot_offload_min_accesses":${variant_min},"slot_offload_max_slot_ratio":${MAX_SLOT_RATIO},"slot_offload_base_threshold":${BASE_THRESHOLD},"slot_offload_pressure_scale":${PRESSURE_SCALE}}}
JSON
      ;;
    gpu_only) ;;
    *) return 1 ;;
  esac
}

start_server() {
  local strategy="$1"
  local cpu_bytes="$2"
  local server_log="$3"
  local command=(
    .venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL"
    --port "$PORT"
    --enforce-eager
    --generation-config vllm
    --enable-prefix-caching
    --max-model-len "$MAX_MODEL_LEN"
    --kv-cache-memory-bytes "$GPU_KV_BYTES"
  )
  if [[ "$strategy" != "gpu_only" ]]; then
    local kv_config
    kv_config="$(build_kv_config "$strategy" "$cpu_bytes")"
    command+=(--kv-transfer-config "$kv_config")
  fi

  "${command[@]}" >"$server_log" 2>&1 &
  ACTIVE_SERVER_PID=$!
  for _ in $(seq 1 300); do
    if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
      local cache_line
      cache_line="$(
        grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$server_log" \
          | tail -n 1 || true
      )"
      if [[ -z "$cache_line" ]]; then
        echo "Could not read GPU KV token capacity from ${server_log}"
        return 1
      fi
      CURRENT_GPU_CACHE_TOKENS="$(
        sed -E 's/.*size: ([0-9,]+) tokens/\1/' <<<"$cache_line" | tr -d ','
      )"
      return 0
    fi
    if ! kill -0 "$ACTIVE_SERVER_PID" >/dev/null 2>&1; then
      echo "vLLM exited during startup. See ${server_log}"
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for vLLM. See ${server_log}"
  return 1
}

run_one() {
  local scenario_set="$1"
  local distribution="$2"
  local requests="$3"
  local concurrency="$4"
  local cpu_bytes="$5"
  local seed="$6"
  local strategy="$7"
  if [[ -n "$CPU_BYTES_OVERRIDE" ]]; then
    cpu_bytes="$CPU_BYTES_OVERRIDE"
  fi
  local groups
  groups="$(groups_for_set "$scenario_set")"
  local effective_min_accesses="$MIN_ACCESSES"
  case "$strategy" in
    value_structure|value_hotness|value_cost) effective_min_accesses=2 ;;
    value_min1|value_min2|value_min3)
      effective_min_accesses="${strategy#value_min}"
      ;;
  esac
  local workload_id
  workload_id="${scenario_set}-${distribution}-r${requests}-c${concurrency}-cpu${cpu_bytes}"
  local result_dir="${RESULT_ROOT}/${PROFILE}/${workload_id}/seed-${seed}/${strategy}"

  if [[ "$RESUME" == "1" && -f "${result_dir}/summary.json" ]]; then
    echo "Skipping completed run: ${result_dir}"
    return
  fi

  mkdir -p "$result_dir"
  cat >"${result_dir}/run_config.json" <<JSON
{
  "profile": "${PROFILE}",
  "strategy": "${strategy}",
  "model": "${MODEL}",
  "scenario_set": "${scenario_set}",
  "prompt_style": "${PROMPT_STYLE}",
  "distribution": "${distribution}",
  "requests": ${requests},
  "concurrency": ${concurrency},
  "groups": ${groups},
  "seed": ${seed},
  "gpu_kv_bytes": ${GPU_KV_BYTES},
  "cpu_bytes": ${cpu_bytes},
  "max_tokens": ${MAX_TOKENS},
  "min_accesses": ${effective_min_accesses}
}
JSON

  echo "Starting ${strategy}: ${workload_id}, seed=${seed}"
  start_server "$strategy" "$cpu_bytes" "${result_dir}/server.log"
  local cpu_capacity_tokens=0
  local bfcl_args=()
  if [[ "$strategy" != "gpu_only" ]]; then
    cpu_capacity_tokens=$((
      cpu_bytes * CURRENT_GPU_CACHE_TOKENS / GPU_KV_BYTES
    ))
  fi
  if [[ -n "$BFCL_DATA" ]]; then
    bfcl_args=(--bfcl-data "$BFCL_DATA")
  fi
  .venv/bin/python benchmarks/slot_offload_benchmark.py \
    --endpoint "http://localhost:${PORT}" \
    --model "$MODEL" \
    --mode "$strategy" \
    --scenario-set "$scenario_set" \
    --prompt-style "$PROMPT_STYLE" \
    --distribution "$distribution" \
    --groups "$groups" \
    --requests "$requests" \
    --repeats 1 \
    --concurrency "$concurrency" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --offload-block-size 16 \
    --gpu-capacity-tokens "$CURRENT_GPU_CACHE_TOKENS" \
    --cpu-capacity-tokens "$cpu_capacity_tokens" \
    --max-tokens "$MAX_TOKENS" \
    --seed "$seed" \
    --output-dir "$result_dir" \
    "${bfcl_args[@]}" \
    2>&1 | tee "${result_dir}/client.log"
  stop_server
}

for workload in "${WORKLOADS[@]}"; do
  read -r scenario_set distribution requests concurrency cpu_bytes <<<"$workload"
  for seed in "${SEEDS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
      run_one \
        "$scenario_set" "$distribution" "$requests" "$concurrency" \
        "$cpu_bytes" "$seed" "$strategy"
    done
  done
done

.venv/bin/python benchmarks/summarize_slot_offload_experiments.py \
  --result-root "${RESULT_ROOT}/${PROFILE}"

echo "Experiment suite completed: ${RESULT_ROOT}/${PROFILE}"
