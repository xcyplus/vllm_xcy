#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

PROFILE="${1:-smoke}"
CALIBRATION_FILE="${CALIBRATION_FILE:-slot_gpu_kv_calibration_modelscope/calibrated_gpu_kv_bytes.sh}"
RESULT_PREFIX="${RESULT_PREFIX:-slot_scale}"
CPU_CACHE_MULTIPLIER="${CPU_CACHE_MULTIPLIER:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_TOKENS="${MAX_TOKENS:-1}"
RESUME="${RESUME:-1}"

MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

case "$PROFILE" in
  smoke|agent_smoke|core|agent_core|reuse_distance|ablation|sensitivity|paper|all) ;;
  *)
    echo "Usage: $0 {smoke|agent_smoke|core|agent_core|reuse_distance|ablation|sensitivity|paper|all}"
    exit 2
    ;;
esac

if [[ ! -f "$CALIBRATION_FILE" ]]; then
  echo "Calibration file not found: ${CALIBRATION_FILE}"
  exit 1
fi

# A failed final model can leave the generated associative array unclosed.
if [[ "$(tail -n 1 "$CALIBRATION_FILE")" != ")" ]]; then
  printf ')\n' >>"$CALIBRATION_FILE"
fi

# shellcheck disable=SC1090
source "$CALIBRATION_FILE"

if ! declare -p GPU_KV_BYTES_BY_MODEL >/dev/null 2>&1; then
  echo "GPU_KV_BYTES_BY_MODEL is missing from ${CALIBRATION_FILE}"
  exit 1
fi

export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-True}"
export MAX_MODEL_LEN MAX_TOKENS RESUME

for model_name in "${MODELS[@]}"; do
  gpu_kv_bytes="${GPU_KV_BYTES_BY_MODEL[$model_name]:-}"
  if [[ -z "$gpu_kv_bytes" ]]; then
    echo "Missing calibration result for ${model_name}"
    exit 1
  fi
  if (( gpu_kv_bytes <= 0 )); then
    echo "Invalid GPU KV cache size for ${model_name}: ${gpu_kv_bytes}"
    exit 1
  fi

  model_tag="${model_name##*/}"
  cpu_kv_bytes=$((gpu_kv_bytes * CPU_CACHE_MULTIPLIER))
  result_root="${RESULT_PREFIX}_${PROFILE}_${model_tag}"

  echo
  echo "Running ${PROFILE}: ${model_name}"
  echo "  GPU KV cache bytes: ${gpu_kv_bytes}"
  echo "  CPU KV cache bytes: ${cpu_kv_bytes}"
  echo "  Result root: ${result_root}"

  MODEL="$model_name" \
  GPU_KV_BYTES="$gpu_kv_bytes" \
  CPU_BYTES_OVERRIDE="$cpu_kv_bytes" \
  RESULT_ROOT="$result_root" \
    bash examples/features/run_slot_offload_experiment_suite.sh "$PROFILE"
done

echo
echo "Model-scale ${PROFILE} experiments completed."
for model_name in "${MODELS[@]}"; do
  model_tag="${model_name##*/}"
  summary="${RESULT_PREFIX}_${PROFILE}_${model_tag}/${PROFILE}/summary.md"
  if [[ -f "$summary" ]]; then
    echo "  ${summary}"
  fi
done
