#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/slime_opd}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
MAIN_PID_FILE="${MAIN_PID_FILE:-${LOG_DIR}/p1_tip_baselines_k16_seed1_aistation_20260513.pid}"
MAIN_LOG="${MAIN_LOG:-${LOG_DIR}/p1_tip_baselines_k16_seed1_aistation_20260513.log}"
TAG="${TAG:-p1_entropy_k16_seed1_aistation_20260513}"

mkdir -p "${LOG_DIR}"

while true; do
  if [[ -f "${MAIN_PID_FILE}" ]]; then
    main_pid="$(cat "${MAIN_PID_FILE}" || true)"
    if [[ -n "${main_pid}" ]] && kill -0 "${main_pid}" 2>/dev/null; then
      sleep 300
      continue
    fi
  fi
  break
done

{
  echo "===== entropy follow-up launch ====="
  date
  echo "main_log=${MAIN_LOG}"
  grep -E "PIPELINE_DONE|DONE mask=|START mask=" "${MAIN_LOG}" || true
} >> "${LOG_DIR}/${TAG}.watcher.log" 2>&1

cd "${SLIME_DIR}"
TAG="${TAG}" \
SEED_LABEL=seed1 \
RUN_LIST="entropy:0.03:310 entropy:0.05:320" \
SEED=1234 \
ROLLOUT_SEED=42 \
OPD_BUDGET_MASK_SEED=42 \
TEACHER_GPU=0 \
RAY_GPUS=1 \
EVAL_GPUS=0,1 \
ACTOR_NUM_GPUS_PER_NODE=1 \
ROLLOUT_NUM_GPUS=1 \
COLOCATE=1 \
ALLOW_GLOBAL_RAY_STOP=1 \
bash ./run_opd_budget_ratio_sweep_20260512.sh
