#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="/path/to/slime-main"
OUT_ROOT="/path/to/outputs/slime_opd"
LOG_DIR="${OUT_ROOT}/logs"
SEED1_TAG="within_q3_teachability_seed1_20260514"
SEED2_TAG="within_q3_teachability_seed2_20260514"
QUEUE_LOG="${LOG_DIR}/queue_${SEED2_TAG}_after_${SEED1_TAG}.log"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

seed1_done() {
  grep -q "PIPELINE_DONE tag=${SEED1_TAG}" "${LOG_DIR}/${SEED1_TAG}.log" 2>/dev/null
}

seed2_active() {
  pgrep -af "${SEED2_TAG}|launch_within_q3_teachability_seed2_20260514|qwen3_1_7b_dapo_budget_k16_ratio003_.*seed2_${SEED2_TAG}" >/dev/null 2>&1
}

seed2_done() {
  grep -q "PIPELINE_DONE tag=${SEED2_TAG}" "${LOG_DIR}/${SEED2_TAG}.log" 2>/dev/null
}

gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tee -a "${QUEUE_LOG}" || true
}

log "queue start: wait for ${SEED1_TAG}, then launch ${SEED2_TAG}"
while ! seed1_done; do
  log "waiting for ${SEED1_TAG}"
  tail -n 8 "${LOG_DIR}/${SEED1_TAG}.log" 2>/dev/null | tee -a "${QUEUE_LOG}" || true
  gpu_snapshot
  sleep 60
done

log "${SEED1_TAG} complete; refresh within-Q3 archive"
cd "${SLIME_DIR}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/path/to/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/path/to/hf_cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/path/to/hf_cache/datasets}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"
python3 tools/aggregate_within_q3_teachability_20260514.py 2>&1 | tee -a "${QUEUE_LOG}"

if seed2_done; then
  log "${SEED2_TAG} already complete; not launching"
  exit 0
fi

if seed2_active; then
  log "${SEED2_TAG} already active; not launching duplicate"
  exit 0
fi

log "launch ${SEED2_TAG}"
chmod +x ./launch_within_q3_teachability_seed2_20260514.sh
setsid bash ./launch_within_q3_teachability_seed2_20260514.sh \
  > "${LOG_DIR}/${SEED2_TAG}.nohup.log" 2>&1 < /dev/null &
echo $! > "${LOG_DIR}/${SEED2_TAG}.pid"
log "launched pid=$(cat "${LOG_DIR}/${SEED2_TAG}.pid")"
gpu_snapshot
