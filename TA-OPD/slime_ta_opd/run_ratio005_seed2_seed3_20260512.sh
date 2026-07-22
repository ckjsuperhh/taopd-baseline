#!/usr/bin/env bash
set -euo pipefail

cd /path/to/slime-main
chmod +x ./run_opd_budget_ratio_sweep_20260512.sh

LOG_DIR="/path/to/outputs/slime_opd/logs"
mkdir -p "${LOG_DIR}"
MASTER_TAG="ratio005_dlearn_vs_q3_seed2_seed3_20260512"
MASTER_LOG="${LOG_DIR}/${MASTER_TAG}.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MASTER_LOG}"
}

run_seed() {
  local label="$1"
  local seed="$2"
  local rollout_seed="$3"
  local mask_seed="$4"
  local tag="budget_ratio005_compare_k16_${label}_20260512"

  log "START ${label} seed=${seed} rollout_seed=${rollout_seed} mask_seed=${mask_seed}"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader | tee -a "${MASTER_LOG}"

  TAG="${tag}" \
  RUN_LIST="dlearn_high:0.05:20 q3_highc:0.05:50" \
  SEED="${seed}" \
  ROLLOUT_SEED="${rollout_seed}" \
  OPD_BUDGET_MASK_SEED="${mask_seed}" \
  ./run_opd_budget_ratio_sweep_20260512.sh 2>&1 | tee -a "${MASTER_LOG}"

  log "DONE ${label}"
}

log "MASTER_START ${MASTER_TAG}"
run_seed seed2 2345 43 43
run_seed seed3 3456 44 44
log "MASTER_DONE ${MASTER_TAG}"
