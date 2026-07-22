#!/usr/bin/env bash
set -euo pipefail

cd /path/to/slime-main

MASK="${MASK:?MASK is required}"
RATIO="${RATIO:?RATIO is required}"
IDX="${IDX:?IDX is required}"
SEED_LABEL="${SEED_LABEL:?SEED_LABEL is required}"
SEED="${SEED:?SEED is required}"
ROLLOUT_SEED="${ROLLOUT_SEED:?ROLLOUT_SEED is required}"
OPD_BUDGET_MASK_SEED="${OPD_BUDGET_MASK_SEED:?OPD_BUDGET_MASK_SEED is required}"
TAG="${TAG:?TAG is required}"

LOG_DIR="/path/to/outputs/slime_opd/logs"
mkdir -p "${LOG_DIR}"
LAUNCH_LOG="${LOG_DIR}/${TAG}_${MASK}_ratio${RATIO/./}_${SEED_LABEL}.setsid.log"

chmod +x ./run_opd_budget_ratio_sweep_20260512.sh

setsid env \
  TAG="${TAG}" \
  SEED_LABEL="${SEED_LABEL}" \
  RUN_LIST="${MASK}:${RATIO}:${IDX}" \
  SEED="${SEED}" \
  ROLLOUT_SEED="${ROLLOUT_SEED}" \
  OPD_BUDGET_MASK_SEED="${OPD_BUDGET_MASK_SEED}" \
  ./run_opd_budget_ratio_sweep_20260512.sh \
  > "${LAUNCH_LOG}" 2>&1 < /dev/null &

pid="$!"
echo "${pid}" > "${LOG_DIR}/${TAG}_${MASK}_ratio${RATIO/./}_${SEED_LABEL}.pid"
echo "LAUNCHED pid=${pid} log=${LAUNCH_LOG}"
