#!/usr/bin/env bash
set -euo pipefail

cd /path/to/slime-main

LOG_DIR="/path/to/outputs/slime_opd/logs"
mkdir -p "${LOG_DIR}"

MASTER_TAG="budget_dlearn_curve_ratio001_010_seed2_seed3_a800_20260512"
MASTER_LOG="${LOG_DIR}/${MASTER_TAG}.log"

chmod +x ./run_opd_budget_ratio_sweep_20260512.sh

setsid bash -lc '
set -euo pipefail
cd /path/to/slime-main
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate verl

run_seed() {
  local seed_label="$1"
  local seed="$2"
  local rollout_seed="$3"
  local mask_seed="$4"
  local tag="budget_dlearn_curve_ratio001_010_${seed_label}_a800_20260512"

  echo "[$(date "+%F %T")] START ${seed_label} seed=${seed} rollout_seed=${rollout_seed} mask_seed=${mask_seed}"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

  TAG="${tag}" \
  SEED_LABEL="${seed_label}" \
  RUN_LIST="dlearn_high:0.01:60 dlearn_high:0.10:70" \
  TEACHER_GPU=0 \
  RAY_GPUS=1 \
  EVAL_GPUS=0,1 \
  ACTOR_NUM_GPUS_PER_NODE=1 \
  ROLLOUT_NUM_GPUS=1 \
  COLOCATE=1 \
  ALLOW_GLOBAL_RAY_STOP=0 \
  SEED="${seed}" \
  ROLLOUT_SEED="${rollout_seed}" \
  OPD_BUDGET_MASK_SEED="${mask_seed}" \
  ./run_opd_budget_ratio_sweep_20260512.sh

  echo "[$(date "+%F %T")] DONE ${seed_label}"
}

run_seed seed2 2345 43 43
run_seed seed3 3456 44 44

echo "[$(date "+%F %T")] MASTER_DONE"
' > "${MASTER_LOG}" 2>&1 < /dev/null &

pid="$!"
echo "${pid}" > "${LOG_DIR}/${MASTER_TAG}.pid"
echo "LAUNCHED pid=${pid} log=${MASTER_LOG}"
