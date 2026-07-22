#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/slime_opd}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
TAG="${TAG:-p1_tip_baselines_seed3_followup_20260514}"

mkdir -p "${LOG_DIR}"

{
  echo "===== P1 TIP-family seed3 follow-up launch ====="
  date
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
} >> "${LOG_DIR}/${TAG}.launcher.log" 2>&1

cd "${SLIME_DIR}"

TAG="${TAG}" \
SEED_LABEL=seed3 \
RUN_LIST="divergence:0.03:610 divergence:0.05:620 tip:0.03:630 tip:0.05:640 random:0.03:650 random:0.05:660 ca_softor:0.03:670 ca_softor:0.05:680 q3:0.03:690 q3:0.05:700 entropy:0.03:710 entropy:0.05:720" \
SEED=1236 \
ROLLOUT_SEED=44 \
OPD_BUDGET_MASK_SEED=44 \
TEACHER_GPU=0 \
RAY_GPUS=1 \
EVAL_GPUS=0,1 \
ACTOR_NUM_GPUS_PER_NODE=1 \
ROLLOUT_NUM_GPUS=1 \
COLOCATE=1 \
ALLOW_GLOBAL_RAY_STOP=1 \
bash ./run_opd_budget_ratio_sweep_20260512.sh
