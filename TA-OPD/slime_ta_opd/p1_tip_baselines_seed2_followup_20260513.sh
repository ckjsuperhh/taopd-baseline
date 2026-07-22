#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/slime_opd}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
TAG="${TAG:-p1_tip_baselines_seed2_followup_20260513}"

mkdir -p "${LOG_DIR}"

{
  echo "===== P1 TIP-family seed2 follow-up launch ====="
  date
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
} >> "${LOG_DIR}/${TAG}.launcher.log" 2>&1

cd "${SLIME_DIR}"

TAG="${TAG}" \
SEED_LABEL=seed2 \
RUN_LIST="divergence:0.03:410 divergence:0.05:420 tip:0.03:430 tip:0.05:440 random:0.03:450 random:0.05:460 ca_softor:0.03:470 ca_softor:0.05:480 q3:0.03:490 q3:0.05:500 entropy:0.03:510 entropy:0.05:520" \
SEED=1235 \
ROLLOUT_SEED=43 \
OPD_BUDGET_MASK_SEED=43 \
TEACHER_GPU=0 \
RAY_GPUS=1 \
EVAL_GPUS=0,1 \
ACTOR_NUM_GPUS_PER_NODE=1 \
ROLLOUT_NUM_GPUS=1 \
COLOCATE=1 \
ALLOW_GLOBAL_RAY_STOP=1 \
bash ./run_opd_budget_ratio_sweep_20260512.sh
