#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/path/to/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/path/to/hf_cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/path/to/hf_cache/datasets}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

cd /path/to/slime-main

TAG=within_q3_teachability_seed3_20260514 \
SEED_LABEL=seed3 \
RUN_LIST="q3_teachability_high:0.03:870 q3_teachability_low:0.03:880 q3_dincompat_high:0.03:890" \
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
