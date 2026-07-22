#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/path/to/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/path/to/hf_cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/path/to/hf_cache/datasets}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

SLIME_DIR="/path/to/slime-main"
OUT_ROOT="/path/to/outputs/slime_opd"
LOG_DIR="${OUT_ROOT}/logs"
TAG="within_q3_teachability_seed2_20260514"
PIPELINE_LOG="${LOG_DIR}/${TAG}.log"
VERL_BIN="/path/to/miniconda3/envs/verl/bin"
PYTHON="${VERL_BIN}/python3"
STUDENT_HF="/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B"
TEACHER_MODEL="/path/to/models/Qwen3/4B"
COMMON_CONTEXT="${OUT_ROOT}/qwen3_1_7b_dapo_diag_k16_exact_20260510_084415/fixed_context/context_bank.parquet"
BASELINE_METRICS="${OUT_ROOT}/qwen3_1_7b_dapo_diag_k16_exact_20260510_084415/fixed_context/theta0_metrics.parquet"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${PIPELINE_LOG}"
}

cd "${SLIME_DIR}"
export PATH="${VERL_BIN}:${PATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

mask="q3_teachability_high"
ratio_tag="ratio003"
save_dir="${OUT_ROOT}/qwen3_1_7b_dapo_budget_k16_${ratio_tag}_${mask}_max64_seed2_${TAG}"
eval_dir="${OUT_ROOT}/${TAG}/${mask}_${ratio_tag}"
gain_summary="${eval_dir}/gain/q3_bootstrap_matching_summary.csv"

if [[ ! -s "${gain_summary}" ]]; then
  log "RESUME_EVAL_START mask=${mask} ratio=0.03 after stalled eval"
  CUDA_VISIBLE_DEVICES="0,1" timeout 60m "${PYTHON}" tools/eval_fixed_context_from_megatron.py \
    --checkpoint-root "${save_dir}" \
    --iteration latest \
    --origin-hf-dir "${STUDENT_HF}" \
    --teacher-hf-dir "${TEACHER_MODEL}" \
    --context-bank "${COMMON_CONTEXT}" \
    --baseline-metrics "${BASELINE_METRICS}" \
    --output-dir "${eval_dir}" \
    --student-device cuda:0 \
    --teacher-device cuda:1 \
    --dtype bfloat16 \
    --topk 16 \
    --force-convert \
    2>&1 | tee -a "${PIPELINE_LOG}"

  "${PYTHON}" tools/analyze_fixed_context_gain.py \
    --input "${eval_dir}/fixed_context_metrics.parquet" \
    --output-dir "${eval_dir}/gain" \
    --bootstrap 1000 \
    --seed 20260512 \
    --use-quadrant \
    2>&1 | tee -a "${PIPELINE_LOG}"
  log "RESUME_EVAL_DONE mask=${mask} ratio=0.03"
else
  log "RESUME_EVAL_SKIP mask=${mask} ratio=0.03 existing=${gain_summary}"
fi

log "RESUME_REMAINING_START seed2 low/dincompat"
TAG="${TAG}" \
SEED_LABEL="seed2" \
RUN_LIST="q3_teachability_low:0.03:850 q3_dincompat_high:0.03:860" \
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

log "RESUME_AGGREGATE_START"
"${PYTHON}" tools/aggregate_within_q3_teachability_20260514.py 2>&1 | tee -a "${PIPELINE_LOG}"
log "RESUME_DONE tag=${TAG}"
