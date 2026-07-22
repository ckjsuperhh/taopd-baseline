#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/slime_opd}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
TAG="${TAG:-p1_tip_baselines_seed3_followup_20260514}"
PIPELINE_LOG="${LOG_DIR}/${TAG}.log"

STUDENT_HF="${STUDENT_HF:-/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B}"
TEACHER_MODEL="${TEACHER_MODEL:-/path/to/models/Qwen3/4B}"
COMMON_CONTEXT="${COMMON_CONTEXT:-/path/to/outputs/slime_opd/qwen3_1_7b_dapo_diag_k16_exact_20260510_084415/fixed_context/context_bank.parquet}"
BASELINE_METRICS="${BASELINE_METRICS:-/path/to/outputs/slime_opd/qwen3_1_7b_dapo_diag_k16_exact_20260510_084415/fixed_context/theta0_metrics.parquet}"

mkdir -p "${LOG_DIR}"

cd "${SLIME_DIR}"
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate verl

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PIPELINE_LOG}"
}

complete_random005() {
  local mask="random"
  local ratio="0.05"
  local ratio_tag="005"
  local run_name="qwen3_1_7b_dapo_budget_k16_ratio${ratio_tag}_${mask}_max64_seed3_${TAG}"
  local save_dir="${OUTPUT_ROOT}/${run_name}"
  local eval_dir="${OUTPUT_ROOT}/${TAG}/${mask}_ratio${ratio_tag}"

  if [[ -f "${eval_dir}/gain/q3_bootstrap_matching_summary.csv" ]]; then
    log "SKIP mask=${mask} ratio=${ratio}: gain summary already exists"
    return
  fi

  log "RESUME_EVAL_START mask=${mask} ratio=${ratio} run=${run_name}"
  mkdir -p "${eval_dir}"

  if [[ ! -f "${eval_dir}/fixed_context_metrics.parquet" ]]; then
    if [[ -d "${eval_dir}/student_hf" ]]; then
      CUDA_VISIBLE_DEVICES=0,1 python3 tools/eval_fixed_context_bank.py \
        --context-bank "${COMMON_CONTEXT}" \
        --student "${eval_dir}/student_hf" \
        --teacher "${TEACHER_MODEL}" \
        --output "${eval_dir}/fixed_context_metrics.parquet" \
        --student-device cuda:0 \
        --teacher-device cuda:1 \
        --dtype bfloat16 \
        --topk 16 \
        --max-response-tokens 256 \
        --baseline-metrics "${BASELINE_METRICS}" \
        2>&1 | tee -a "${PIPELINE_LOG}"
    else
      CUDA_VISIBLE_DEVICES=0,1 python3 tools/eval_fixed_context_from_megatron.py \
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
    fi
  fi

  python3 tools/analyze_fixed_context_gain.py \
    --input "${eval_dir}/fixed_context_metrics.parquet" \
    --output-dir "${eval_dir}/gain" \
    --bootstrap 1000 \
    --seed 20260512 \
    --use-quadrant \
    2>&1 | tee -a "${PIPELINE_LOG}"

  log "DONE mask=${mask} ratio=${ratio} save=${save_dir} eval=${eval_dir}"
}

log "RESUME_AFTER_REBOOT_START tag=${TAG}"
complete_random005

TAG="${TAG}" \
SEED_LABEL=seed3 \
RUN_LIST="ca_softor:0.03:670 ca_softor:0.05:680 q3:0.03:690 q3:0.05:700 entropy:0.03:710 entropy:0.05:720" \
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
