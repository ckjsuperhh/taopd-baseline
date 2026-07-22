#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/path/to/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/path/to/hf_cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/path/to/hf_cache/datasets}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
OUT_ROOT="${OUT_ROOT:-/path/to/outputs/slime_opd}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs}"
PYTHON="${PYTHON:-python3}"
STUDENT_HF="${STUDENT_HF:-/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B}"
TEACHER_MODEL="${TEACHER_MODEL:-/path/to/models/Qwen3/4B}"
COMMON_CONTEXT="${COMMON_CONTEXT:-${OUT_ROOT}/qwen3_1_7b_dapo_diag_k16_exact_20260510_084415/fixed_context/context_bank.parquet}"
BASELINE_METRICS="${BASELINE_METRICS:-${OUT_ROOT}/qwen3_1_7b_dapo_diag_k16_exact_20260510_084415/fixed_context/theta0_metrics.parquet}"
EVAL_GPUS="${EVAL_GPUS:-0,1}"
STUDENT_DEVICE="${STUDENT_DEVICE:-cuda:0}"
TEACHER_DEVICE="${TEACHER_DEVICE:-cuda:1}"
TIMEOUT="${TIMEOUT:-60m}"
RUN_LOG="${LOG_DIR}/within_q3_missing_eval_20260514.log"

mkdir -p "${LOG_DIR}"
cd "${SLIME_DIR}"
export PATH="/path/to/miniconda3/envs/verl/bin:${PATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${RUN_LOG}"
}

run_eval_one() {
  local seed_label="$1"
  local mask="$2"
  local tag="within_q3_teachability_${seed_label}_20260514"
  local ratio_tag="ratio003"
  local save_dir="${OUT_ROOT}/qwen3_1_7b_dapo_budget_k16_${ratio_tag}_${mask}_max64_${seed_label}_${tag}"
  local eval_dir="${OUT_ROOT}/${tag}/${mask}_${ratio_tag}"
  local gain_summary="${eval_dir}/gain/q3_bootstrap_matching_summary.csv"
  local eval_log="${LOG_DIR}/within_q3_eval_${mask}_${seed_label}_full_20260514.log"

  if [[ -s "${gain_summary}" ]]; then
    log "SKIP seed=${seed_label} mask=${mask}: gain exists at ${gain_summary}"
    return 0
  fi

  if [[ ! -d "${save_dir}" ]]; then
    log "MISSING_SAVE_DIR seed=${seed_label} mask=${mask}: ${save_dir}"
    return 0
  fi

  mkdir -p "${eval_dir}"
  log "EVAL_START seed=${seed_label} mask=${mask} save=${save_dir} eval=${eval_dir}"

  set +e
  CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" timeout "${TIMEOUT}" "${PYTHON}" tools/eval_fixed_context_from_megatron.py \
    --checkpoint-root "${save_dir}" \
    --iteration latest \
    --origin-hf-dir "${STUDENT_HF}" \
    --teacher-hf-dir "${TEACHER_MODEL}" \
    --context-bank "${COMMON_CONTEXT}" \
    --baseline-metrics "${BASELINE_METRICS}" \
    --output-dir "${eval_dir}" \
    --student-device "${STUDENT_DEVICE}" \
    --teacher-device "${TEACHER_DEVICE}" \
    --dtype bfloat16 \
    --topk 16 \
    --force-convert \
    2>&1 | tee -a "${eval_log}"
  local eval_ec=${PIPESTATUS[0]}
  set -e

  if [[ "${eval_ec}" -ne 0 ]]; then
    log "EVAL_FAILED seed=${seed_label} mask=${mask} exit=${eval_ec} log=${eval_log}"
    return 0
  fi

  log "GAIN_START seed=${seed_label} mask=${mask}"
  "${PYTHON}" tools/analyze_fixed_context_gain.py \
    --input "${eval_dir}/fixed_context_metrics.parquet" \
    --output-dir "${eval_dir}/gain" \
    --bootstrap 1000 \
    --seed 20260512 \
    --use-quadrant \
    2>&1 | tee -a "${eval_log}"
  log "DONE seed=${seed_label} mask=${mask} gain=${gain_summary}"

  "${PYTHON}" tools/aggregate_within_q3_teachability_20260514.py 2>&1 | tee -a "${RUN_LOG}"
}

log "QUEUE_START eval_gpus=${EVAL_GPUS} student_device=${STUDENT_DEVICE} teacher_device=${TEACHER_DEVICE} timeout=${TIMEOUT}"
run_eval_one seed3 q3_teachability_low
run_eval_one seed3 q3_dincompat_high
run_eval_one seed2 q3_teachability_high
run_eval_one seed2 q3_teachability_low
run_eval_one seed2 q3_dincompat_high
"${PYTHON}" tools/aggregate_within_q3_teachability_20260514.py 2>&1 | tee -a "${RUN_LOG}"
log "QUEUE_DONE"
