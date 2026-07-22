#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/path/to/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/path/to/hf_cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/path/to/hf_cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/path/to/hf_cache/transformers}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

OUT_ROOT="/path/to/outputs/slime_opd/storyline_20260513/downstream_smoke_20260515"
LOG_DIR="/path/to/outputs/slime_opd/logs"
RUN_DIR="${OUT_ROOT}/gsm8k_cot_full"
LOG_FILE="${LOG_DIR}/downstream_gsm8k_cot_full_20260515.log"
GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TIMEOUT="${TIMEOUT:-150m}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate lmeval

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

run_one() {
  local name="$1"
  local model="$2"
  local out_dir="${RUN_DIR}/${name}"
  local done_flag="${out_dir}/DONE"

  if [[ -f "${done_flag}" ]]; then
    log "SKIP name=${name} done=${done_flag}"
    return 0
  fi

  mkdir -p "${out_dir}"
  log "START name=${name} model=${model} gpu=${GPU}"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" timeout "${TIMEOUT}" lm-eval run \
    --model hf \
    --model_args "pretrained=${model},dtype=bfloat16,trust_remote_code=True" \
    --tasks gsm8k_cot \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${out_dir}" \
    2>&1 | tee -a "${LOG_FILE}"
  local ec=${PIPESTATUS[0]}
  set -e

  if [[ "${ec}" -eq 0 ]]; then
    date '+%F %T' > "${done_flag}"
    log "DONE name=${name} out=${out_dir}"
  else
    log "FAILED name=${name} exit=${ec} out=${out_dir}"
    return "${ec}"
  fi
}

log "QUEUE_START run_dir=${RUN_DIR}"
run_one "base_qwen3_1p7b" "/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B"
run_one "dlearn_high_ratio003_seed2" "/path/to/outputs/slime_opd/budget_common_context_k16_ratio003_seed2_20260511/dlearn_high_max64_seed2/student_hf"
run_one "q3_highc_ratio003_seed2" "/path/to/outputs/slime_opd/budget_common_context_k16_ratio003_seed2_20260511/q3_highc_max64_seed2/student_hf"
run_one "tip_ratio003_seed2" "/path/to/outputs/slime_opd/p1_tip_baselines_seed2_followup_20260513/tip_ratio003/student_hf"
log "QUEUE_DONE run_dir=${RUN_DIR}"
