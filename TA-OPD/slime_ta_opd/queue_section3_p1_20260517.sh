#!/usr/bin/env bash
set -uo pipefail

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/slime_opd}"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
cd "${SLIME_DIR}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/path/to/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/path/to/hf_cache/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/path/to/hf_cache/hub}"

log_step() {
  echo
  echo "===== $(date '+%F %T') :: $* ====="
}

run_step() {
  local name="$1"
  shift
  log_step "${name}"
  "$@"
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    echo "STEP_FAILED name=${name} rc=${rc}"
  else
    echo "STEP_OK name=${name}"
  fi
  return "${rc}"
}

log_step "Initial GPU state"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

run_step "14B->1.7B retopk K=8/32" bash "${SLIME_DIR}/run_14b_to_1p7b_retopk_8_32_20260517.sh"

TORCH4=/path/to/models/Qwen3/4B/Qwen3-4B_torch_dist
if [[ ! -f "${TORCH4}/latest_checkpointed_iteration.txt" ]]; then
  log_step "Converting Qwen3-4B HF -> torch_dist"
  if [[ -f /path/to/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /path/to/miniconda3/etc/profile.d/conda.sh
    conda activate opsd
  fi
  NPROC_PER_NODE=1 MASTER_PORT=29644 bash "${SLIME_DIR}/convert_qwen3_4b_to_torch_dist.sh"
  CONVERT_RC=$?
  if [[ "${CONVERT_RC}" -ne 0 ]]; then
    echo "STEP_FAILED name=convert_qwen3_4b rc=${CONVERT_RC}"
  else
    echo "STEP_OK name=convert_qwen3_4b"
  fi
else
  echo "Qwen3-4B torch_dist already exists at ${TORCH4}"
fi

if [[ -f "${TORCH4}/latest_checkpointed_iteration.txt" ]]; then
  run_step "8B->4B diagnostic" bash "${SLIME_DIR}/run_8b_to_4b_fixed_context_300_20260517.sh"
  run_step "14B->4B diagnostic" bash "${SLIME_DIR}/run_14b_to_4b_fixed_context_300_20260517.sh"
else
  echo "Skipping 8B/14B -> 4B diagnostics because torch_dist is missing: ${TORCH4}"
fi

run_step "4B->1.7B GSM8K-COT diagnostic" bash "${SLIME_DIR}/run_4b_to_1p7b_gsm8k_fixed_context_300_20260517.sh"

log_step "Final P0 refresh"
python3 "${SLIME_DIR}/tools/build_section3_p0_analysis_20260517.py" || true
if [[ -f "${OUTPUT_ROOT}/storyline_20260513/collect_opd_research_assets_20260513.py" ]]; then
  python3 "${OUTPUT_ROOT}/storyline_20260513/collect_opd_research_assets_20260513.py" || true
fi
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
echo "P1 queue finished at $(date '+%F %T')"
