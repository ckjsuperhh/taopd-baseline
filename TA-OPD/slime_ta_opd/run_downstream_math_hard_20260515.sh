#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/hf_cache
export HF_HUB_CACHE=/path/to/hf_cache/hub
export TRANSFORMERS_CACHE=/path/to/hf_cache/transformers
export HF_DATASETS_CACHE=/path/to/hf_cache/datasets
export PYTHONPATH=/path/to/storage/user/python_deps/lmeval_math:${PYTHONPATH:-}

EVAL_BIN=/path/to/miniconda3/envs/lmeval/bin/lm-eval
PY_BIN=python3
TASK=${TASK:-leaderboard_math_hard}
BATCH_SIZE=${BATCH_SIZE:-4}
OUT_ROOT=/path/to/outputs/slime_opd/storyline_20260513/downstream_smoke_20260515/math_hard
SUMMARY_SCRIPT=/path/to/slime-main/tools/summarize_math_hard_20260515.py

mkdir -p "${OUT_ROOT}"

BASE=/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B
DLEARN=/path/to/outputs/slime_opd/budget_common_context_k16_ratio003_seed2_20260511/dlearn_high_max64_seed2/student_hf
Q3_HIGHC=/path/to/outputs/slime_opd/budget_common_context_k16_ratio003_seed2_20260511/q3_highc_max64_seed2/student_hf
TIP=/path/to/outputs/slime_opd/p1_tip_baselines_seed2_followup_20260513/tip_ratio003/student_hf
DIVERGENCE=/path/to/outputs/slime_opd/p1_tip_baselines_seed2_followup_20260513/divergence_ratio003/student_hf

run_one() {
  local gpu="$1"
  local label="$2"
  local model="$3"
  local out_dir="${OUT_ROOT}/${label}"
  mkdir -p "${out_dir}"

  if find "${out_dir}" -name 'results_*.json' -print -quit | grep -q .; then
    echo "[skip] ${label}: existing result found under ${out_dir}"
    "${PY_BIN}" "${SUMMARY_SCRIPT}" || true
    return 0
  fi

  echo "[run] gpu=${gpu} label=${label} model=${model} task=${TASK}"
  CUDA_VISIBLE_DEVICES="${gpu}" timeout 360m "${EVAL_BIN}" run \
    --model hf \
    --model_args "pretrained=${model},dtype=bfloat16,trust_remote_code=True" \
    --tasks "${TASK}" \
    --device cuda:0 \
    --batch_size "${BATCH_SIZE}" \
    --log_samples \
    --output_path "${out_dir}"

  "${PY_BIN}" "${SUMMARY_SCRIPT}" || true
}

queue_gpu0() {
  run_one 0 base_qwen3_1p7b "${BASE}"
  run_one 0 dlearn_high_ratio003_seed2 "${DLEARN}"
  run_one 0 divergence_ratio003_seed2 "${DIVERGENCE}"
}

queue_gpu1() {
  run_one 1 q3_highc_ratio003_seed2 "${Q3_HIGHC}"
  run_one 1 tip_ratio003_seed2 "${TIP}"
}

queue_gpu0 &
pid0=$!
queue_gpu1 &
pid1=$!

wait "${pid0}"
wait "${pid1}"

"${PY_BIN}" "${SUMMARY_SCRIPT}" || true
