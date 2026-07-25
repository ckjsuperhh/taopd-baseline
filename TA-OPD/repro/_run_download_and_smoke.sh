#!/usr/bin/env bash
# 下模型 + 转 student + 备数据 + 跑 smoke test
# 前提: setup 已完成 (TE 2.10 + sglang + apex + Megatron + slime + patches)
set -eo pipefail
LOG="${HOME}/taopd-faithful-logs/pipeline.log"
mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1
echo "=== $(date) ==="

export MAMBA_EXE="${HOME}/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
eval "$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}")"
micromamba activate ta_opd_faithful

# 路径 (同 run_apex_faithful.sh)
REPO_ROOT="${HOME}/taopd-baseline/TA-OPD"
PROJECT_ROOT="${HOME}/taopd-baseline"
DATA_DIR="${PROJECT_ROOT}/data"
MODEL_DIR="${PROJECT_ROOT}/modelweights"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs"
mkdir -p "${DATA_DIR}" "${MODEL_DIR}" "${OUTPUT_ROOT}"

TEACHER_MODEL="${MODEL_DIR}/Qwen3-4B"
STUDENT_HF="${MODEL_DIR}/Qwen3-1.7B"
STUDENT_TORCH_DIST="${MODEL_DIR}/Qwen3-1.7B-torch_dist"

# 中国用 hf-mirror
export HF_ENDPOINT="https://hf-mirror.com"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "DATA_DIR=${DATA_DIR}"

echo ""
echo "=== [1/4] Download Qwen3-4B (teacher) ==="
if [[ -f "${TEACHER_MODEL}/config.json" ]]; then
  echo "已存在, 跳过"
else
  hf download Qwen/Qwen3-4B --local-dir "${TEACHER_MODEL}"
fi

echo ""
echo "=== [2/4] Download Qwen3-1.7B (student) ==="
if [[ -f "${STUDENT_HF}/config.json" ]]; then
  echo "已存在, 跳过"
else
  hf download Qwen/Qwen3-1.7B --local-dir "${STUDENT_HF}"
fi

echo ""
echo "=== [3/4] Convert Qwen3-1.7B -> torch_dist ==="
if [[ -d "${STUDENT_TORCH_DIST}" ]] && [[ -f "${STUDENT_TORCH_DIST}/latest_checkpointed_iteration.txt" || -d "${STUDENT_TORCH_DIST}/iter_0000000" ]]; then
  echo "已存在, 跳过"
else
  export SLIME_DIR="${REPO_ROOT}/slime_ta_opd"
  export MEGATRON_LM_DIR="${HOME}/taopd-faithful/Megatron-LM"
  # GPU 0 被占用, 用 GPU 1
  export CUDA_VISIBLE_DEVICES=1
  cd "${SLIME_DIR}"
  HF_MODEL="${STUDENT_HF}" \
  SAVE_DIR="${STUDENT_TORCH_DIST}" \
  NPROC_PER_NODE=1 \
  FORCE=1 \
  bash ./convert_qwen3_1_7b_to_torch_dist.sh 2>&1 | tail -30
fi

echo ""
echo "=== [4/4] Prepare DAPO-Math-17k ==="
if [[ -d "${DATA_DIR}/DAPO-Math-17k" ]] || [[ -f "${DATA_DIR}/dapo-math-17k.jsonl" ]]; then
  echo "已存在, 跳过"
else
  hf download ByteDance-Seed/DAPO-Math-17k --repo-type dataset --local-dir "${DATA_DIR}/DAPO-Math-17k"
fi

echo ""
echo "=== 全部完成 ==="
echo "Teacher: ${TEACHER_MODEL}"
echo "Student HF: ${STUDENT_HF}"
echo "Student torch_dist: ${STUDENT_TORCH_DIST}"
ls "${STUDENT_TORCH_DIST}" 2>/dev/null | head -5
echo "Data: ${DATA_DIR}"
ls "${DATA_DIR}" 2>/dev/null | head -5

echo "=== $(date) exit=$? ==="
