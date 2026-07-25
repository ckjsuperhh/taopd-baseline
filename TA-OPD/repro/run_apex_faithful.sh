#!/usr/bin/env bash
# Apex 服务器上的完整 faithful 流水线 (严格官方 build_conda.sh)
#
# 流程:
#   1. setup_faithful.sh        — 装 micromamba + conda env + CUDA 12.9 +
#                                  torch 2.9.1 + sglang(源码) + flash-attn 2.7.4
#                                  + transformer_engine + apex + ... + apply patches
#   2. 02_download_models.sh    — 下 Qwen3-4B + Qwen3-1.7B (hf-mirror)
#   3. 03_convert_student.sh    — Qwen3-1.7B → torch_dist (Megatron 格式)
#   4. 04_prepare_data.sh       — 下 DAPO-Math-17k + GSM8K-COT
#   5. 05_smoke_test_faithful.sh — 2 个 smoke run (pure_opd + ta_opd)
#
# 用法 (在 apex 服务器上):
#   # 默认装在 ~/taopd-faithful (如果 /home 空间不够就换挂载点):
#   BASE_DIR=/ext1/kejiechen/taopd-faithful \
#   bash ~/taopd-baseline/TA-OPD/repro/run_apex_faithful.sh
#
# 预计耗时: 4-6 小时 (含源码编译 flash-attn + transformer_engine + apex)
set -eo pipefail

# ── 路径配置 ─────────────────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-${HOME}/taopd-faithful}"
export BASE_DIR
FAITHFUL_ENV="${FAITHFUL_ENV:-ta_opd_faithful}"
export FAITHFUL_ENV

# 找到 REPO_ROOT (这个脚本所在的上一级)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

# Apex 服务器的资源路径 (覆盖 00_env.sh 默认值)
export PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT%/*}}"
export DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
export MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/modelweights}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"

# conda/micromamba 选择
MICROMAMBA_ROOT="${MICROMAMBA_ROOT:-${HOME}/micromamba}"
export MICROMAMBA_ROOT

echo "========================================="
echo " Apex faithful pipeline"
echo " REPO_ROOT    = ${REPO_ROOT}"
echo " BASE_DIR     = ${BASE_DIR}"
echo " PROJECT_ROOT = ${PROJECT_ROOT}"
echo " MODEL_DIR    = ${MODEL_DIR}"
echo " DATA_DIR     = ${DATA_DIR}"
echo " OUTPUT_ROOT  = ${OUTPUT_ROOT}"
echo " FAITHFUL_ENV = ${FAITHFUL_ENV}"
echo "========================================="

# ── 磁盘检查 ───────────────────────────────────────────────────────────
AVAIL_GB=$(df -BG "${BASE_DIR%/*}" 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
echo "可用磁盘: ${AVAIL_GB:-?} G (在 ${BASE_DIR%/*})"
if [[ -n "${AVAIL_GB}" ]] && [[ "${AVAIL_GB}" -lt 80 ]]; then
  echo "❌ 磁盘不足: 完整 faithful build + 模型 + 数据需要 ≥80G"
  echo "   换挂载点: BASE_DIR=/ext1/kejiechen/taopd-faithful bash $0"
  exit 1
fi

# ── Step 1: faithful env build ───────────────────────────────────────────
echo ""
echo "==================== Step 1/5: setup_faithful ===================="
bash "${REPO_ROOT}/repro/setup_faithful.sh"

# ── Step 2: 下载模型 ────────────────────────────────────────────────────
echo ""
echo "==================== Step 2/5: download_models ===================="
mkdir -p "${MODEL_DIR}" "${DATA_DIR}" "${OUTPUT_ROOT}"

# 用 faithful env 跑 huggingface-cli
if [[ -x "${MICROMAMBA_ROOT}/bin/micromamba" ]]; then
  eval "$("${MICROMAMBA_ROOT}/bin/micromamba" shell hook --shell bash)"
  micromamba activate "${FAITHFUL_ENV}"
else
  CONDA_SH="${CONDA_SH:-$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")/etc/profile.d/conda.sh}"
  source "${CONDA_SH}"
  conda activate "${FAITHFUL_ENV}"
fi

if [[ ! -f "${MODEL_DIR}/Qwen3-4B/config.json" ]]; then
  echo "下载 Qwen3-4B..."
  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-4B \
    --local-dir "${MODEL_DIR}/Qwen3-4B"
fi
if [[ ! -f "${MODEL_DIR}/Qwen3-1.7B/config.json" ]]; then
  echo "下载 Qwen3-1.7B (Base, 用作 student)..."
  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-1.7B-Base \
    --local-dir "${MODEL_DIR}/Qwen3-1.7B"
fi

# ── Step 3: 转换 student 到 torch_dist ─────────────────────────────────
echo ""
echo "==================== Step 3/5: convert_student ===================="
if [[ ! -f "${MODEL_DIR}/Qwen3-1.7B_torch_dist/latest_checkpointed_iteration.txt" ]]; then
  # 用 faithful env 跑 03_convert_student.sh, 但要 override SLIME_DIR/MEGATRON_LM_DIR
  export SLIME_DIR="${BASE_DIR}/slime"
  export MEGATRON_LM_DIR="${BASE_DIR}/Megatron-LM"
  export STUDENT_HF="${MODEL_DIR}/Qwen3-1.7B"
  export STUDENT_TORCH_DIST="${MODEL_DIR}/Qwen3-1.7B_torch_dist"

  # 03_convert_student.sh 用 activate_env (conda), 在 micromamba env 下会失败
  # 所以直接复刻它的核心逻辑:
  cd "${SLIME_DIR}"
  source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"
  CONVERT_GPU="${CONVERT_GPU:-0}"
  CUDA_VISIBLE_DEVICES="${CONVERT_GPU}" torchrun \
    --standalone --nnodes 1 --nproc_per_node 1 \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --no-rope-fusion --transformer-impl local \
    --no-persist-layer-norm --no-gradient-accumulation-fusion \
    --hf-checkpoint "${STUDENT_HF}" \
    --save "${STUDENT_TORCH_DIST}"
fi
echo "✅ student torch_dist ready: $(cat ${MODEL_DIR}/Qwen3-1.7B_torch_dist/latest_checkpointed_iteration.txt)"

# ── Step 4: 准备数据 ──────────────────────────────────────────────────
echo ""
echo "==================== Step 4/5: prepare_data ===================="
DAPO_DIR="${DATA_DIR}/DAPO-Math-17k-dedup"
mkdir -p "${DAPO_DIR}"
if [[ ! -f "${DAPO_DIR}/dapo_math_17k_dedup_slime.jsonl" ]]; then
  echo "下载 DAPO-Math-17k-dedup..."
  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download bytedance-research/DAPO-Math-17k-dedup \
    --repo-type dataset --local-dir "${DAPO_DIR}" || {
    echo "⚠ huggingface-cli 失败, 试 wget 原始 jsonl..."
    # fallback: 直接从 dataset repo 拉
    wget -qO "${DAPO_DIR}/dapo_math_17k_dedup_slime.jsonl" \
      "https://hf-mirror.com/datasets/bytedance-research/DAPO-Math-17k-dedup/resolve/main/dapo_math_17k_dedup_slime.jsonl" \
      || { echo "❌ 数据下载失败, 请手动准备"; exit 1; }
  }
fi

# ── Step 5: smoke test ──────────────────────────────────────────────────
echo ""
echo "==================== Step 5/5: smoke_test_faithful ===================="
bash "${REPO_ROOT}/repro/05_smoke_test_faithful.sh"

echo ""
echo "========================================="
echo " 🎉 Apex faithful pipeline 完成!"
echo " Smoke test 输出: ${OUTPUT_ROOT}/smoke_test_faithful"
echo "========================================="
