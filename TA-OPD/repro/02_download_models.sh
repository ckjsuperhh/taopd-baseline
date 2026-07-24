#!/usr/bin/env bash
set -eo pipefail  # 不能用 -u：conda 内部 deactivate 脚本有 unbound variable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

echo "========================================="
echo " Step 2: Download model weights"
echo "========================================="

mkdir -p "${MODEL_DIR}"

# Detect HF CLI: new name is `hf`, old name is `huggingface-cli`
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
  # 新版 hf CLI 不支持 --local-dir-use-symlinks (默认就是 no symlinks)
  HF_SYMLINK_FLAG=""
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
  HF_SYMLINK_FLAG="--local-dir-use-symlinks False"
else
  echo "❌ 找不到 hf/huggingface-cli (conda env 里应该已装 huggingface_hub)"
  exit 1
fi
echo "Using HF CLI: ${HF_CLI}"

# ── 1. Download Qwen3-4B (teacher) ──────────────────────────────────────
echo "[1/2] Downloading Qwen3-4B (teacher)..."
if [[ -d "${TEACHER_MODEL}" ]] && [[ -f "${TEACHER_MODEL}/config.json" ]]; then
  echo "  Already exists at ${TEACHER_MODEL}, skipping."
else
  ${HF_CLI} download Qwen/Qwen3-4B \
    --local-dir "${TEACHER_MODEL}" ${HF_SYMLINK_FLAG}
  echo "  Downloaded to ${TEACHER_MODEL}"
fi

# ── 2. Download Qwen3-1.7B (student) ────────────────────────────────────
echo "[2/2] Downloading Qwen3-1.7B (student)..."
if [[ -d "${STUDENT_HF}" ]] && [[ -f "${STUDENT_HF}/config.json" ]]; then
  echo "  Already exists at ${STUDENT_HF}, skipping."
else
  ${HF_CLI} download Qwen/Qwen3-1.7B \
    --local-dir "${STUDENT_HF}" ${HF_SYMLINK_FLAG}
  echo "  Downloaded to ${STUDENT_HF}"
fi

echo ""
echo "========================================="
echo " Model download complete!"
echo " Teacher: ${TEACHER_MODEL}"
echo " Student: ${STUDENT_HF}"
echo "========================================="
