#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

echo "========================================="
echo " Step 2: Download model weights"
echo "========================================="

mkdir -p "${MODEL_DIR}"

# ── 1. Download Qwen3-4B (teacher) ──────────────────────────────────────
echo "[1/2] Downloading Qwen3-4B (teacher)..."
if [[ -d "${TEACHER_MODEL}" ]] && [[ -f "${TEACHER_MODEL}/config.json" ]]; then
  echo "  Already exists at ${TEACHER_MODEL}, skipping."
else
  huggingface-cli download Qwen/Qwen3-4B \
    --local-dir "${TEACHER_MODEL}" \
    --local-dir-use-symlinks False
  echo "  Downloaded to ${TEACHER_MODEL}"
fi

# ── 2. Download Qwen3-1.7B (student) ────────────────────────────────────
echo "[2/2] Downloading Qwen3-1.7B (student)..."
if [[ -d "${STUDENT_HF}" ]] && [[ -f "${STUDENT_HF}/config.json" ]]; then
  echo "  Already exists at ${STUDENT_HF}, skipping."
else
  huggingface-cli download Qwen/Qwen3-1.7B \
    --local-dir "${STUDENT_HF}" \
    --local-dir-use-symlinks False
  echo "  Downloaded to ${STUDENT_HF}"
fi

echo ""
echo "========================================="
echo " Model download complete!"
echo " Teacher: ${TEACHER_MODEL}"
echo " Student: ${STUDENT_HF}"
echo "========================================="
