#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

echo "========================================="
echo " Step 3: Convert student to torch_dist"
echo "========================================="

if [[ -d "${STUDENT_TORCH_DIST}" ]] && [[ -f "${STUDENT_TORCH_DIST}/latest_checkpointed_iteration.txt" ]]; then
  echo "torch_dist already exists at ${STUDENT_TORCH_DIST}, skipping."
  echo "latest_checkpointed_iteration=$(cat "${STUDENT_TORCH_DIST}/latest_checkpointed_iteration.txt")"
  exit 0
fi

export PYTHONPATH="${MEGATRON_LM_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
TORCH_CUDA_LIB="$(get_torch_cuda_lib)"
export LD_LIBRARY_PATH="${TORCH_CUDA_LIB}:${LD_LIBRARY_PATH:-}"

cd "${SLIME_DIR}"
source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"

echo "Converting ${STUDENT_HF} → ${STUDENT_TORCH_DIST}"
echo "Using 1 GPU for conversion..."

CONVERT_GPU="${CONVERT_GPU:-0}"
CUDA_VISIBLE_DEVICES="${CONVERT_GPU}" torchrun \
  --standalone --nnodes 1 --nproc_per_node 1 \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --no-rope-fusion \
  --transformer-impl local \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --hf-checkpoint "${STUDENT_HF}" \
  --save "${STUDENT_TORCH_DIST}"

if [[ -f "${STUDENT_TORCH_DIST}/latest_checkpointed_iteration.txt" ]]; then
  echo ""
  echo "========================================="
  echo " Conversion complete!"
  echo " Output: ${STUDENT_TORCH_DIST}"
  echo " latest_checkpointed_iteration=$(cat "${STUDENT_TORCH_DIST}/latest_checkpointed_iteration.txt")"
  echo "========================================="
else
  echo "ERROR: conversion failed — no latest_checkpointed_iteration.txt" >&2
  exit 1
fi
