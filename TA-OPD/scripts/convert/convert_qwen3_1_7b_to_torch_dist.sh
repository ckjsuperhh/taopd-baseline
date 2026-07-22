#!/usr/bin/env bash
set -euo pipefail

# Convert Qwen3-1.7B HuggingFace checkpoint to slime/Megatron torch_dist format.
#
# Usage on the server:
#   cd /path/to/slime-main
#   bash /path/to/slime-main/convert_qwen3_1_7b_to_torch_dist.sh
#
# Optional overrides:
#   HF_MODEL=/path/to/hf_model SAVE_DIR=/path/to/output_torch_dist NPROC_PER_NODE=1 bash convert_qwen3_1_7b_to_torch_dist.sh

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-}"

if [[ -z "${MEGATRON_LM_DIR}" ]]; then
  if [[ -d /path/to/Megatron-LM ]]; then
    MEGATRON_LM_DIR=/path/to/Megatron-LM
  elif [[ -d /root/Megatron-LM ]]; then
    MEGATRON_LM_DIR=/root/Megatron-LM
  elif [[ -d /path/to/Megatron-LM ]]; then
    MEGATRON_LM_DIR=/path/to/Megatron-LM
  else
    MEGATRON_LM_DIR=/root/Megatron-LM
  fi
fi

HF_MODEL="${HF_MODEL:-/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B}"
SAVE_DIR="${SAVE_DIR:-/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B_torch_dist}"

# Qwen3-1.7B is small enough to convert on one GPU. Increase only if needed.
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29617}"
FORCE="${FORCE:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
# Some Megatron environments do not have Transformer Engine. Megatron's
# default rope fusion requires TE, so keep it disabled unless explicitly changed.
EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:---no-rope-fusion --transformer-impl local --no-persist-layer-norm --no-gradient-accumulation-fusion}"

if [[ ! -d "${SLIME_DIR}" ]]; then
  echo "ERROR: SLIME_DIR does not exist: ${SLIME_DIR}" >&2
  exit 1
fi

if [[ ! -d "${MEGATRON_LM_DIR}" ]]; then
  echo "ERROR: MEGATRON_LM_DIR does not exist: ${MEGATRON_LM_DIR}" >&2
  echo "Set MEGATRON_LM_DIR=/path/to/Megatron-LM and retry." >&2
  exit 1
fi

if [[ ! -d "${HF_MODEL}" ]]; then
  echo "ERROR: HF_MODEL does not exist: ${HF_MODEL}" >&2
  exit 1
fi

if [[ -e "${SAVE_DIR}" ]]; then
  if [[ "${FORCE}" != "1" ]]; then
    echo "ERROR: SAVE_DIR already exists: ${SAVE_DIR}" >&2
    echo "Set FORCE=1 to move it aside and reconvert." >&2
    exit 1
  fi
  backup="${SAVE_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
  echo "SAVE_DIR exists; moving old directory to ${backup}"
  mv "${SAVE_DIR}" "${backup}"
fi

mkdir -p "$(dirname "${SAVE_DIR}")"
cd "${SLIME_DIR}"

source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"

export PYTHONPATH="${MEGATRON_LM_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"

python - <<'PY'
import importlib.util
import sys

missing = []
for name in ("mbridge", "megatron"):
    if importlib.util.find_spec(name) is None:
        missing.append(name)

if missing:
    print("ERROR: missing Python module(s):", ", ".join(missing), file=sys.stderr)
    if "mbridge" in missing:
        print("Install in the active conda env with: python -m pip install mbridge==0.15.1", file=sys.stderr)
    if "megatron" in missing:
        print("Check MEGATRON_LM_DIR and PYTHONPATH.", file=sys.stderr)
    raise SystemExit(1)

from mbridge import AutoBridge  # noqa: F401
import slime_plugins.mbridge  # noqa: F401
PY

echo "=== Qwen3-1.7B HF -> torch_dist conversion ==="
echo "SLIME_DIR       = ${SLIME_DIR}"
echo "MEGATRON_LM_DIR = ${MEGATRON_LM_DIR}"
echo "HF_MODEL        = ${HF_MODEL}"
echo "SAVE_DIR        = ${SAVE_DIR}"
echo "NPROC_PER_NODE  = ${NPROC_PER_NODE}"
echo "EXTRA_ARGS      = ${EXTRA_MEGATRON_ARGS}"
echo

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "CHECK_ONLY=1, preflight passed. Exiting before conversion."
  exit 0
fi

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  ${EXTRA_MEGATRON_ARGS} \
  --hf-checkpoint "${HF_MODEL}" \
  --save "${SAVE_DIR}"

echo
echo "=== Conversion finished ==="
echo "Checkpoint tracker:"
cat "${SAVE_DIR}/latest_checkpointed_iteration.txt"
echo
echo "Output directory:"
find "${SAVE_DIR}" -maxdepth 2 -type d | sort
