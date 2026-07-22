#!/usr/bin/env bash
set -euo pipefail

cd /path/to/slime-main
chmod +x ./run_ratio005_seed2_seed3_20260512.sh

TAG="ratio005_dlearn_vs_q3_seed2_seed3_20260512"
LOG_DIR="/path/to/outputs/slime_opd/logs"
mkdir -p "${LOG_DIR}"

echo "LAUNCH=${TAG}"
echo "GPU_BEFORE"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

setsid ./run_ratio005_seed2_seed3_20260512.sh \
  > "${LOG_DIR}/${TAG}.setsid.log" 2>&1 < /dev/null &
pid="$!"
echo "${pid}" > "${LOG_DIR}/${TAG}.pid"
echo "PID=${pid}"
