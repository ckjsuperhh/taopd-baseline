#!/usr/bin/env bash
set -euo pipefail

cd /path/to/slime-main
chmod +x ./run_opd_budget_ratio_sweep_20260512.sh

TAG="${TAG:-budget_ratio_sweep_k16_seed1_20260512_resume_v1}"
RUN_LIST="${RUN_LIST:-dlearn_high:0.05:20 dlearn_high:0.10:30 q3_highc:0.01:40 q3_highc:0.05:50}"
LOG_DIR="/path/to/outputs/slime_opd/logs"
mkdir -p "${LOG_DIR}"

echo "LAUNCH_TAG=${TAG}"
echo "RUN_LIST=${RUN_LIST}"
echo "GPU_BEFORE"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

setsid env TAG="${TAG}" RUN_LIST="${RUN_LIST}" ./run_opd_budget_ratio_sweep_20260512.sh \
  > "${LOG_DIR}/${TAG}.setsid.log" 2>&1 < /dev/null &
pid="$!"
echo "${pid}" > "${LOG_DIR}/${TAG}.pid"
echo "PID=${pid}"
