#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
LOG_FILE="${LOG_FILE:-/path/to/outputs/slime_opd/logs/p1_p2_aggregate_watcher_20260513.log}"
ITERATIONS="${ITERATIONS:-72}"
SLEEP_SECONDS="${SLEEP_SECONDS:-600}"

for i in $(seq 1 "${ITERATIONS}"); do
  {
    echo "===== aggregate iteration ${i} ====="
    date
    cd "${SLIME_DIR}"
    source /path/to/miniconda3/etc/profile.d/conda.sh
    conda activate verl
    python3 aggregate_p1_p2_tip_baselines_20260513.py
  } >> "${LOG_FILE}" 2>&1 || true
  sleep "${SLEEP_SECONDS}"
done
