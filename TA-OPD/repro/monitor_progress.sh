#!/usr/bin/env bash
# Quick training progress monitor
# Usage: bash monitor_progress.sh

set -eo pipefail

SWEEP_ROOT="/inspire/hdd/project/multi-agent/zhangweinan-24046/dk/outputs/main_sweep"
LOG_DIR="${SWEEP_ROOT}/logs"

echo "========================================="
echo " TA-OPD Training Progress Monitor"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

# tmux session status
echo ""
echo "--- tmux session ---"
tmux ls 2>/dev/null || echo "No tmux sessions!"

# GPU status
echo ""
echo "--- GPU Status ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null

# Completed runs (checkpoints)
echo ""
echo "--- Completed Runs (with checkpoints) ---"
completed=0
for d in "${SWEEP_ROOT}"/k16_*/; do
  if [[ -f "${d}*/latest_checkpointed_iteration.txt" ]] 2>/dev/null; then
    run_name=$(basename "$(dirname "${d}")")/$(basename "${d}")
    iter=$(cat "${d}"/*/latest_checkpointed_iteration.txt 2>/dev/null | head -1)
    echo "  ✓ $(basename "${d}") (iteration ${iter})"
    completed=$((completed + 1))
  fi
done
echo "Total completed: ${completed}/84"

# Log files
echo ""
echo "--- Recent Log Files ---"
ls -lt "${LOG_DIR}"/*.log 2>/dev/null | head -10 | awk '{print $6, $7, $8, $9, $5}'

# Check if training process is alive
echo ""
echo "--- Process Status ---"
if pgrep -f "06_run_all_training" >/dev/null 2>&1; then
  echo "  ✓ Training script is running"
else
  echo "  ✗ Training script is NOT running!"
fi

if pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
  echo "  ✓ Teacher SGLang servers running ($(pgrep -cf "sglang.launch_server") instances)"
else
  echo "  - No teacher SGLang servers"
fi

if pgrep -f "train.py" >/dev/null 2>&1; then
  echo "  ✓ Training workers active"
else
  echo "  - No training workers (may be between runs)"
fi

echo ""
echo "--- To attach to training terminal ---"
echo "  tmux attach -t taopd_train"
echo "  (Ctrl+B then D to detach)"
echo "========================================="
