#!/usr/bin/env bash
set -euo pipefail
SLIME_DIR=${SLIME_DIR:-/path/to/slime-main}
LOG_DIR=${LOG_DIR:-/path/to/outputs/slime_opd/logs}
SHARD=${SHARD:-/path/to/models/Qwen3/14B/Qwen3-14B/model-00001-of-00008.safetensors}
DL_PID_FILE=${DL_PID_FILE:-${LOG_DIR}/qwen3_14b_missing_shard_download_20260517.pid}
RUN_SCRIPT=${RUN_SCRIPT:-${SLIME_DIR}/run_14b_to_1p7b_fixed_context_300_20260517.sh}
MIN_BYTES=${MIN_BYTES:-3000000000}
mkdir -p "$LOG_DIR"
echo "=== queued 14B -> 1.7B diag300 after shard ==="
echo "SHARD=$SHARD"
echo "RUN_SCRIPT=$RUN_SCRIPT"
while true; do
  if [[ -s "$SHARD" ]]; then
    size=$(stat -c%s "$SHARD" 2>/dev/null || echo 0)
    if [[ "$size" -ge "$MIN_BYTES" ]]; then
      echo "$(date '+%F %T') shard ready size=$size"
      break
    fi
  fi
  if [[ -f "$DL_PID_FILE" ]]; then
    pid=$(cat "$DL_PID_FILE" || true)
    if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
      echo "$(date '+%F %T') waiting for shard download pid=$pid"
    else
      echo "$(date '+%F %T') download pid not active; shard not ready yet"
    fi
  else
    echo "$(date '+%F %T') no download pid file; shard not ready yet"
  fi
  sleep 300
done
cd "$SLIME_DIR"
exec bash "$RUN_SCRIPT"
