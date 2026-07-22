#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
LOG_DIR="${LOG_DIR:-/path/to/outputs/slime_opd/logs}"
WAIT_PID_FILE="${WAIT_PID_FILE:-${LOG_DIR}/8b_to_1p7b_diag300_20260517.pid}"
RUN_SCRIPT="${RUN_SCRIPT:-${SLIME_DIR}/run_4b_to_1p7b_heldout_fixed_context_300_20260517.sh}"

mkdir -p "${LOG_DIR}"
echo "=== queued 4B -> 1.7B heldout300 diagnostic ==="
echo "Waiting for: ${WAIT_PID_FILE}"
echo "Run script: ${RUN_SCRIPT}"

while true; do
  pid=""
  if [[ -f "${WAIT_PID_FILE}" ]]; then
    pid="$(cat "${WAIT_PID_FILE}" || true)"
  fi
  if [[ -z "${pid}" ]] || ! ps -p "${pid}" >/dev/null 2>&1; then
    break
  fi
  echo "$(date '+%F %T') still waiting for PID ${pid}"
  sleep 300
done

echo "$(date '+%F %T') upstream job finished; giving CUDA/Ray one minute to release resources."
sleep 60
cd "${SLIME_DIR}"
exec bash "${RUN_SCRIPT}"
