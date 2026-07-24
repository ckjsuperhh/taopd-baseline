#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh" 2>/dev/null || true

TMUX_SESSION="${TMUX_SESSION:-ta_opd_repro}"
LOG_FILE="${OUTPUT_ROOT:-/inspire/hdd/project/multi-agent/zhangweinan-24046/dk/outputs}/logs/run_all_$(date +%Y%m%d_%H%M%S).log"

# ══════════════════════════════════════════════════════════════════════════
# Auto-launch in tmux if not already inside one
# ══════════════════════════════════════════════════════════════════════════
if [[ -z "${TMUX:-}" ]]; then
  # Not inside tmux — launch a persistent session
  if command -v tmux &>/dev/null; then
    mkdir -p "$(dirname "${LOG_FILE}")"

    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
      echo "tmux session '${TMUX_SESSION}' already exists."
      echo "  Attach:  tmux attach -t ${TMUX_SESSION}"
      echo "  Kill:    tmux kill-session -t ${TMUX_SESSION}"
      exit 0
    fi

    echo "============================================================"
    echo " Launching pipeline in tmux session: ${TMUX_SESSION}"
    echo "============================================================"
    echo ""
    echo "  Log file:  ${LOG_FILE}"
    echo ""
    echo "  To monitor progress:"
    echo "    tmux attach -t ${TMUX_SESSION}       # attach to session"
    echo "    tail -f ${LOG_FILE}                  # tail the log"
    echo ""
    echo "  To detach from tmux:  Ctrl-B then D"
    echo "  To kill the session:  tmux kill-session -t ${TMUX_SESSION}"
    echo ""

    tmux new-session -d -s "${TMUX_SESSION}" \
      "bash '${BASH_SOURCE[0]}' ${1:-} 2>&1 | tee '${LOG_FILE}'"
    exit 0
  else
    echo "WARNING: tmux not found. Falling back to nohup."
    mkdir -p "$(dirname "${LOG_FILE}")"
    echo "  Log file: ${LOG_FILE}"
    echo "  Run: tail -f ${LOG_FILE}"
    echo ""
    nohup bash "${BASH_SOURCE[0]}" ${1:-} > "${LOG_FILE}" 2>&1 &
    echo "  PID: $!"
    exit 0
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
# Running inside tmux — execute the pipeline
# ══════════════════════════════════════════════════════════════════════════

echo "============================================================"
echo " TA-OPD 4B→1.7B Full Reproduction Pipeline"
echo " Qwen3-4B (teacher) → Qwen3-1.7B (student)"
echo ""
echo " 84 training runs + 2 diagnostics + downstream eval"
echo " Started: $(date '+%F %T')"
echo " tmux session: ${TMUX_SESSION:-inside}"
echo "============================================================"
echo ""

# Allow starting from a specific step: bash run_all.sh [start_step]
START_STEP="${1:-1}"

STEPS=(
  "01_setup_env.sh:Environment setup (conda, torch, deps, patches)"
  "02_download_models.sh:Download Qwen3-4B + Qwen3-1.7B"
  "03_convert_student.sh:Convert student to torch_dist"
  "04_prepare_data.sh:Prepare DAPO-Math-17k + heldout + GSM8K"
  "05_smoke_test.sh:Smoke test (2 quick runs)"
  "06_run_all_training.sh:Main training sweep (84 runs, ~2-3 days)"
  "07_run_diagnostics.sh:Diagnostic runs (heldout + gsm8k)"
  "08_downstream_eval.sh:Downstream evaluation (GSM8K, AIME)"
  "09_aggregate_results.sh:Aggregate results and generate tables"
)

echo "Pipeline steps:"
for i in "${!STEPS[@]}"; do
  IFS=: read -r script desc <<< "${STEPS[$i]}"
  step_num=$((i + 1))
  if [[ "${step_num}" -lt "${START_STEP}" ]]; then
    echo "  [SKIP] Step ${step_num}: ${desc}"
  else
    echo "  [    ] Step ${step_num}: ${desc}"
  fi
done
echo ""

for i in "${!STEPS[@]}"; do
  IFS=: read -r script desc <<< "${STEPS[$i]}"
  step_num=$((i + 1))

  if [[ "${step_num}" -lt "${START_STEP}" ]]; then
    continue
  fi

  echo "============================================================"
  echo " Step ${step_num}: ${desc}"
  echo " Script: ${script}"
  echo " Started: $(date '+%F %T')"
  echo "============================================================"
  echo ""

  if bash "${SCRIPT_DIR}/${script}"; then
    echo ""
    echo ">>> Step ${step_num} PASSED ($(date '+%F %T'))"
    echo ""
  else
    echo ""
    echo ">>> Step ${step_num} FAILED ($(date '+%F %T'))"
    echo ">>> To resume from this step:"
    echo ">>>   bash repro/run_all.sh ${step_num}"
    echo ""
    exit 1
  fi
done

echo "============================================================"
echo " ALL STEPS COMPLETE!"
echo " $(date '+%F %T')"
echo ""
echo " Results: ${OUTPUT_ROOT}/results"
echo "============================================================"
