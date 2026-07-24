#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " TA-OPD 4B→1.7B Full Reproduction Pipeline"
echo " Qwen3-4B (teacher) → Qwen3-1.7B (student)"
echo ""
echo " 84 training runs + 2 diagnostics + downstream eval"
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
    echo ">>> To resume from this step: bash repro/run_all.sh ${step_num}"
    echo ""
    exit 1
  fi
done

echo "============================================================"
echo " ALL STEPS COMPLETE!"
echo " $(date '+%F %T')"
echo ""
echo " Results: $(source "${SCRIPT_DIR}/00_env.sh" 2>/dev/null; echo "${OUTPUT_ROOT}/results")"
echo "============================================================"
