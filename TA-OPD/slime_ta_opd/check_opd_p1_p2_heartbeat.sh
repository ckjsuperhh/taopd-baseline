#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="/path/to/slime-main"
OUTPUT_ROOT="/path/to/outputs/slime_opd"
TAG="p1_tip_baselines_seed3_followup_20260514"
LOG="${OUTPUT_ROOT}/logs/${TAG}.log"
RUN_DIR="${OUTPUT_ROOT}/${TAG}"
AGG_DIR="${OUTPUT_ROOT}/storyline_20260513/p1_p2_tip_baselines"
ARCHIVE_SCRIPT="${OUTPUT_ROOT}/storyline_20260513/collect_opd_research_assets_20260513.py"

echo "NOW"
date
echo "GPU"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo "PROCS"
ps -eo pid,ppid,stat,etime,cmd \
  | grep -E "p1_tip_baselines_seed3|run_opd_budget|sglang|ray|torchrun|train.py|eval_fixed_context|aggregate_p1_p2" \
  | grep -v grep || true

echo "LOG_PROGRESS"
grep -E "DONE mask=|START mask=|TRAIN_DONE mask=|PIPELINE_DONE|EVAL_START mask=" "${LOG}" | tail -n 160 || true

echo "SUMMARY_FILES"
find "${RUN_DIR}" -maxdepth 4 -type f -name q3_bootstrap_matching_summary.csv -print | sort || true

echo "SUMMARY_CONTENTS"
while IFS= read -r f; do
  echo "FILE=${f}"
  cat "${f}"
done < <(find "${RUN_DIR}" -maxdepth 4 -type f -name q3_bootstrap_matching_summary.csv -print | sort)

cd "${SLIME_DIR}"
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate verl
python3 aggregate_p1_p2_tip_baselines_20260513.py >/tmp/p1_agg_hb_seed3.out
python3 "${ARCHIVE_SCRIPT}" >/tmp/p1_archive_hb_seed3.out

echo "AGG_FOCUS"
grep -E "^(divergence|tip|random|ca_softor|q3|entropy)," "${AGG_DIR}/p1_p2_tip_baseline_aggregate_current.csv" || true
echo "MISSING_COUNT"
tail -n +2 "${AGG_DIR}/p1_p2_tip_baseline_missing_current.csv" | wc -l
echo "ARCHIVE"
tail -n 8 /tmp/p1_archive_hb_seed3.out
