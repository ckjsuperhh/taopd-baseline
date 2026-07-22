#!/usr/bin/env bash
set -euo pipefail

RUN="/path/to/outputs/slime_opd/qwen3_1_7b_dapo_budget_k16_ratio005_random_max64_seed3_p1_tip_baselines_seed3_followup_20260514"
EVAL="/path/to/outputs/slime_opd/p1_tip_baselines_seed3_followup_20260514/random_ratio005"
LOG="/path/to/outputs/slime_opd/logs/p1_tip_baselines_seed3_followup_20260514.log"

echo "RUN_DIR"
ls -la "${RUN}" 2>/dev/null || true
echo "CKPT"
find "${RUN}" -maxdepth 2 -type f \( -name latest_checkpointed_iteration.txt -o -name "*.pt" -o -name "*.parquet" \) 2>/dev/null | sort | tail -n 60 || true
echo "TOKEN_BANK"
find "${RUN}/token_bank" -maxdepth 1 -type f 2>/dev/null | sort | tail -n 30 || true
echo "EVAL_DIR"
find "${EVAL}" -maxdepth 3 -type f 2>/dev/null | sort || true
echo "LOG_TAIL"
tail -n 140 "${LOG}" || true
