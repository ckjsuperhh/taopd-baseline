#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/slime_opd}"
STORY_ROOT="${STORY_ROOT:-${OUTPUT_ROOT}/storyline_20260513/scale_context_robustness_20260517}"
PY="${PY:-python3}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/path/to/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/path/to/hf_cache/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/path/to/hf_cache/hub}"
export PYTHONPATH="/path/to/Megatron-LM:${SLIME_DIR}:/path/to/miniconda3/envs/verl/lib/python3.10/site-packages:${PYTHONPATH:-}"

TEACHER_MODEL="${TEACHER_MODEL:-/path/to/models/Qwen3/14B/Qwen3-14B}"
STUDENT_HF="${STUDENT_HF:-/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B}"
RUN_NAME="${RUN_NAME:-qwen3_1_7b_student_qwen3_14b_teacher_diag300_k16_20260517}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
BASE_DIAG="${BASE_DIAG:-${STORY_ROOT}/14b_to_1p7b_diag300}"

cd "${SLIME_DIR}"

for K in 8 32; do
  DIAG_DIR="${STORY_ROOT}/14b_to_1p7b_diag300_k${K}"
  mkdir -p "${DIAG_DIR}"
  cp "${BASE_DIAG}/context_bank.parquet" "${DIAG_DIR}/context_bank.parquet"
  echo "=== 14B -> 1.7B retopk K=${K} ==="

  "${PY}" tools/eval_fixed_context_bank.py \
    --context-bank "${DIAG_DIR}/context_bank.parquet" \
    --student "${STUDENT_HF}" \
    --teacher "${TEACHER_MODEL}" \
    --output "${DIAG_DIR}/theta0_metrics.parquet" \
    --student-device cuda:0 \
    --teacher-device cuda:1 \
    --dtype bfloat16 \
    --topk "${K}" \
    --max-samples 300 \
    --max-response-tokens 192 \
    --trust-remote-code

  "${PY}" tools/eval_fixed_context_from_megatron.py \
    --checkpoint-root "${SAVE_DIR}" \
    --iteration latest \
    --origin-hf-dir "${STUDENT_HF}" \
    --teacher-hf-dir "${TEACHER_MODEL}" \
    --context-bank "${DIAG_DIR}/context_bank.parquet" \
    --baseline-metrics "${DIAG_DIR}/theta0_metrics.parquet" \
    --output-dir "${DIAG_DIR}/eval_latest" \
    --student-device cuda:0 \
    --teacher-device cuda:1 \
    --dtype bfloat16 \
    --topk "${K}" \
    --max-samples 300 \
    --max-response-tokens 192 \
    --force-convert

  "${PY}" tools/analyze_fixed_context_gain.py \
    --input "${DIAG_DIR}/eval_latest/fixed_context_metrics.parquet" \
    --output-dir "${DIAG_DIR}/analysis/gain" \
    --bootstrap 1000

  TOPK="${K}" DIAG_DIR="${DIAG_DIR}" SAVE_DIR="${SAVE_DIR}" "${PY}" - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ["DIAG_DIR"])
ctx = pd.read_parquet(root / "context_bank.parquet")
metrics = pd.read_parquet(root / "eval_latest/fixed_context_metrics.parquet")
q3 = pd.read_csv(root / "analysis/gain/q3_bootstrap_matching_summary.csv").iloc[0]
reg = pd.read_csv(root / "analysis/gain/regression_summary.csv")
k = os.environ["TOPK"]
lines = [
    f"# 14B -> 1.7B 300-context retopk K={k}",
    "",
    f"Source run: `{os.environ['SAVE_DIR']}`.",
    f"Context bank: {len(ctx)} samples; eval metrics: {len(metrics)} token rows.",
    f"Q3 bootstrap high-minus-low mean diff: {q3.bootstrap_mean_diff:.6f} [{q3.bootstrap_ci_low:.6f}, {q3.bootstrap_ci_high:.6f}]; matching mean diff: {q3.matching_mean_diff:.6f} over {int(q3.matching_pairs)} pairs.",
    "",
    "Regression summary:",
    reg.to_markdown(index=False),
]
(root / f"diag300_k{k}_summary.md").write_text("\n".join(lines) + "\n")
print(root / f"diag300_k{k}_summary.md")
PY
done

"${PY}" tools/build_section3_p0_analysis_20260517.py || true
if [[ -f "${OUTPUT_ROOT}/storyline_20260513/collect_opd_research_assets_20260513.py" ]]; then
  "${PY}" "${OUTPUT_ROOT}/storyline_20260513/collect_opd_research_assets_20260513.py" || true
fi
