#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/slime_opd}"
STORY_ROOT="${STORY_ROOT:-${OUTPUT_ROOT}/storyline_20260513/scale_context_robustness_20260517}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/path/to/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/path/to/hf_cache/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/path/to/hf_cache/hub}"

TEACHER_MODEL="${TEACHER_MODEL:-/path/to/models/Qwen3/8B/Qwen/Qwen3-8B}"
STUDENT_HF="${STUDENT_HF:-/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B}"
RUN_NAME="${RUN_NAME:-qwen3_1_7b_student_qwen3_8b_teacher_short_diag_k16_v2_20260517}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
DIAG_DIR="${DIAG_DIR:-${STORY_ROOT}/8b_to_1p7b_short}"
DRIVER_LOG="${LOG_DIR}/${RUN_NAME}_driver.log"

mkdir -p "${LOG_DIR}" "${DIAG_DIR}"
cd "${SLIME_DIR}"

echo "=== 8B -> 1.7B short fixed-context diagnostic ==="
echo "RUN_NAME=${RUN_NAME}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "DIAG_DIR=${DIAG_DIR}"
echo "DRIVER_LOG=${DRIVER_LOG}"

# Two-GPU A800 node layout:
#   GPU0: 8B teacher SGLang
#   GPU1: 1.7B student actor + rollout, colocated for this short diagnostic
TEACHER_MODEL="${TEACHER_MODEL}" \
STUDENT_HF="${STUDENT_HF}" \
RUN_NAME="${RUN_NAME}" \
SAVE_DIR="${SAVE_DIR}" \
TEACHER_GPU=0 \
RAY_GPUS=1 \
COLOCATE=1 \
ACTOR_NUM_GPUS_PER_NODE=1 \
ROLLOUT_NUM_GPUS=1 \
TEACHER_PORT=13241 \
TEACHER_NCCL_PORT=23241 \
RAY_PORT=26441 \
RAY_DASHBOARD_PORT=8341 \
RAY_OBJECT_MANAGER_PORT=28441 \
RAY_NODE_MANAGER_PORT=28442 \
RAY_DASHBOARD_AGENT_LISTEN_PORT=28443 \
RAY_DASHBOARD_AGENT_GRPC_PORT=28444 \
RAY_METRICS_EXPORT_PORT=28445 \
RAY_TEMP_DIR=/tmp/slime_ray_26441 \
DATA_DIR=/path/to/slime-main/data/DAPO-Math-17k-dedup \
PROMPT_DATA=/path/to/slime-main/data/DAPO-Math-17k-dedup/dapo_math_17k_dedup_slime.jsonl \
NUM_ROLLOUT=4 \
ROLLOUT_BATCH_SIZE=2 \
N_SAMPLES_PER_PROMPT=1 \
GLOBAL_BATCH_SIZE=2 \
ROLLOUT_MAX_RESPONSE_LEN=256 \
MICRO_BATCH_SIZE=1 \
SAVE_INTERVAL=1 \
TEACHER_MEM_FRACTION=0.62 \
TEACHER_CUDA_GRAPH_MAX_BS=2 \
ROLLOUT_MEM_FRACTION=0.22 \
SGLANG_CUDA_GRAPH_MAX_BS=2 \
OPD_TOPK_METRICS_K=16 \
OPD_TOKEN_BANK_RAW_TOPK=1 \
OPD_TOKEN_BANK_PAIR_ID=qwen3_8b_to_qwen3_1p7b \
OPD_EXACT_CMASS=1 \
OPD_EXACT_CMASS_MAX_UNION=128 \
SAVE_DEBUG_ROLLOUT_DATA=1 \
SEED=817 \
ROLLOUT_SEED=817 \
OPD_BUDGET_MASK_SEED=817 \
bash examples/on_policy_distillation/run-qwen3-1.7B-sampled-opd-sglang.sh \
  > "${DRIVER_LOG}" 2>&1

echo "Training/logprob pass finished. Building fixed-context bank..."
python tools/export_fixed_context_bank.py \
  --debug-rollout-data "${SAVE_DIR}/debug_rollout_data/*.pt" \
  --output "${DIAG_DIR}/context_bank.parquet" \
  --max-samples 40

echo "Scoring theta0 against 8B teacher..."
python tools/eval_fixed_context_bank.py \
  --context-bank "${DIAG_DIR}/context_bank.parquet" \
  --student "${STUDENT_HF}" \
  --teacher "${TEACHER_MODEL}" \
  --output "${DIAG_DIR}/theta0_metrics.parquet" \
  --student-device cuda:0 \
  --teacher-device cuda:1 \
  --dtype bfloat16 \
  --topk 16 \
  --max-samples 40 \
  --max-response-tokens 128

echo "Converting/scoring latest Megatron checkpoint against 8B teacher..."
python tools/eval_fixed_context_from_megatron.py \
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
  --topk 16 \
  --max-samples 40 \
  --max-response-tokens 128 \
  --force-convert

echo "Analyzing fixed-context gain..."
python tools/analyze_fixed_context_gain.py \
  --input "${DIAG_DIR}/eval_latest/fixed_context_metrics.parquet" \
  --output-dir "${DIAG_DIR}/analysis/gain" \
  --bootstrap 800

python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("/path/to/outputs/slime_opd/storyline_20260513/scale_context_robustness_20260517/8b_to_1p7b_short")
ctx = pd.read_parquet(root / "context_bank.parquet")
reg = pd.read_csv(root / "analysis/gain/regression_summary.csv")
q3 = pd.read_csv(root / "analysis/gain/q3_bootstrap_matching_summary.csv")
row = q3.iloc[0]
lines = [
    "# 8B -> 1.7B short fixed-context diagnostic",
    "",
    f"Context bank: {len(ctx)} samples, {int(ctx.resp_len.sum())} response tokens before eval cap.",
    f"Q3 bootstrap high-minus-low mean diff: {row.bootstrap_mean_diff:.6f} [{row.bootstrap_ci_low:.6f}, {row.bootstrap_ci_high:.6f}]; matching mean diff: {row.matching_mean_diff:.6f} over {int(row.matching_pairs)} pairs.",
    "",
    "Regression summary:",
    reg.to_markdown(index=False),
    "",
    "Note: this is a deliberately short teacher-scale diagnostic on the A800 two-GPU node, intended as Section 3 robustness evidence rather than a main training result.",
]
(root / "short_diag_summary.md").write_text("\n".join(lines))
print(root / "short_diag_summary.md")
PY

echo "Done: ${DIAG_DIR}"
