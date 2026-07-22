#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/slime_opd}"
STORY_ROOT="${STORY_ROOT:-${OUTPUT_ROOT}/analysis/scale_context_robustness_20260517}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/path/to/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/path/to/hf_cache/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/path/to/hf_cache/hub}"
export RAY_DISABLE_DOCKER_CPU_WARNING=1

PY="${PY:-python3}"
TEACHER_MODEL="${TEACHER_MODEL:-/path/to/models/Qwen3/4B}"
STUDENT_HF="${STUDENT_HF:-/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B}"
GSM8K_DATA="${GSM8K_DATA:-${SLIME_DIR}/data/GSM8K-COT/gsm8k_cot_slime_300_seed41717.jsonl}"
RUN_NAME="${RUN_NAME:-qwen3_1_7b_student_qwen3_4b_teacher_gsm8k300_k16_20260517}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
DIAG_DIR="${DIAG_DIR:-${STORY_ROOT}/gsm8k_4b_to_1p7b_300}"
DRIVER_LOG="${LOG_DIR}/${RUN_NAME}_driver.log"

mkdir -p "${LOG_DIR}" "${DIAG_DIR}" "$(dirname "${GSM8K_DATA}")"
cd "${SLIME_DIR}"

if [[ ! -s "${GSM8K_DATA}" ]]; then
  echo "Creating deterministic GSM8K-COT prompt shard..."
  GSM8K_DATA="${GSM8K_DATA}" "${PY}" - <<'PY'
import json
import os
import random
from pathlib import Path

source = Path(os.environ.get("GSM8K_SOURCE_JSONL", "/path/to/gsm8k_cot_samples.jsonl"))
out = Path(os.environ["GSM8K_DATA"])
rows = []
seen = set()
for line in source.open():
    obj = json.loads(line)
    doc_id = obj.get("doc_id")
    if doc_id in seen:
        continue
    seen.add(doc_id)
    prompt = obj.get("arguments", {}).get("gen_args_0", {}).get("arg_0")
    if not prompt:
        question = obj.get("doc", {}).get("question", "")
        prompt = f"Solve the following grade-school math problem.\n\n{question}"
    rows.append({"prompt": prompt, "source": "gsm8k_cot_lmeval_log", "doc_id": doc_id})
if len(rows) < 300:
    raise SystemExit(f"not enough GSM8K rows: {len(rows)}")
rng = random.Random(41717)
chosen = rng.sample(rows, 300)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in chosen) + "\n")
out.with_suffix(".meta.json").write_text(json.dumps({"source": str(source), "seed": 41717, "n_written": len(chosen)}, indent=2))
print(f"wrote {out} with {len(chosen)} prompts")
PY
fi

echo "=== 4B -> 1.7B GSM8K-COT 300-context fixed-context diagnostic ==="
echo "RUN_NAME=${RUN_NAME}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "DIAG_DIR=${DIAG_DIR}"
echo "GSM8K_DATA=${GSM8K_DATA}"
echo "DRIVER_LOG=${DRIVER_LOG}"

TEACHER_MODEL="${TEACHER_MODEL}" \
STUDENT_HF="${STUDENT_HF}" \
RUN_NAME="${RUN_NAME}" \
SAVE_DIR="${SAVE_DIR}" \
TEACHER_GPU=0 \
RAY_GPUS=1 \
COLOCATE=1 \
ACTOR_NUM_GPUS_PER_NODE=1 \
ROLLOUT_NUM_GPUS=1 \
TEACHER_PORT=13331 \
TEACHER_NCCL_PORT=23331 \
RAY_PORT=26531 \
RAY_DASHBOARD_PORT=8431 \
RAY_OBJECT_MANAGER_PORT=28731 \
RAY_NODE_MANAGER_PORT=28732 \
RAY_DASHBOARD_AGENT_LISTEN_PORT=28733 \
RAY_DASHBOARD_AGENT_GRPC_PORT=28734 \
RAY_METRICS_EXPORT_PORT=28735 \
RAY_TEMP_DIR=/tmp/slime_ray_26531 \
DATA_DIR="$(dirname "${GSM8K_DATA}")" \
PROMPT_DATA="${GSM8K_DATA}" \
NUM_ROLLOUT=151 \
ROLLOUT_BATCH_SIZE=2 \
N_SAMPLES_PER_PROMPT=1 \
GLOBAL_BATCH_SIZE=2 \
ROLLOUT_MAX_RESPONSE_LEN=256 \
MICRO_BATCH_SIZE=1 \
SAVE_INTERVAL=25 \
TEACHER_MEM_FRACTION=0.54 \
TEACHER_CUDA_GRAPH_MAX_BS=2 \
ROLLOUT_MEM_FRACTION=0.22 \
SGLANG_CUDA_GRAPH_MAX_BS=2 \
OPD_TOPK_METRICS_K=16 \
OPD_TOKEN_BANK_RAW_TOPK=1 \
OPD_TOKEN_BANK_PAIR_ID=qwen3_4b_to_qwen3_1p7b_gsm8k300 \
OPD_EXACT_CMASS=1 \
OPD_EXACT_CMASS_MAX_UNION=128 \
SAVE_DEBUG_ROLLOUT_DATA=1 \
SEED=41731 \
ROLLOUT_SEED=41731 \
OPD_BUDGET_MASK_SEED=41731 \
bash examples/on_policy_distillation/run-qwen3-1.7B-sampled-opd-sglang.sh \
  > "${DRIVER_LOG}" 2>&1

export PYTHONPATH="/path/to/Megatron-LM:${SLIME_DIR}:/path/to/miniconda3/envs/verl/lib/python3.10/site-packages:${PYTHONPATH:-}"

"${PY}" tools/export_fixed_context_bank.py \
  --debug-rollout-data "${SAVE_DIR}/debug_rollout_data/*.pt" \
  --output "${DIAG_DIR}/context_bank.parquet" \
  --max-samples 300

"${PY}" tools/eval_fixed_context_bank.py \
  --context-bank "${DIAG_DIR}/context_bank.parquet" \
  --student "${STUDENT_HF}" \
  --teacher "${TEACHER_MODEL}" \
  --output "${DIAG_DIR}/theta0_metrics.parquet" \
  --student-device cuda:0 \
  --teacher-device cuda:1 \
  --dtype bfloat16 \
  --topk 16 \
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
  --topk 16 \
  --max-samples 300 \
  --max-response-tokens 192 \
  --force-convert

"${PY}" tools/analyze_fixed_context_gain.py \
  --input "${DIAG_DIR}/eval_latest/fixed_context_metrics.parquet" \
  --output-dir "${DIAG_DIR}/analysis/gain" \
  --bootstrap 1000

DIAG_DIR="${DIAG_DIR}" SAVE_DIR="${SAVE_DIR}" "${PY}" - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ["DIAG_DIR"])
ctx = pd.read_parquet(root / "context_bank.parquet")
metrics = pd.read_parquet(root / "eval_latest/fixed_context_metrics.parquet")
q3 = pd.read_csv(root / "analysis/gain/q3_bootstrap_matching_summary.csv").iloc[0]
reg = pd.read_csv(root / "analysis/gain/regression_summary.csv")
lines = [
    "# 4B -> 1.7B GSM8K-COT 300-context diagnostic",
    "",
    f"Source run: `{os.environ['SAVE_DIR']}`.",
    f"Context bank: {len(ctx)} samples; eval metrics: {len(metrics)} token rows.",
    f"Q3 bootstrap high-minus-low mean diff: {q3.bootstrap_mean_diff:.6f} [{q3.bootstrap_ci_low:.6f}, {q3.bootstrap_ci_high:.6f}]; matching mean diff: {q3.matching_mean_diff:.6f} over {int(q3.matching_pairs)} pairs.",
    "",
    "Regression summary:",
    reg.to_markdown(index=False),
]
(root / "gsm8k300_summary.md").write_text("\n".join(lines) + "\n")
print(root / "gsm8k300_summary.md")
PY

"${PY}" tools/build_section3_analysis.py || true
