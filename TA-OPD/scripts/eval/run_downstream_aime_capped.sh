#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/hf_cache
export HF_HUB_CACHE=/path/to/hf_cache/hub
export TRANSFORMERS_CACHE=/path/to/hf_cache/transformers
export HF_DATASETS_CACHE=/path/to/hf_cache/datasets

GPU_ID="${GPU_ID:-1}"
RUN_DIR="${RUN_DIR:-/path/to/outputs/slime_opd/analysis/downstream_smoke_20260515/aime24_25_capped4096}"
LOG_DIR="/path/to/outputs/slime_opd/logs"
LOG_FILE="${LOG_DIR}/downstream_aime24_25_capped4096_20260515.log"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate lmeval

declare -A MODELS=(
  [base_qwen3_1p7b]="/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B"
  [dlearn_high_ratio003_seed2]="/path/to/outputs/slime_opd/budget_common_context_k16_ratio003_seed2_20260511/dlearn_high_max64_seed2/student_hf"
  [q3_highc_ratio003_seed2]="/path/to/outputs/slime_opd/budget_common_context_k16_ratio003_seed2_20260511/q3_highc_max64_seed2/student_hf"
  [tip_ratio003_seed2]="/path/to/outputs/slime_opd/p1_tip_baselines_seed2_followup_20260513/tip_ratio003/student_hf"
)

echo "[$(date '+%F %T')] QUEUE_START run_dir=${RUN_DIR} gpu=${GPU_ID}" | tee -a "${LOG_FILE}"

for NAME in base_qwen3_1p7b dlearn_high_ratio003_seed2 q3_highc_ratio003_seed2 tip_ratio003_seed2; do
  MODEL="${MODELS[$NAME]}"
  OUT="${RUN_DIR}/${NAME}"
  mkdir -p "${OUT}"
  echo "[$(date '+%F %T')] START name=${NAME} model=${MODEL}" | tee -a "${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" timeout 240m lm-eval run \
    --model hf \
    --model_args "pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True" \
    --tasks aime24,aime25 \
    --gen_kwargs max_gen_toks=4096 \
    --device cuda:0 \
    --batch_size 1 \
    --output_path "${OUT}" 2>&1 | tee -a "${LOG_FILE}"
  echo "[$(date '+%F %T')] DONE name=${NAME}" | tee -a "${LOG_FILE}"
done

python3 - <<'PY'
from pathlib import Path
import csv
import json

root = Path("/path/to/outputs/slime_opd/analysis/downstream_smoke_20260515/aime24_25_capped4096")
rows = []
for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    files = sorted(model_dir.rglob("results_*.json"))
    if not files:
        continue
    path = files[-1]
    data = json.loads(path.read_text())
    results = data.get("results", {})
    for task in ("aime24", "aime25"):
        res = results.get(task, {})
        rows.append({
            "model": model_dir.name,
            "task": task,
            "exact_match": res.get("exact_match"),
            "exact_match_stderr": res.get("exact_match_stderr"),
            "result_json": str(path),
        })

base = {}
for r in rows:
    if r["model"] == "base_qwen3_1p7b":
        base[r["task"]] = r.get("exact_match")
for r in rows:
    try:
        r["exact_match_delta_vs_base"] = float(r["exact_match"]) - float(base[r["task"]])
    except Exception:
        r["exact_match_delta_vs_base"] = ""

out_csv = root / "aime24_25_capped4096_summary.csv"
out_md = root / "aime24_25_capped4096_summary.md"
if rows:
    fields = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(x):
        if x in ("", None):
            return ""
        return f"{float(x):.4f}"

    lines = [
        "# AIME24/25 Capped-4096 Eval",
        "",
        "| model | task | exact match | delta vs base |",
        "| --- | --- | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['task']} | {fmt(r['exact_match'])} | {fmt(r['exact_match_delta_vs_base'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(out_md.read_text())
else:
    print("No AIME result JSONs found.")
PY

echo "[$(date '+%F %T')] QUEUE_DONE run_dir=${RUN_DIR}" | tee -a "${LOG_FILE}"
