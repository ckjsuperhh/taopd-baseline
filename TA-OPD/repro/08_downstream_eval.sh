#!/usr/bin/env bash
set -eo pipefail  # 不能用 -u：conda 内部 deactivate 脚本有 unbound variable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

echo "========================================="
echo " Step 8: Downstream evaluation"
echo "========================================="

DOWNSTREAM_ROOT="${OUTPUT_ROOT}/downstream"
mkdir -p "${DOWNSTREAM_ROOT}"

# Select representative models for downstream eval
# seed2 at ratio 0.03 (matches paper's main comparison)
SEED2_TAG="k16_ratio003_seed2_${DATE_TAG}"
SWEEP_ROOT="${OUTPUT_ROOT}/main_sweep"

declare -A MODELS
MODELS=(
  ["base_qwen3_1p7b"]="${STUDENT_HF}"
  ["dlearn_high_ratio003_seed2"]="${SWEEP_ROOT}/${SEED2_TAG}/dlearn_high_max64_seed2/student_hf"
  ["q3_highc_ratio003_seed2"]="${SWEEP_ROOT}/${SEED2_TAG}/q3_highc_max64_seed2/student_hf"
  ["tip_ratio003_seed2"]="${SWEEP_ROOT}/${SEED2_TAG}/tip_max64_seed2/student_hf"
)

# Fallback: try to find student_hf from any seed2 run
for key in "${!MODELS[@]}"; do
  if [[ "${key}" == "base_qwen3_1p7b" ]]; then continue; fi
  path="${MODELS[$key]}"
  if [[ ! -d "${path}" ]]; then
    found=$(find "${SWEEP_ROOT}" -path "*${key/_ratio003_seed2/}*seed2*/student_hf" -type d 2>/dev/null | head -1 || true)
    if [[ -n "${found}" ]]; then
      MODELS["${key}"]="${found}"
      echo "  Found ${key} at: ${found}"
    else
      echo "  WARNING: ${key} not found at ${path}"
    fi
  fi
done

# ── 1. GSM8K-COT ────────────────────────────────────────────────────────
echo ""
echo "[1/3] GSM8K-COT evaluation..."
GSM8K_OUT="${DOWNSTREAM_ROOT}/gsm8k_cot"
mkdir -p "${GSM8K_OUT}"
EVAL_GPU="${EVAL_GPU:-6}"

for model_name in "${!MODELS[@]}"; do
  model_path="${MODELS[${model_name}]}"
  done_flag="${GSM8K_OUT}/${model_name}/DONE"

  if [[ -f "${done_flag}" ]]; then
    echo "  SKIP ${model_name} (already done)"
    continue
  fi
  if [[ ! -d "${model_path}" ]]; then
    echo "  SKIP ${model_name} (model not found: ${model_path})"
    continue
  fi

  echo "  Evaluating ${model_name}..."
  mkdir -p "${GSM8K_OUT}/${model_name}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" timeout 150m lm-eval run \
    --model hf \
    --model_args "pretrained=${model_path},dtype=bfloat16,trust_remote_code=True" \
    --tasks gsm8k_cot \
    --device cuda:0 \
    --batch_size 8 \
    --output_path "${GSM8K_OUT}/${model_name}" \
    --log_samples \
    && touch "${done_flag}" \
    || echo "  WARNING: ${model_name} GSM8K eval failed/timed out"
done

# ── 2. AIME24/25 ─────────────────────────────────────────────────────────
echo ""
echo "[2/3] AIME24/25 evaluation..."
AIME_OUT="${DOWNSTREAM_ROOT}/aime24_25"
mkdir -p "${AIME_OUT}"

for model_name in "${!MODELS[@]}"; do
  model_path="${MODELS[${model_name}]}"
  done_flag="${AIME_OUT}/${model_name}/DONE"

  if [[ -f "${done_flag}" ]]; then
    echo "  SKIP ${model_name} (already done)"
    continue
  fi
  if [[ ! -d "${model_path}" ]]; then
    echo "  SKIP ${model_name} (model not found)"
    continue
  fi

  echo "  Evaluating ${model_name}..."
  mkdir -p "${AIME_OUT}/${model_name}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" timeout 240m lm-eval run \
    --model hf \
    --model_args "pretrained=${model_path},dtype=bfloat16,trust_remote_code=True" \
    --tasks aime24,aime25 \
    --gen_kwargs "max_gen_toks=4096" \
    --device cuda:0 \
    --batch_size 1 \
    --output_path "${AIME_OUT}/${model_name}" \
    --log_samples \
    && touch "${done_flag}" \
    || echo "  WARNING: ${model_name} AIME eval failed/timed out"
done

# ── 3. Collect results ───────────────────────────────────────────────────
echo ""
echo "[3/3] Collecting results..."

python3 << 'PYEOF'
import json, os, csv, glob

downstream_root = os.environ.get("DOWNSTREAM_ROOT", "")

def find_results(eval_dir):
    results = {}
    for model_dir in sorted(glob.glob(os.path.join(eval_dir, "*"))):
        model_name = os.path.basename(model_dir)
        for f in glob.glob(os.path.join(model_dir, "**", "results_*.json"), recursive=True):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                for task, metrics in data.get("results", {}).items():
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            results.setdefault(model_name, {})[f"{task}/{metric_name}"] = value
            except Exception:
                pass
    return results

# GSM8K
print("\n=== GSM8K-COT Results ===")
gsm8k_results = find_results(os.path.join(downstream_root, "gsm8k_cot"))
for model, metrics in sorted(gsm8k_results.items()):
    for k, v in sorted(metrics.items()):
        print(f"  {model}: {k} = {v:.4f}")

# AIME
print("\n=== AIME24/25 Results ===")
aime_results = find_results(os.path.join(downstream_root, "aime24_25"))
for model, metrics in sorted(aime_results.items()):
    for k, v in sorted(metrics.items()):
        print(f"  {model}: {k} = {v:.4f}")
PYEOF

echo ""
echo "========================================="
echo " Downstream evaluation complete!"
echo " Results: ${DOWNSTREAM_ROOT}"
echo "========================================="
