#!/usr/bin/env bash
set -eo pipefail  # 不能用 -u：conda 内部 deactivate 脚本有 unbound variable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

echo "========================================="
echo " Step 4: Prepare training & eval data"
echo "========================================="

DAPO_DIR="${DATA_DIR}/DAPO-Math-17k-dedup"
GSM8K_DIR="${DATA_DIR}/GSM8K-COT"
mkdir -p "${DAPO_DIR}" "${GSM8K_DIR}"

# ── 1. Download and process DAPO-Math-17k ─────────────────────────────────
echo "[1/4] Downloading DAPO-Math-17k..."
RAW_DIR="${DATA_DIR}/dapo_math_17k_raw"
mkdir -p "${RAW_DIR}"

# Python heredoc 用 os.environ.get 读这些变量, 必须 export
export DAPO_DIR RAW_DIR GSM8K_DIR

python3 << 'PYEOF'
import json, os, sys
from datasets import load_dataset

RAW_DIR = os.environ.get("RAW_DIR", "")
DAPO_DIR = os.environ.get("DAPO_DIR", "")

raw_path = os.path.join(DAPO_DIR, "dapo_math_17k_raw.jsonl")
slime_path = os.path.join(DAPO_DIR, "dapo_math_17k_dedup_slime.jsonl")

if os.path.exists(slime_path):
    print(f"  Slime-format data already exists at {slime_path}, skipping.")
    sys.exit(0)

print("  Loading DAPO-Math-17k from HuggingFace...")
ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")

def transform(example):
    prompt = example["prompt"][0]["content"] if example.get("prompt") else None
    label = example["reward_model"]["ground_truth"] if example.get("reward_model") else None
    return {"prompt": prompt, "label": label}

print("  Transforming to {prompt, label} format...")
ds2 = ds.map(transform, remove_columns=ds.column_names)
ds2 = ds2.filter(lambda x: x["prompt"] is not None and x["label"] is not None)

print(f"  Total samples after transform: {len(ds2)}")

print("  Deduplicating by prompt text...")
seen = set()
deduped = []
for example in ds2:
    key = example["prompt"].strip()
    if key not in seen:
        seen.add(key)
        deduped.append(example)

print(f"  Samples after dedup: {len(deduped)}")

os.makedirs(DAPO_DIR, exist_ok=True)
with open(slime_path, "w") as f:
    for item in deduped:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"  Written to {slime_path}")
print(f"  Total deduplicated samples: {len(deduped)}")
PYEOF

# ── 2. Generate heldout-300 split ────────────────────────────────────────
echo "[2/4] Generating heldout-300 split (seed=${DIAG_HELDOUT_SEED})..."

if [[ -f "${HELDOUT_DATA}" ]]; then
  echo "  Heldout data already exists at ${HELDOUT_DATA}, skipping."
else
  python3 << PYEOF
import json, random, os

slime_path = "${PROMPT_DATA}"
heldout_path = "${HELDOUT_DATA}"
seed = ${DIAG_HELDOUT_SEED}

with open(slime_path) as f:
    all_data = [json.loads(line) for line in f if line.strip()]

n = len(all_data)
half = n // 2
pool = all_data[half:]
print(f"  Total samples: {n}, latter-half pool: {len(pool)}")

rng = random.Random(seed)
selected_indices = sorted(rng.sample(range(len(pool)), min(400, len(pool))))
selected = [pool[i] for i in selected_indices]

os.makedirs(os.path.dirname(heldout_path), exist_ok=True)
with open(heldout_path, "w") as f:
    for item in selected:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

idx_path = heldout_path.replace(".jsonl", ".indices.json")
with open(idx_path, "w") as f:
    json.dump({"seed": seed, "pool_size": len(pool), "n_selected": len(selected),
               "indices": selected_indices}, f)

print(f"  Written {len(selected)} heldout samples to {heldout_path}")
PYEOF
fi

# ── 3. Prepare GSM8K-COT 300 samples ────────────────────────────────────
echo "[3/4] Preparing GSM8K-COT 300 samples (seed=${DIAG_HELDOUT_SEED})..."

if [[ -f "${GSM8K_DATA}" ]]; then
  echo "  GSM8K data already exists at ${GSM8K_DATA}, skipping."
else
  python3 << PYEOF
import json, random, os

gsm8k_path = "${GSM8K_DATA}"
seed = ${DIAG_HELDOUT_SEED}

# Download GSM8K from HuggingFace and create COT-format prompts
from datasets import load_dataset

print("  Downloading GSM8K from HuggingFace...")
ds = load_dataset("gsm8k", "main", split="test")

def make_cot_prompt(question):
    return (
        "Solve the following math problem step by step. "
        "Put your final answer in \\boxed{{}}.\n\n"
        f"Problem: {question}\n\nSolution:"
    )

samples = []
seen = set()
for i, example in enumerate(ds):
    q = example["question"].strip()
    if q in seen:
        continue
    seen.add(q)
    answer = example["answer"]
    gt = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
    samples.append({
        "prompt": make_cot_prompt(q),
        "label": gt,
        "doc_id": i,
    })

rng = random.Random(seed)
selected_indices = sorted(rng.sample(range(len(samples)), min(300, len(samples))))
selected = [samples[i] for i in selected_indices]

os.makedirs(os.path.dirname(gsm8k_path), exist_ok=True)
with open(gsm8k_path, "w") as f:
    for item in selected:
        out = {"prompt": item["prompt"], "label": item["label"]}
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"  Written {len(selected)} GSM8K-COT samples to {gsm8k_path}")
PYEOF
fi

# ── 4. Verify ─────────────────────────────────────────────────────────────
echo "[4/4] Verifying data files..."
echo ""
for f in "${PROMPT_DATA}" "${HELDOUT_DATA}" "${GSM8K_DATA}"; do
  if [[ -f "$f" ]]; then
    n=$(wc -l < "$f")
    echo "  OK  ${f} (${n} lines)"
  else
    echo "  MISSING  ${f}"
  fi
done

echo ""
echo "========================================="
echo " Data preparation complete!"
echo "========================================="
