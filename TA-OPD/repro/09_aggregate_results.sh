#!/usr/bin/env bash
set -eo pipefail  # 不能用 -u：conda 内部 deactivate 脚本有 unbound variable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

echo "========================================="
echo " Step 9: Aggregate results"
echo "========================================="

RESULTS_DIR="${OUTPUT_ROOT}/results"
mkdir -p "${RESULTS_DIR}"

export PYTHONPATH="$(get_pythonpath):${PYTHONPATH:-}"

# ── 1. Collect fixed-context eval summaries ──────────────────────────────
echo "[1/4] Collecting fixed-context eval summaries..."

python3 << 'PYEOF'
import os, glob, json, csv

sweep_root = os.environ.get("OUTPUT_ROOT", "") + "/main_sweep"
results_dir = os.environ.get("OUTPUT_ROOT", "") + "/results"
os.makedirs(results_dir, exist_ok=True)

rows = []
for tag_dir in sorted(glob.glob(os.path.join(sweep_root, "k16_*"))):
    tag = os.path.basename(tag_dir)
    for run_dir in sorted(glob.glob(os.path.join(tag_dir, "*_max64_*"))):
        run_name = os.path.basename(run_dir)
        fc_metrics = os.path.join(run_dir, "fixed_context", "fixed_context_metrics.parquet")
        gain_summary = os.path.join(run_dir, "fixed_context", "gain", "regression_summary.csv")

        parts = run_name.split("_")
        mask = parts[0]
        seed_label = parts[-1]
        ratio = "unknown"
        for p in parts:
            if p.startswith("max64"):
                continue

        # Extract ratio from tag
        for p in tag.split("_"):
            if p.startswith("ratio"):
                ratio = p.replace("ratio", "0.")
                if ratio == "0.10": ratio = "0.10"
                elif ratio == "0.01": ratio = "0.01"
                elif ratio == "0.03": ratio = "0.03"
                elif ratio == "0.05": ratio = "0.05"

        entry = {"tag": tag, "mask": mask, "ratio": ratio, "seed": seed_label,
                 "has_metrics": os.path.exists(fc_metrics),
                 "has_gain": os.path.exists(gain_summary)}
        rows.append(entry)

out_csv = os.path.join(results_dir, "all_runs_summary.csv")
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["tag", "mask", "ratio", "seed", "has_metrics", "has_gain"])
    writer.writeheader()
    writer.writerows(rows)

print(f"  Written {len(rows)} entries to {out_csv}")

total = len(rows)
with_metrics = sum(1 for r in rows if r["has_metrics"])
with_gain = sum(1 for r in rows if r["has_gain"])
print(f"  With metrics: {with_metrics}/{total}")
print(f"  With gain analysis: {with_gain}/{total}")
PYEOF

# ── 2. Run aggregation scripts ───────────────────────────────────────────
echo "[2/4] Running aggregation scripts..."

# aggregate_tip_baselines.py
if [[ -f "${TOOLS_DIR}/aggregate_tip_baselines.py" ]]; then
  echo "  Running aggregate_tip_baselines.py..."
  python3 "${TOOLS_DIR}/aggregate_tip_baselines.py" 2>&1 || echo "  WARNING: aggregate_tip_baselines.py failed"
fi

# aggregate_budget_ratio_curve.py
if [[ -f "${TOOLS_DIR}/aggregate_budget_ratio_curve.py" ]]; then
  echo "  Running aggregate_budget_ratio_curve.py..."
  python3 "${TOOLS_DIR}/aggregate_budget_ratio_curve.py" 2>&1 || echo "  WARNING: aggregate_budget_ratio_curve.py failed"
fi

# aggregate_ratio005_multiseed.py
if [[ -f "${TOOLS_DIR}/aggregate_ratio005_multiseed.py" ]]; then
  echo "  Running aggregate_ratio005_multiseed.py..."
  python3 "${TOOLS_DIR}/aggregate_ratio005_multiseed.py" 2>&1 || echo "  WARNING: aggregate_ratio005_multiseed.py failed"
fi

# ── 3. Generate method comparison table ──────────────────────────────────
echo "[3/4] Generating method comparison table..."

python3 << 'PYEOF'
import os, glob, csv

results_dir = os.environ.get("OUTPUT_ROOT", "") + "/results"
sweep_root = os.environ.get("OUTPUT_ROOT", "") + "/main_sweep"

# Aggregate mean gain per (mask, ratio) across seeds
aggregates = {}
for tag_dir in sorted(glob.glob(os.path.join(sweep_root, "k16_*"))):
    tag = os.path.basename(tag_dir)
    for run_dir in sorted(glob.glob(os.path.join(tag_dir, "*_max64_*"))):
        gain_csv = os.path.join(run_dir, "fixed_context", "gain", "regression_summary.csv")
        if not os.path.exists(gain_csv):
            continue

        run_name = os.path.basename(run_dir)
        mask = run_name.split("_max64_")[0] if "_max64_" in run_name else run_name
        seed = run_name.split("_max64_")[-1] if "_max64_" in run_name else "unknown"

        # Extract ratio from tag
        ratio = "unknown"
        for part in tag.split("_"):
            if part.startswith("ratio"):
                r = part[5:]
                if len(r) >= 2:
                    ratio = r[0] + "." + r[1:]

        try:
            with open(gain_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (mask, ratio)
                    if key not in aggregates:
                        aggregates[key] = {"values": [], "seeds": []}
                    # Try to get mean gain
                    for col in ["mean_gain", "mean_G_KLf", "mean", "gain_mean"]:
                        if col in row:
                            try:
                                aggregates[key]["values"].append(float(row[col]))
                                aggregates[key]["seeds"].append(seed)
                            except (ValueError, TypeError):
                                pass
                            break
        except Exception:
            pass

out_csv = os.path.join(results_dir, "method_comparison.csv")
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mask", "ratio", "n_seeds", "mean_gain", "std_gain"])
    for (mask, ratio), data in sorted(aggregates.items()):
        vals = data["values"]
        if vals:
            import statistics
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0
            writer.writerow([mask, ratio, len(vals), f"{mean:.4f}", f"{std:.4f}"])

print(f"  Written method comparison to {out_csv}")
PYEOF

# ── 4. Summary ────────────────────────────────────────────────────────────
echo "[4/4] Summary..."
echo ""
echo "Generated files:"
for f in "${RESULTS_DIR}"/*.csv "${RESULTS_DIR}"/*.png; do
  [[ -f "$f" ]] && echo "  $(basename "$f") ($(wc -l < "$f") lines)"
done

echo ""
echo "========================================="
echo " Result aggregation complete!"
echo " Output: ${RESULTS_DIR}"
echo "========================================="
