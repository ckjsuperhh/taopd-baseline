#!/usr/bin/env python3
"""Aggregate P1/P2 TIP-baseline OPD runs.

The script is intentionally tolerant of missing runs so it can be executed
while a long baseline queue is still running.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("/path/to/outputs/slime_opd")
OUT_DIR = BASE / "analysis" / "p1_p2_tip_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)


ENTRIES = [
    # Existing ratio=0.03, three-seed intervention runs.
    (1, "dlearn_high", 0.03, "qwen3_1_7b_dapo_budget_k16_ratio003_dlearn_high_max64_seed1_20260511_113941", "budget_common_context_k16_ratio003_seed1_20260511/dlearn_high_max64_seed1"),
    (2, "dlearn_high", 0.03, "qwen3_1_7b_dapo_budget_k16_ratio003_dlearn_high_max64_seed2_20260511_133848", "budget_common_context_k16_ratio003_seed2_20260511/dlearn_high_max64_seed2"),
    (3, "dlearn_high", 0.03, "qwen3_1_7b_dapo_budget_k16_ratio003_dlearn_high_max64_seed3_20260511_145028", "budget_common_context_k16_ratio003_seed3_20260511/dlearn_high_max64_seed3"),
    (1, "q3_highc", 0.03, "qwen3_1_7b_dapo_budget_k16_ratio003_q3_highc_seed1_20260511_083756", "budget_common_context_k16_ratio003_seed1_20260511/q3_highc_seed1"),
    (2, "q3_highc", 0.03, "qwen3_1_7b_dapo_budget_k16_ratio003_q3_highc_max64_seed2_20260511_132130", "budget_common_context_k16_ratio003_seed2_20260511/q3_highc_max64_seed2"),
    (3, "q3_highc", 0.03, "qwen3_1_7b_dapo_budget_k16_ratio003_q3_highc_max64_seed3_20260511_143257", "budget_common_context_k16_ratio003_seed3_20260511/q3_highc_max64_seed3"),
    (1, "q3_lowc", 0.03, "qwen3_1_7b_dapo_budget_k16_ratio003_q3_lowc_seed1_20260511_090304", "budget_common_context_k16_ratio003_seed1_20260511/q3_lowc_seed1"),
    (2, "q3_lowc", 0.03, "qwen3_1_7b_dapo_budget_k16_ratio003_q3_lowc_max64_seed2_20260511_130244", "budget_common_context_k16_ratio003_seed2_20260511/q3_lowc_max64_seed2"),
    (3, "q3_lowc", 0.03, "qwen3_1_7b_dapo_budget_k16_ratio003_q3_lowc_max64_seed3_20260511_141441", "budget_common_context_k16_ratio003_seed3_20260511/q3_lowc_max64_seed3"),
    (1, "dincompat_high", 0.03, "qwen3_1_7b_dapo_budget_k16_ratio003_dincompat_high_max64_seed1_20260511_120701", "budget_common_context_k16_ratio003_seed1_20260511/dincompat_high_max64_seed1"),
    (2, "dincompat_high", 0.03, "qwen3_1_7b_dapo_budget_k16_ratio003_dincompat_high_max64_seed2_20260511_135637", "budget_common_context_k16_ratio003_seed2_20260511/dincompat_high_max64_seed2"),
    (3, "dincompat_high", 0.03, "qwen3_1_7b_dapo_budget_k16_ratio003_dincompat_high_max64_seed3_20260511_151121", "budget_common_context_k16_ratio003_seed3_20260511/dincompat_high_max64_seed3"),
    # Existing ratio=0.05 main comparison.
    (1, "dlearn_high", 0.05, "qwen3_1_7b_dapo_budget_k16_ratio005_dlearn_high_max64_seed1_budget_ratio_sweep_k16_seed1_20260512_resume_v1", "budget_ratio_sweep_k16_seed1_20260512_resume_v1/dlearn_high_ratio005"),
    (2, "dlearn_high", 0.05, "qwen3_1_7b_dapo_budget_k16_ratio005_dlearn_high_max64_seed2_budget_ratio005_dlearn_seed2_cluster_20260512", "budget_ratio005_dlearn_seed2_cluster_20260512/dlearn_high_ratio005"),
    (3, "dlearn_high", 0.05, "qwen3_1_7b_dapo_budget_k16_ratio005_dlearn_high_max64_seed3_budget_ratio005_dlearn_seed3_cluster_20260512", "budget_ratio005_dlearn_seed3_cluster_20260512/dlearn_high_ratio005"),
    (1, "q3_highc", 0.05, "qwen3_1_7b_dapo_budget_k16_ratio005_q3_highc_max64_seed1_budget_ratio_sweep_k16_seed1_20260512_resume_v1", "budget_ratio_sweep_k16_seed1_20260512_resume_v1/q3_highc_ratio005"),
    (2, "q3_highc", 0.05, "qwen3_1_7b_dapo_budget_k16_ratio005_q3_highc_max64_seed2_budget_ratio005_q3_seed2_cluster_20260512", "budget_ratio005_q3_seed2_cluster_20260512/q3_highc_ratio005"),
    (3, "q3_highc", 0.05, "qwen3_1_7b_dapo_budget_k16_ratio005_q3_highc_max64_seed3_budget_ratio005_q3_seed3_cluster_20260512", "budget_ratio005_q3_seed3_cluster_20260512/q3_highc_ratio005"),
]


P1_TAG = "p1_tip_baselines_k16_seed1_cluster_20260513"
for mask in ["random", "divergence", "tip", "ca_softor", "q3"]:
    for ratio, tag in [(0.03, "003"), (0.05, "005")]:
        ENTRIES.append(
            (
                1,
                mask,
                ratio,
                f"qwen3_1_7b_dapo_budget_k16_ratio{tag}_{mask}_max64_seed1_{P1_TAG}",
                f"{P1_TAG}/{mask}_ratio{tag}",
            )
        )

ENTROPY_TAG = "p1_entropy_k16_seed1_cluster_20260513"
for ratio, tag in [(0.03, "003"), (0.05, "005")]:
    ENTRIES.append(
        (
            1,
            "entropy",
            ratio,
            f"qwen3_1_7b_dapo_budget_k16_ratio{tag}_entropy_max64_seed1_{ENTROPY_TAG}",
            f"{ENTROPY_TAG}/entropy_ratio{tag}",
        )
    )

P1_SEED2_TAG = "p1_tip_baselines_seed2_followup_20260513"
for mask in ["random", "divergence", "tip", "ca_softor", "q3", "entropy"]:
    for ratio, tag in [(0.03, "003"), (0.05, "005")]:
        ENTRIES.append(
            (
                2,
                mask,
                ratio,
                f"qwen3_1_7b_dapo_budget_k16_ratio{tag}_{mask}_max64_seed2_{P1_SEED2_TAG}",
                f"{P1_SEED2_TAG}/{mask}_ratio{tag}",
            )
        )

P1_SEED3_TAG = "p1_tip_baselines_seed3_followup_20260514"
for mask in ["random", "divergence", "tip", "ca_softor", "q3", "entropy"]:
    for ratio, tag in [(0.03, "003"), (0.05, "005")]:
        ENTRIES.append(
            (
                3,
                mask,
                ratio,
                f"qwen3_1_7b_dapo_budget_k16_ratio{tag}_{mask}_max64_seed3_{P1_SEED3_TAG}",
                f"{P1_SEED3_TAG}/{mask}_ratio{tag}",
            )
        )


def _read_gain(eval_dir: Path) -> pd.Series:
    return pd.read_csv(eval_dir / "gain" / "q3_bootstrap_matching_summary.csv").iloc[0]


def _read_keep(run_dir: Path) -> tuple[int, int, float, float, float, float]:
    merged = run_dir / "token_bank" / "merged.parquet"
    if merged.exists():
        tb = pd.read_parquet(merged)
    else:
        csvs = sorted((run_dir / "token_bank").glob("rollout_*.csv"))
        if not csvs:
            raise FileNotFoundError(run_dir / "token_bank")
        tb = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    valid = tb[tb["loss_mask_original"].astype(int) == 1] if "loss_mask_original" in tb.columns else tb
    kept = valid[valid["budget_keep"].astype(int) == 1] if "budget_keep" in valid.columns else valid
    selected_q3_frac = float((kept["quadrant"].astype(str) == "Q3_lowH_highD").mean()) if len(kept) and "quadrant" in kept.columns else float("nan")
    selected_dlearn = float(pd.to_numeric(kept.get("Dlearn", pd.Series(dtype=float)), errors="coerce").mean()) if len(kept) else float("nan")
    selected_dincompat = float(pd.to_numeric(kept.get("Dincompat", pd.Series(dtype=float)), errors="coerce").mean()) if len(kept) else float("nan")
    return int(len(valid)), int(len(kept)), float(len(kept) / max(len(valid), 1)), selected_q3_frac, selected_dlearn, selected_dincompat


def read_entry(seed: int, mask: str, ratio: float, run_rel: str, eval_rel: str) -> tuple[dict | None, dict | None]:
    run_dir = BASE / run_rel
    eval_dir = BASE / eval_rel
    try:
        gain = _read_gain(eval_dir)
        token_rows, selected_rows, keep_ratio, q3_frac, dlearn, dincompat = _read_keep(run_dir)
        metrics = pd.read_parquet(eval_dir / "fixed_context_metrics.parquet")
    except Exception as exc:
        return None, {
            "seed": seed,
            "mask": mask,
            "ratio": ratio,
            "run_dir": str(run_dir),
            "eval_dir": str(eval_dir),
            "missing_reason": repr(exc),
        }
    bootstrap = float(gain["bootstrap_mean_diff"])
    matching = float(gain["matching_mean_diff"])
    return {
        "seed": seed,
        "mask": mask,
        "ratio": ratio,
        "token_rows": token_rows,
        "selected_rows": selected_rows,
        "keep_ratio": keep_ratio,
        "gain_per_keep": bootstrap / keep_ratio if keep_ratio else float("nan"),
        "matching_gain_per_keep": matching / keep_ratio if keep_ratio else float("nan"),
        "selected_q3_frac": q3_frac,
        "selected_mean_Dlearn": dlearn,
        "selected_mean_Dincompat": dincompat,
        "mean_G_KLf": float(pd.to_numeric(metrics["G_KLf"], errors="coerce").mean()),
        "bootstrap_mean_diff": bootstrap,
        "bootstrap_ci_low": float(gain["bootstrap_ci_low"]),
        "bootstrap_ci_high": float(gain["bootstrap_ci_high"]),
        "matching_pairs": int(gain["matching_pairs"]),
        "matching_mean_diff": matching,
        "matching_median_diff": float(gain["matching_median_diff"]),
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
    }, None


def main() -> None:
    rows = []
    missing = []
    for entry in ENTRIES:
        row, miss = read_entry(*entry)
        if row is not None:
            rows.append(row)
        if miss is not None:
            missing.append(miss)
    detail = pd.DataFrame(rows).sort_values(["ratio", "mask", "seed"]).reset_index(drop=True)
    detail.to_csv(OUT_DIR / "p1_p2_tip_baseline_detail_current.csv", index=False)
    missing_path = OUT_DIR / "p1_p2_tip_baseline_missing_current.csv"
    if missing:
        pd.DataFrame(missing).to_csv(missing_path, index=False)
    else:
        pd.DataFrame(columns=["seed", "mask", "ratio", "run_dir", "eval_dir", "missing_reason"]).to_csv(
            missing_path,
            index=False,
        )

    agg = (
        detail.groupby(["mask", "ratio"])
        .agg(
            n=("seed", "count"),
            keep_ratio_mean=("keep_ratio", "mean"),
            keep_ratio_std=("keep_ratio", "std"),
            bootstrap_mean=("bootstrap_mean_diff", "mean"),
            bootstrap_std=("bootstrap_mean_diff", "std"),
            gain_per_keep_mean=("gain_per_keep", "mean"),
            gain_per_keep_std=("gain_per_keep", "std"),
            matching_mean=("matching_mean_diff", "mean"),
            selected_q3_frac_mean=("selected_q3_frac", "mean"),
            selected_mean_Dlearn=("selected_mean_Dlearn", "mean"),
            selected_mean_Dincompat=("selected_mean_Dincompat", "mean"),
        )
        .reset_index()
        .sort_values(["ratio", "bootstrap_mean"], ascending=[True, False])
    )
    agg.to_csv(OUT_DIR / "p1_p2_tip_baseline_aggregate_current.csv", index=False)
    print("DETAIL")
    print(detail[["seed", "mask", "ratio", "keep_ratio", "bootstrap_mean_diff", "gain_per_keep", "matching_mean_diff"]].to_string(index=False))
    print("\nAGG")
    print(agg.to_string(index=False))
    if missing:
        print(f"\nMissing entries: {len(missing)} -> {OUT_DIR / 'p1_p2_tip_baseline_missing_current.csv'}")
    print(f"\nOUT_DIR={OUT_DIR}")


if __name__ == "__main__":
    main()
