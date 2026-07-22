from pathlib import Path

import pandas as pd


BASE = Path("/path/to/outputs/slime_opd")
OUT_CSV = BASE / "dapo_budget_ratio_curve_k16_seed1_summary_20260512.csv"
OUT_PNG = BASE / "dapo_budget_ratio_curve_k16_seed1_bootstrap_matching_20260512.png"

ENTRIES = [
    (
        "dlearn_high",
        0.01,
        "qwen3_1_7b_dapo_budget_k16_ratio001_dlearn_high_max64_seed1_budget_ratio_sweep_k16_seed1_20260512_v2",
        "budget_ratio_sweep_k16_seed1_20260512_v2/dlearn_high_ratio001",
    ),
    (
        "dlearn_high",
        0.03,
        "qwen3_1_7b_dapo_budget_k16_ratio003_dlearn_high_max64_seed1_20260511_113941",
        "budget_common_context_k16_ratio003_seed1_20260511/dlearn_high_max64_seed1",
    ),
    (
        "dlearn_high",
        0.05,
        "qwen3_1_7b_dapo_budget_k16_ratio005_dlearn_high_max64_seed1_budget_ratio_sweep_k16_seed1_20260512_resume_v1",
        "budget_ratio_sweep_k16_seed1_20260512_resume_v1/dlearn_high_ratio005",
    ),
    (
        "dlearn_high",
        0.10,
        "qwen3_1_7b_dapo_budget_k16_ratio010_dlearn_high_max64_seed1_budget_ratio_sweep_k16_seed1_20260512_resume_v1",
        "budget_ratio_sweep_k16_seed1_20260512_resume_v1/dlearn_high_ratio010",
    ),
    (
        "q3_highc",
        0.01,
        "qwen3_1_7b_dapo_budget_k16_ratio001_q3_highc_max64_seed1_budget_ratio_sweep_k16_seed1_20260512_resume_v1",
        "budget_ratio_sweep_k16_seed1_20260512_resume_v1/q3_highc_ratio001",
    ),
    (
        "q3_highc",
        0.03,
        "qwen3_1_7b_dapo_budget_k16_ratio003_q3_highc_seed1_20260511_083756",
        "budget_common_context_k16_ratio003_seed1_20260511/q3_highc_seed1",
    ),
    (
        "q3_highc",
        0.05,
        "qwen3_1_7b_dapo_budget_k16_ratio005_q3_highc_max64_seed1_budget_ratio_sweep_k16_seed1_20260512_resume_v1",
        "budget_ratio_sweep_k16_seed1_20260512_resume_v1/q3_highc_ratio005",
    ),
]


def read_row(mask: str, ratio: float, run_rel: str, eval_rel: str) -> dict:
    run_dir = BASE / run_rel
    eval_dir = BASE / eval_rel
    token_bank = pd.read_parquet(run_dir / "token_bank" / "merged.parquet")
    metrics = pd.read_parquet(eval_dir / "fixed_context_metrics.parquet")
    gain = pd.read_csv(eval_dir / "gain" / "q3_bootstrap_matching_summary.csv").iloc[0]
    kept = token_bank[token_bank["budget_keep"].astype(bool)]
    q = kept["quadrant"].astype(str) if "quadrant" in kept.columns else pd.Series([], dtype=str)
    return {
        "seed": 1,
        "mask": mask,
        "ratio": ratio,
        "token_rows": int(len(token_bank)),
        "keep_ratio": float(len(kept) / len(token_bank)),
        "selected_rows": int(len(kept)),
        "selected_q3_frac": float(q.str.startswith("Q3").mean()) if len(kept) else float("nan"),
        "mean_G_KLf": float(metrics["G_KLf"].mean()),
        "bootstrap_mean_diff": float(gain["bootstrap_mean_diff"]),
        "bootstrap_ci_low": float(gain["bootstrap_ci_low"]),
        "bootstrap_ci_high": float(gain["bootstrap_ci_high"]),
        "matching_pairs": int(gain["matching_pairs"]),
        "matching_mean_diff": float(gain["matching_mean_diff"]),
        "matching_median_diff": float(gain["matching_median_diff"]),
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
    }


def plot(df: pd.DataFrame) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        print("matplotlib is not installed; skip PNG plot.")
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    colors = {"dlearn_high": "#1f77b4", "q3_highc": "#d62728"}
    labels = {"dlearn_high": "Dlearn-high", "q3_highc": "Q3-highC"}
    for mask, sub in df.groupby("mask"):
        sub = sub.sort_values("ratio")
        axes[0].plot(
            sub["ratio"],
            sub["bootstrap_mean_diff"],
            marker="o",
            label=labels.get(mask, mask),
            color=colors.get(mask),
        )
        axes[0].fill_between(
            sub["ratio"],
            sub["bootstrap_ci_low"],
            sub["bootstrap_ci_high"],
            color=colors.get(mask),
            alpha=0.12,
        )
        axes[1].plot(
            sub["ratio"],
            sub["matching_mean_diff"],
            marker="o",
            label=labels.get(mask, mask),
            color=colors.get(mask),
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Prompt-cluster bootstrap")
    axes[1].set_title("Within-prompt matching")
    for ax in axes:
        ax.set_xlabel("Budget ratio")
        ax.set_ylabel("Q3 highC-lowC gain")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)


def main() -> None:
    df = pd.DataFrame([read_row(*entry) for entry in ENTRIES])
    df = df.sort_values(["mask", "ratio"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    plot(df)
    cols = [
        "mask",
        "ratio",
        "keep_ratio",
        "bootstrap_mean_diff",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "matching_mean_diff",
        "mean_G_KLf",
    ]
    print(df[cols].to_string(index=False))
    print(f"CSV={OUT_CSV}")
    print(f"PNG={OUT_PNG}")


if __name__ == "__main__":
    main()
