from pathlib import Path

import pandas as pd


BASE = Path("/path/to/outputs/slime_opd")
OUT_DETAIL = BASE / "dapo_budget_ratio005_dlearn_vs_q3_seeds1_2_3_a800_detail_20260512.csv"
OUT_AGG = BASE / "dapo_budget_ratio005_dlearn_vs_q3_seeds1_2_3_a800_aggregate_20260512.csv"

ENTRIES = [
    (
        1,
        "dlearn_high",
        "qwen3_1_7b_dapo_budget_k16_ratio005_dlearn_high_max64_seed1_budget_ratio_sweep_k16_seed1_20260512_resume_v1",
        "budget_ratio_sweep_k16_seed1_20260512_resume_v1/dlearn_high_ratio005",
    ),
    (
        1,
        "q3_highc",
        "qwen3_1_7b_dapo_budget_k16_ratio005_q3_highc_max64_seed1_budget_ratio_sweep_k16_seed1_20260512_resume_v1",
        "budget_ratio_sweep_k16_seed1_20260512_resume_v1/q3_highc_ratio005",
    ),
    (
        2,
        "dlearn_high",
        "qwen3_1_7b_dapo_budget_k16_ratio005_dlearn_high_max64_seed2_budget_ratio005_dlearn_seed2_a800_20260512",
        "budget_ratio005_dlearn_seed2_a800_20260512/dlearn_high_ratio005",
    ),
    (
        2,
        "q3_highc",
        "qwen3_1_7b_dapo_budget_k16_ratio005_q3_highc_max64_seed2_budget_ratio005_q3_seed2_a800_20260512",
        "budget_ratio005_q3_seed2_a800_20260512/q3_highc_ratio005",
    ),
    (
        3,
        "dlearn_high",
        "qwen3_1_7b_dapo_budget_k16_ratio005_dlearn_high_max64_seed3_budget_ratio005_dlearn_seed3_a800_20260512",
        "budget_ratio005_dlearn_seed3_a800_20260512/dlearn_high_ratio005",
    ),
    (
        3,
        "q3_highc",
        "qwen3_1_7b_dapo_budget_k16_ratio005_q3_highc_max64_seed3_budget_ratio005_q3_seed3_a800_20260512",
        "budget_ratio005_q3_seed3_a800_20260512/q3_highc_ratio005",
    ),
]


def read_row(seed: int, mask: str, run_rel: str, eval_rel: str) -> dict:
    run_dir = BASE / run_rel
    eval_dir = BASE / eval_rel
    token_bank = pd.read_parquet(run_dir / "token_bank" / "merged.parquet")
    kept = token_bank[token_bank["budget_keep"].astype(bool)]
    gain = pd.read_csv(eval_dir / "gain" / "q3_bootstrap_matching_summary.csv").iloc[0]
    metrics = pd.read_parquet(eval_dir / "fixed_context_metrics.parquet")
    return {
        "seed": seed,
        "mask": mask,
        "ratio": 0.05,
        "token_rows": int(len(token_bank)),
        "keep_ratio": float(len(kept) / len(token_bank)),
        "selected_rows": int(len(kept)),
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


def main() -> None:
    detail = pd.DataFrame([read_row(*entry) for entry in ENTRIES])
    detail = detail.sort_values(["seed", "mask"]).reset_index(drop=True)
    detail.to_csv(OUT_DETAIL, index=False)

    agg = (
        detail.groupby("mask")
        .agg(
            n=("seed", "count"),
            keep_ratio_mean=("keep_ratio", "mean"),
            keep_ratio_std=("keep_ratio", "std"),
            bootstrap_mean=("bootstrap_mean_diff", "mean"),
            bootstrap_std=("bootstrap_mean_diff", "std"),
            matching_mean=("matching_mean_diff", "mean"),
            matching_std=("matching_mean_diff", "std"),
            mean_G_KLf_mean=("mean_G_KLf", "mean"),
            mean_G_KLf_std=("mean_G_KLf", "std"),
        )
        .reset_index()
    )
    agg.to_csv(OUT_AGG, index=False)

    print("DETAIL")
    print(
        detail[
            [
                "seed",
                "mask",
                "keep_ratio",
                "bootstrap_mean_diff",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "matching_mean_diff",
                "mean_G_KLf",
            ]
        ].to_string(index=False)
    )
    print("\nAGGREGATE")
    print(agg.to_string(index=False))
    print(f"\nDETAIL_CSV={OUT_DETAIL}")
    print(f"AGG_CSV={OUT_AGG}")


if __name__ == "__main__":
    main()
