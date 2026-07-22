#!/usr/bin/env python3
"""Plot budget-accuracy curves from an experiment summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _sem(s: pd.Series) -> float:
    if len(s) <= 1:
        return 0.0
    return float(s.std(ddof=1) / (len(s) ** 0.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV/parquet with method, budget, metric columns.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-col", default="method")
    parser.add_argument("--budget-col", default="budget_ratio")
    parser.add_argument("--metric-col", default="score")
    parser.add_argument("--seed-col", default="seed")
    parser.add_argument("--title", default="Budget-accuracy curve")
    args = parser.parse_args()

    path = Path(args.input)
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    for col in [args.method_col, args.budget_col, args.metric_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {path}")

    summary = (
        df.groupby([args.method_col, args.budget_col], dropna=False)[args.metric_col]
        .agg(["mean", "std", "count", _sem])
        .reset_index()
        .rename(columns={"_sem": "sem"})
    )
    summary["ci95"] = 1.96 * summary["sem"]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "budget_accuracy_summary.csv", index=False)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(summary.to_string(index=False))
        print(f"saved={outdir / 'budget_accuracy_summary.csv'}")
        return

    plt.figure(figsize=(7, 4.5))
    for method, sub in summary.groupby(args.method_col):
        sub = sub.sort_values(args.budget_col)
        plt.errorbar(
            sub[args.budget_col],
            sub["mean"],
            yerr=sub["ci95"],
            marker="o",
            capsize=3,
            label=str(method),
        )
    plt.xlabel(args.budget_col)
    plt.ylabel(args.metric_col)
    plt.title(args.title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "budget_accuracy_curve.png", dpi=180)
    print(summary.to_string(index=False))
    print(f"saved={outdir}")


if __name__ == "__main__":
    main()
