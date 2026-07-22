#!/usr/bin/env python3
"""Within-Q3 teachability analysis for sampled-token OPD.

This is a P0 analysis utility for the OPD teachability project.  It treats
TIP's Q3 (low student entropy, high teacher-student divergence) as the
conditioning set, then asks whether Dlearn still ranks token utility inside
that region.

The script intentionally builds Dlearn/Dincompat only from baseline/pre-update
teacher-student statistics.  The fixed-context gain column is used only as the
dependent variable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input suffix: {path}")


def _first(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing all candidate columns: {candidates}")


def _norm(s: pd.Series, q_low: float, q_high: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo = x.quantile(q_low)
    hi = x.quantile(q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.0)


def _winsor(s: pd.Series, low: float, high: float) -> pd.Series:
    if low <= 0 and high >= 1:
        return s
    lo = s.quantile(low)
    hi = s.quantile(high)
    return s.clip(lo, hi)


def _prepare(df_raw: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    gain_col = args.gain_col or _first(df_raw, ["G_KLf", "G_KLr"])
    h_col = args.h_col or _first(df_raw, ["Hs_full_base", "Hs_full", "H_norm", "Hs_topk_norm"])
    d_col = args.d_col or _first(df_raw, ["KLf_full_base", "KLf_full", "D_norm", "KLf_union"])
    c_col = args.c_col or _first(df_raw, ["Cmass_true", "Cmass", "C_norm", "CBC", "Coverlap"])
    group_col = args.group_col or _first(df_raw, ["sample_index", "group_index", "sample_ordinal"])
    pos_col = args.pos_col or _first(df_raw, ["pos_norm", "tok_pos"])

    df = df_raw.copy()
    df["G_raw"] = pd.to_numeric(df[gain_col], errors="coerce")
    df["G"] = _winsor(df["G_raw"], args.winsor_low, args.winsor_high)
    df["Hn"] = _norm(df[h_col], args.q_low, args.q_high)
    df["Dn"] = _norm(df[d_col], args.q_low, args.q_high)
    df["Cn"] = _norm(df[c_col], args.q_low, args.q_high)
    df["pos"] = _norm(df[pos_col], 0.0, 1.0)
    df["Dlearn"] = df["Dn"] * df["Cn"]
    df["Dincompat"] = df["Dn"] * (1.0 - df["Cn"])
    df["tip_score"] = df["Hn"] + df["Dn"] - df["Hn"] * df["Dn"]
    df["group"] = df[group_col]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["G", "Hn", "Dn", "Cn", "group"])

    h_cut = df["Hn"].quantile(args.h_quantile)
    d_cut = df["Dn"].quantile(args.d_quantile)
    df["is_q3"] = (df["Hn"] <= h_cut) & (df["Dn"] >= d_cut)
    meta = {
        "gain_col": gain_col,
        "h_col": h_col,
        "d_col": d_col,
        "c_col": c_col,
        "group_col": group_col,
        "pos_col": pos_col,
        "q3_rule": f"Hn <= q{args.h_quantile} and Dn >= q{args.d_quantile}",
        "winsor": f"{args.winsor_low},{args.winsor_high}",
    }
    return df, meta


def _cluster_mean_ci(df: pd.DataFrame, value_col: str, group_col: str, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    if df.empty:
        return np.nan, np.nan, np.nan
    mean = float(df[value_col].mean())
    groups = df[group_col].dropna().unique()
    if len(groups) <= 1 or n_boot <= 0:
        return mean, np.nan, np.nan
    by_group = {g: df[df[group_col] == g][value_col].to_numpy() for g in groups}
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        xs = np.concatenate([by_group[g] for g in sampled])
        vals.append(float(np.mean(xs)))
    return mean, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _high_low_ci(q3: pd.DataFrame, feature: str, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    lo_cut = q3[feature].quantile(0.25)
    hi_cut = q3[feature].quantile(0.75)
    low = q3[q3[feature] <= lo_cut]
    high = q3[q3[feature] >= hi_cut]
    diff = float(high["G"].mean() - low["G"].mean()) if len(low) and len(high) else np.nan
    groups = q3["group"].dropna().unique()
    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        b = pd.concat([q3[q3["group"] == g] for g in sampled], ignore_index=True)
        b_low = b[b[feature] <= lo_cut]
        b_high = b[b[feature] >= hi_cut]
        if len(b_low) and len(b_high):
            boot.append(float(b_high["G"].mean() - b_low["G"].mean()))
    return {
        "feature": feature,
        "low_cut": float(lo_cut),
        "high_cut": float(hi_cut),
        "low_tokens": int(len(low)),
        "high_tokens": int(len(high)),
        "mean_low": float(low["G"].mean()) if len(low) else np.nan,
        "mean_high": float(high["G"].mean()) if len(high) else np.nan,
        "high_minus_low": diff,
        "ci_low": float(np.quantile(boot, 0.025)) if boot else np.nan,
        "ci_high": float(np.quantile(boot, 0.975)) if boot else np.nan,
    }


def _bucket_summary(q3: pd.DataFrame, feature: str, k_value: int, rng: np.random.Generator, n_boot: int) -> list[dict[str, object]]:
    work = q3.copy()
    try:
        work["bucket"] = pd.qcut(work[feature], q=4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop")
    except ValueError:
        work["bucket"] = "all"
    rows = []
    for bucket, sub in work.groupby("bucket", observed=False):
        mean, ci_low, ci_high = _cluster_mean_ci(sub, "G", "group", rng, n_boot)
        rows.append(
            {
                "K": k_value,
                "feature": feature,
                "bucket": str(bucket),
                "tokens": int(len(sub)),
                "groups": int(sub["group"].nunique()),
                "mean_G": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_feature": float(sub[feature].mean()) if len(sub) else np.nan,
                "mean_Hn": float(sub["Hn"].mean()) if len(sub) else np.nan,
                "mean_Dn": float(sub["Dn"].mean()) if len(sub) else np.nan,
                "mean_Cn": float(sub["Cn"].mean()) if len(sub) else np.nan,
            }
        )
    return rows


def _plot_outputs(bucket_df: pd.DataFrame, diff_df: pd.DataFrame, outdir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plots: {exc}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.dpi": 300,
        }
    )
    colors = {"Dlearn": "#2E7D6B", "Dincompat": "#D28A3C", "Cn": "#8A5A99"}

    k16 = bucket_df[bucket_df["K"] == 16]
    if not k16.empty:
        fig, ax = plt.subplots(figsize=(7.2, 3.7))
        features = ["Dlearn", "Dincompat", "Cn"]
        x = np.arange(4)
        width = 0.24
        labels = ["Q1 low", "Q2", "Q3", "Q4 high"]
        for j, feat in enumerate(features):
            sub = k16[k16["feature"] == feat].set_index("bucket").reindex(labels)
            y = sub["mean_G"].to_numpy(dtype=float)
            lo = sub["ci_low"].to_numpy(dtype=float)
            hi = sub["ci_high"].to_numpy(dtype=float)
            yerr = np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)])
            ax.bar(x + (j - 1) * width, y, width=width, yerr=yerr, capsize=3, color=colors[feat], label=feat)
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean fixed-context gain (winsorized G_KLf)")
        ax.set_title("Within TIP-Q3, Dlearn ranks token utility (K=16)")
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(outdir / f"within_q3_bucket_gain_k16.{ext}", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    for feat in ["Dlearn", "Dincompat", "Cn"]:
        sub = diff_df[diff_df["feature"] == feat].sort_values("K")
        if sub.empty:
            continue
        xvals = sub["K"].to_numpy()
        y = sub["high_minus_low"].to_numpy(dtype=float)
        lo = sub["ci_low"].to_numpy(dtype=float)
        hi = sub["ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)])
        ax.errorbar(xvals, y, yerr=yerr, marker="o", capsize=4, linewidth=2, label=feat, color=colors[feat])
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(sorted(diff_df["K"].unique()))
    ax.set_xlabel("Top-K diagnostic support")
    ax.set_ylabel("Q4 high - Q1 low gain inside Q3")
    ax.set_title("Within-Q3 high-low bucket contrast")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"within_q3_high_low_diff_across_k.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="K:path pair, e.g. 16:/path/fixed_context_metrics.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gain-col", default=None)
    parser.add_argument("--h-col", default=None)
    parser.add_argument("--d-col", default=None)
    parser.add_argument("--c-col", default=None)
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--pos-col", default=None)
    parser.add_argument("--q-low", type=float, default=0.05)
    parser.add_argument("--q-high", type=float, default=0.95)
    parser.add_argument("--h-quantile", type=float, default=0.5)
    parser.add_argument("--d-quantile", type=float, default=0.5)
    parser.add_argument("--winsor-low", type=float, default=0.01)
    parser.add_argument("--winsor-high", type=float, default=0.99)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    bucket_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    meta: dict[str, object] = {"inputs": args.input, "per_input": {}}
    features = ["Dlearn", "Dincompat", "Cn"]

    for item in args.input:
        if ":" not in item:
            raise ValueError(f"--input must be K:path, got {item}")
        k_text, path_text = item.split(":", 1)
        k_value = int(k_text)
        path = Path(path_text)
        df_raw = _read(path)
        df, item_meta = _prepare(df_raw, args)
        q3 = df[df["is_q3"]].copy()
        meta["per_input"][str(k_value)] = {
            **item_meta,
            "path": str(path),
            "tokens": int(len(df)),
            "q3_tokens": int(len(q3)),
            "q3_groups": int(q3["group"].nunique()),
        }
        for feat in features:
            bucket_rows.extend(_bucket_summary(q3, feat, k_value, rng, args.bootstrap))
            row = _high_low_ci(q3, feat, rng, args.bootstrap)
            row["K"] = k_value
            row["q3_tokens"] = int(len(q3))
            row["q3_groups"] = int(q3["group"].nunique())
            diff_rows.append(row)

    bucket_df = pd.DataFrame(bucket_rows)
    diff_df = pd.DataFrame(diff_rows)
    bucket_df.to_csv(outdir / "within_q3_bucket_summary.csv", index=False)
    diff_df.to_csv(outdir / "within_q3_high_low_summary.csv", index=False)
    (outdir / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _plot_outputs(bucket_df, diff_df, outdir)

    print("WROTE", outdir)
    print(diff_df.sort_values(["feature", "K"]).to_string(index=False))


if __name__ == "__main__":
    main()
