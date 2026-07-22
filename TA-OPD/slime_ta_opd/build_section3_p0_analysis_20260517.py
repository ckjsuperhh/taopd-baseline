#!/usr/bin/env python3
"""Build paper-ready Section 3 diagnostics for OPD Token Teachability.

The script reads fixed-context diagnostic artifacts, joins initial-state
features from theta0 with fixed-context gain from the latest checkpoint, and
exports robust statistics, within-Q3 support buckets, standardized regressions,
and support-definition checks.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label: str
    root: Path
    include_main: bool = True


DEFAULT_ROOT = Path("/path/to/outputs/slime_opd/storyline_20260513")
SCALE_ROOT = DEFAULT_ROOT / "scale_context_robustness_20260517"


DATASETS = [
    DatasetSpec(
        "heldout_4b_to_1p7b_300",
        "4B -> 1.7B held-out 300",
        SCALE_ROOT / "heldout_4b_to_1p7b_300",
    ),
    DatasetSpec(
        "8b_to_1p7b_diag300",
        "8B -> 1.7B 300",
        SCALE_ROOT / "8b_to_1p7b_diag300",
    ),
    DatasetSpec(
        "14b_to_1p7b_diag300",
        "14B -> 1.7B 300",
        SCALE_ROOT / "14b_to_1p7b_diag300",
    ),
    DatasetSpec(
        "14b_to_1p7b_diag300_k8",
        "14B -> 1.7B 300 K=8",
        SCALE_ROOT / "14b_to_1p7b_diag300_k8",
    ),
    DatasetSpec(
        "14b_to_1p7b_diag300_k32",
        "14B -> 1.7B 300 K=32",
        SCALE_ROOT / "14b_to_1p7b_diag300_k32",
    ),
    DatasetSpec(
        "gsm8k_4b_to_1p7b_300",
        "4B -> 1.7B GSM8K-COT 300",
        SCALE_ROOT / "gsm8k_4b_to_1p7b_300",
    ),
    DatasetSpec(
        "8b_to_4b_diag300",
        "8B -> 4B 300",
        SCALE_ROOT / "8b_to_4b_diag300",
    ),
    DatasetSpec(
        "14b_to_4b_diag300",
        "14B -> 4B 300",
        SCALE_ROOT / "14b_to_4b_diag300",
    ),
    DatasetSpec(
        "8b_to_1p7b_short",
        "8B -> 1.7B short smoke",
        SCALE_ROOT / "8b_to_1p7b_short",
        include_main=False,
    ),
]


KEYS = ["sample_ordinal", "tok_pos"]
SUPPORT_PROXIES = [
    "Cmass_true",
    "CBC",
    "Coverlap",
    "shared_teacher_topk_mass",
    "shared_student_topk_mass",
    "topk_overlap_frac",
    "topk_jaccard",
    "reach_t1",
    "target_in_student_topk",
    "top1_match",
]


def robust_norm(s: pd.Series, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo = x.quantile(q_low)
    hi = x.quantile(q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.0)


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    std = x.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.mean()) / std


def winsorize(s: pd.Series, q: float = 0.01) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo = x.quantile(q)
    hi = x.quantile(1 - q)
    return x.clip(lo, hi)


def trimmed_mean(s: pd.Series, q: float = 0.05) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return math.nan
    lo = np.quantile(x, q)
    hi = np.quantile(x, 1 - q)
    y = x[(x >= lo) & (x <= hi)]
    return float(np.mean(y)) if len(y) else math.nan


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return beta, pred, r2


def cluster_bootstrap_diff(
    df: pd.DataFrame,
    score_col: str,
    y_col: str = "G_KLf",
    group_col: str = "sample_ordinal",
    low_q: float = 1 / 3,
    high_q: float = 2 / 3,
    n_boot: int = 1000,
    seed: int = 41717,
) -> dict[str, float]:
    if df.empty or score_col not in df:
        return {"mean_diff": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    low_cut = df[score_col].quantile(low_q)
    high_cut = df[score_col].quantile(high_q)
    low = df[df[score_col] <= low_cut]
    high = df[df[score_col] >= high_cut]
    mean_diff = float(high[y_col].mean() - low[y_col].mean()) if len(low) and len(high) else math.nan
    groups = df[group_col].dropna().unique()
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    if len(groups) == 0:
        return {"mean_diff": mean_diff, "ci_low": math.nan, "ci_high": math.nan}
    low_flag = df[score_col] <= low_cut
    high_flag = df[score_col] >= high_cut
    work = pd.DataFrame(
        {
            "group": df[group_col].to_numpy(),
            "low_sum": np.where(low_flag, df[y_col].to_numpy(float), 0.0),
            "low_count": low_flag.astype(int).to_numpy(),
            "high_sum": np.where(high_flag, df[y_col].to_numpy(float), 0.0),
            "high_count": high_flag.astype(int).to_numpy(),
        }
    )
    by_group = work.groupby("group", sort=False)[["low_sum", "low_count", "high_sum", "high_count"]].sum()
    by_group = by_group.reindex(groups).fillna(0.0)
    low_sum = by_group["low_sum"].to_numpy(float)
    low_count = by_group["low_count"].to_numpy(float)
    high_sum = by_group["high_sum"].to_numpy(float)
    high_count = by_group["high_count"].to_numpy(float)
    n_groups = len(groups)
    for _ in range(n_boot):
        idx = rng.integers(0, n_groups, size=n_groups)
        lc = low_count[idx].sum()
        hc = high_count[idx].sum()
        if lc > 0 and hc > 0:
            diffs.append(float(high_sum[idx].sum() / hc - low_sum[idx].sum() / lc))
    if not diffs:
        return {"mean_diff": mean_diff, "ci_low": math.nan, "ci_high": math.nan}
    return {
        "mean_diff": mean_diff,
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }


def within_prompt_matching(
    df: pd.DataFrame,
    score_col: str,
    y_col: str = "G_KLf",
    group_col: str = "sample_ordinal",
    controls: tuple[str, ...] = ("Hn", "Dn", "pos_norm"),
) -> dict[str, float]:
    diffs: list[float] = []
    pairs = 0
    for _, sub in df.groupby(group_col):
        sub = sub.dropna(subset=[score_col, y_col, *controls])
        if len(sub) < 2:
            continue
        cut = sub[score_col].median()
        high = sub[sub[score_col] > cut]
        low = sub[sub[score_col] <= cut]
        if high.empty or low.empty:
            continue
        low_x = low.loc[:, controls].to_numpy(float)
        for _, hrow in high.iterrows():
            hx = hrow.loc[list(controls)].to_numpy(float)
            j = int(np.square(low_x - hx).sum(axis=1).argmin())
            diffs.append(float(hrow[y_col] - low.iloc[j][y_col]))
            pairs += 1
    return {
        "matching_pairs": pairs,
        "matching_mean_diff": float(np.mean(diffs)) if diffs else math.nan,
        "matching_median_diff": float(np.median(diffs)) if diffs else math.nan,
    }


def add_overlap_proxy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Coverlap" in out and "topk_overlap_frac" not in out:
        out["topk_overlap_frac"] = out["Coverlap"]
    if "Coverlap" in out and "topk_jaccard" not in out:
        # K=16 overlap fraction is intersection/K; jaccard = inter/(2K-inter).
        inter_frac = pd.to_numeric(out["Coverlap"], errors="coerce").clip(0, 1)
        out["topk_jaccard"] = inter_frac / (2 - inter_frac + 1e-12)
    if "teacher_top1_id" in out and "student_top1_id" in out and "top1_match" not in out:
        out["top1_match"] = (out["teacher_top1_id"] == out["student_top1_id"]).astype(float)
    if "shared_teacher_topk_mass" not in out and "Cmass_true" in out:
        out["shared_teacher_topk_mass"] = out["Cmass_true"]
    if "shared_student_topk_mass" not in out:
        # Local fallback: student top-K mass under itself is usually saturated.
        out["shared_student_topk_mass"] = 1.0
    return out


def load_dataset(spec: DatasetSpec) -> pd.DataFrame | None:
    fixed_path = spec.root / "eval_latest" / "fixed_context_metrics.parquet"
    theta0_path = spec.root / "theta0_metrics.parquet"
    if not fixed_path.exists():
        return None
    fixed = pd.read_parquet(fixed_path)
    fixed_gain = fixed[KEYS + ["G_KLf", "G_KLr", "KLf_full_base", "KLr_full_base", "Hs_full_base", "Ht_full_base"]].copy()
    if theta0_path.exists():
        theta0 = pd.read_parquet(theta0_path)
        theta0 = theta0.drop(columns=[c for c in ["G_KLf", "G_KLr"] if c in theta0.columns], errors="ignore")
        df = theta0.merge(fixed_gain, on=KEYS, how="inner", suffixes=("", "_gain"))
    else:
        df = fixed.copy()
    df = add_overlap_proxy_columns(df)
    df["dataset"] = spec.name
    df["dataset_label"] = spec.label
    df["include_main"] = spec.include_main
    df["H"] = pd.to_numeric(df.get("Hs_full", df.get("Hs_full_base")), errors="coerce")
    df["Ht"] = pd.to_numeric(df.get("Ht_full", df.get("Ht_full_base")), errors="coerce")
    df["D"] = pd.to_numeric(df.get("KLf_full", df.get("KLf_full_base")), errors="coerce")
    df["C"] = pd.to_numeric(df["Cmass_true"], errors="coerce")
    for col in ["H", "Ht", "D", "C", "G_KLf", "G_KLr", "pos_norm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["H", "D", "C", "G_KLf", "pos_norm"])
    df["Hn"] = robust_norm(df["H"])
    df["Dn"] = robust_norm(df["D"])
    df["Cn"] = robust_norm(df["C"])
    df["Htn"] = robust_norm(df["Ht"])
    df["Dlearn"] = df["Dn"] * df["Cn"]
    df["Dincompat"] = df["Dn"] * (1.0 - df["Cn"])
    df["TIP"] = df["Hn"] + df["Dn"] - df["Hn"] * df["Dn"]
    df["H_plus_teach"] = df["Hn"] + df["Dlearn"] - df["Hn"] * df["Dlearn"]
    df["Q3_lowH_highD"] = (df["Hn"] <= df["Hn"].quantile(0.5)) & (df["Dn"] >= df["Dn"].quantile(0.5))
    df["G_KLf_winsor01"] = winsorize(df["G_KLf"], 0.01)
    df["G_KLf_winsor05"] = winsorize(df["G_KLf"], 0.05)
    return df


def robust_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in df.groupby(["dataset", "dataset_label", "include_main"], dropna=False):
        dataset, label, include_main = name
        x = sub["G_KLf"]
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": label,
                "include_main": bool(include_main),
                "tokens": int(len(sub)),
                "contexts": int(sub["sample_ordinal"].nunique()),
                "raw_mean": float(x.mean()),
                "median": float(x.median()),
                "trimmed_mean_5_95": trimmed_mean(x, 0.05),
                "winsor_mean_1_99": float(winsorize(x, 0.01).mean()),
                "winsor_mean_5_95": float(winsorize(x, 0.05).mean()),
                "q05": float(x.quantile(0.05)),
                "q95": float(x.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def bucket_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bucket_specs = [
        ("all", lambda x: x),
        ("Q3_lowH_highD", lambda x: x[x["Q3_lowH_highD"]]),
    ]
    score_cols = ["Cn", "Dlearn", "Dincompat", "Dn", "TIP", "H_plus_teach"]
    for (dataset, label), dsub in df.groupby(["dataset", "dataset_label"]):
        for region, selector in bucket_specs:
            sub_region = selector(dsub)
            if len(sub_region) < 20:
                continue
            for score in score_cols:
                if score not in sub_region:
                    continue
                try:
                    bins = pd.qcut(sub_region[score], q=4, labels=False, duplicates="drop")
                except ValueError:
                    continue
                tmp = sub_region.assign(bucket=bins)
                for b, bsub in tmp.groupby("bucket", dropna=True):
                    rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": label,
                            "region": region,
                            "score": score,
                            "bucket": int(b),
                            "tokens": int(len(bsub)),
                            "contexts": int(bsub["sample_ordinal"].nunique()),
                            "score_mean": float(bsub[score].mean()),
                            "G_mean": float(bsub["G_KLf"].mean()),
                            "G_median": float(bsub["G_KLf"].median()),
                            "G_trimmed_mean_5_95": trimmed_mean(bsub["G_KLf"], 0.05),
                            "G_winsor_mean_1_99": float(winsorize(bsub["G_KLf"], 0.01).mean()),
                        }
                    )
    return pd.DataFrame(rows)


def standardized_regressions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = {
        "H_D_interaction": ["Hn", "Dn", "Hn_x_Dn", "pos_norm", "Htn"],
        "H_D_C": ["Hn", "Dn", "Hn_x_Dn", "Cn", "pos_norm", "Htn"],
        "teach_decomp": ["Hn", "Dlearn", "Dincompat", "pos_norm", "Htn"],
    }
    df = df.copy()
    df["Hn_x_Dn"] = df["Hn"] * df["Dn"]
    for (dataset, label), dsub in df.groupby(["dataset", "dataset_label"]):
        for y_col in ["G_KLf", "G_KLf_winsor01", "G_KLf_winsor05"]:
            clean = dsub.dropna(subset=[y_col, "Hn", "Dn", "Cn", "Dlearn", "Dincompat", "pos_norm", "Htn"]).copy()
            if len(clean) < 50:
                continue
            y = zscore(clean[y_col]).to_numpy(float)
            for model_name, cols in models.items():
                xcols = []
                for c in cols:
                    clean[f"z_{c}"] = zscore(clean[c])
                    xcols.append(f"z_{c}")
                x = np.column_stack([np.ones(len(clean)), clean[xcols].to_numpy(float)])
                beta, pred, r2 = ols(y, x)
                for c, b in zip(["intercept", *cols], beta):
                    rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": label,
                            "y_col": y_col,
                            "model": model_name,
                            "term": c,
                            "std_coef": float(b),
                            "r2": float(r2),
                            "tokens": int(len(clean)),
                        }
                    )
    return pd.DataFrame(rows)


def q3_support_summary(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for (dataset, label), dsub in df.groupby(["dataset", "dataset_label"]):
        q3 = dsub[dsub["Q3_lowH_highD"]].copy()
        for proxy in SUPPORT_PROXIES:
            if proxy not in q3:
                continue
            q3[f"{proxy}_n"] = robust_norm(q3[proxy])
            boot = cluster_bootstrap_diff(q3, f"{proxy}_n", n_boot=n_boot, seed=seed)
            match = within_prompt_matching(q3, f"{proxy}_n")
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": label,
                    "region": "Q3_lowH_highD",
                    "proxy": proxy,
                    "tokens": int(len(q3)),
                    "contexts": int(q3["sample_ordinal"].nunique()),
                    "proxy_mean": float(q3[proxy].mean()),
                    "proxy_std": float(q3[proxy].std(ddof=0)),
                    **boot,
                    **match,
                }
            )
    return pd.DataFrame(rows)


def spline_sanity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Residual after H, D, position and teacher entropy, then non-parametric bins.
    for (dataset, label), dsub in df.groupby(["dataset", "dataset_label"]):
        clean = dsub.dropna(subset=["G_KLf_winsor01", "Hn", "Dn", "pos_norm", "Htn", "Cn", "Dlearn", "Dincompat"]).copy()
        if len(clean) < 50:
            continue
        y = clean["G_KLf_winsor01"].to_numpy(float)
        x = np.column_stack([np.ones(len(clean)), clean[["Hn", "Dn", "pos_norm", "Htn"]].to_numpy(float)])
        _, pred, base_r2 = ols(y, x)
        clean["resid_after_HD_pos_Ht"] = y - pred
        for score in ["Cn", "Dlearn", "Dincompat", "TIP", "H_plus_teach"]:
            try:
                bins = pd.qcut(clean[score], q=10, labels=False, duplicates="drop")
            except ValueError:
                continue
            tmp = clean.assign(score_bin=bins)
            for b, bsub in tmp.groupby("score_bin", dropna=True):
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": label,
                        "score": score,
                        "bin": int(b),
                        "tokens": int(len(bsub)),
                        "score_mean": float(bsub[score].mean()),
                        "residual_mean": float(bsub["resid_after_HD_pos_Ht"].mean()),
                        "residual_median": float(bsub["resid_after_HD_pos_Ht"].median()),
                        "base_r2": float(base_r2),
                    }
                )
    return pd.DataFrame(rows)


def read_existing_q3_summaries() -> pd.DataFrame:
    rows = []
    for spec in DATASETS:
        path1 = spec.root / "analysis" / "gain" / "q3_bootstrap_matching_summary.csv"
        path2 = spec.root / "q3_bootstrap_matching_summary.csv"
        path = path1 if path1.exists() else path2
        if not path.exists():
            continue
        row = pd.read_csv(path).iloc[0].to_dict()
        row.update({"dataset": spec.name, "dataset_label": spec.label, "include_main": spec.include_main, "source": str(path)})
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(outdir: Path, tables: dict[str, pd.DataFrame]) -> None:
    q3 = tables["existing_q3"].copy()
    if len(q3):
        q3 = q3[q3["include_main"]].copy()
    lines = [
        "# Section 3 P0 Paper-Ready Diagnostics",
        "",
        "This package consolidates fixed-context evidence for the Section 3 claim: raw teacher--student divergence is a coarse proxy for learning value, and high-divergence tokens differ in local teachability.",
        "",
        "## Main 300-context diagnostics",
    ]
    if len(q3):
        view = q3[[
            "dataset_label",
            "bootstrap_mean_diff",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "matching_pairs",
            "matching_mean_diff",
        ]].copy()
        lines.append(view.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Recommended Section 3 usage",
            "",
            "- Use bootstrap high-minus-low and prompt matching as the primary effect estimates.",
            "- Use robust gain statistics instead of raw means when discussing absolute fixed-context gain; raw means can be distorted by heavy-tailed token losses.",
            "- For within-Q3 analysis, emphasize support/compatibility buckets (`Cmass_true`, `CBC`, shared teacher top-K mass) rather than raw divergence alone.",
            "- Treat the 8B short smoke diagnostic as appendix/sanity only.",
            "",
            "## Key files",
            "",
            "- `existing_q3_scale_context_summary.csv`: compact scale/context table.",
            "- `robust_gain_stats.csv`: median, trimmed, and winsorized fixed-context gains.",
            "- `within_q3_support_proxy_summary.csv`: support-definition robustness in Q3.",
            "- `standardized_regression_coefficients.csv`: standardized coefficients with raw and winsorized gains.",
            "- `spline_sanity_bins.csv`: nonparametric residual-vs-score bin curves for GAM-like sanity checks.",
            "- `bucket_gain_curves.csv`: quartile bucket curves for figure sources.",
        ]
    )
    (outdir / "section3_p0_evidence_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_ROOT / "section3_paper_ready_20260517"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=41717)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    meta = []
    for spec in DATASETS:
        df = load_dataset(spec)
        if df is None:
            continue
        frames.append(df)
        meta.append(
            {
                "dataset": spec.name,
                "dataset_label": spec.label,
                "root": str(spec.root),
                "include_main": spec.include_main,
                "tokens": int(len(df)),
                "contexts": int(df["sample_ordinal"].nunique()),
            }
        )
    if not frames:
        raise SystemExit("No datasets found.")
    all_df = pd.concat(frames, ignore_index=True)

    tables = {
        "dataset_manifest": pd.DataFrame(meta),
        "existing_q3": read_existing_q3_summaries(),
        "robust_gain": robust_gain_table(all_df),
        "bucket_gain": bucket_table(all_df),
        "regression": standardized_regressions(all_df),
        "support_q3": q3_support_summary(all_df, args.bootstrap, args.seed),
        "spline": spline_sanity_table(all_df),
    }
    tables["dataset_manifest"].to_csv(outdir / "section3_dataset_manifest.csv", index=False)
    tables["existing_q3"].to_csv(outdir / "existing_q3_scale_context_summary.csv", index=False)
    tables["robust_gain"].to_csv(outdir / "robust_gain_stats.csv", index=False)
    tables["bucket_gain"].to_csv(outdir / "bucket_gain_curves.csv", index=False)
    tables["regression"].to_csv(outdir / "standardized_regression_coefficients.csv", index=False)
    tables["support_q3"].to_csv(outdir / "within_q3_support_proxy_summary.csv", index=False)
    tables["spline"].to_csv(outdir / "spline_sanity_bins.csv", index=False)
    all_df[[
        "dataset",
        "dataset_label",
        "sample_ordinal",
        "tok_pos",
        "pos_norm",
        "Hn",
        "Dn",
        "Cn",
        "Dlearn",
        "Dincompat",
        "TIP",
        "H_plus_teach",
        "Q3_lowH_highD",
        "G_KLf",
        "G_KLf_winsor01",
    ]].to_parquet(outdir / "section3_analysis_tokens.parquet", index=False)
    write_markdown(outdir, tables)

    print(f"saved={outdir}")
    print(tables["existing_q3"].to_string(index=False))


if __name__ == "__main__":
    main()
