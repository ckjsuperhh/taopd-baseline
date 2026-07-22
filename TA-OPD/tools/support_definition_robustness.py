#!/usr/bin/env python3
"""Build support-definition robustness tables for OPD teachability.

This script intentionally depends only on the fixed-context metric parquet files
already produced by the OPD diagnostic pipeline. It compares several choices of
the local-support proxy C in the decomposition

    Dlearn = D * C,    Dincompat = D * (1 - C)

and reports whether the learnable component remains the stronger predictor of
fixed-context OPD gain.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DATASETS = {
    "main_k4": "/path/to/outputs/slime_opd/qwen3_1_7b_dapo_diag_k4_exact_20260510_080727/fixed_context/eval_latest/fixed_context_metrics.parquet",
    "main_k8": "/path/to/outputs/slime_opd/qwen3_1_7b_dapo_diag_k8_exact_20260510_082406/fixed_context/eval_latest/fixed_context_metrics.parquet",
    "main_k16": "/path/to/outputs/slime_opd/qwen3_1_7b_dapo_diag_k16_exact_20260510_084415/fixed_context/eval_latest/fixed_context_metrics.parquet",
    "heldout_k16": "/path/to/outputs/slime_opd/qwen3_1_7b_student_qwen3_4b_teacher_diag_k16_exact_wait_20260510_071229/fixed_context/eval_latest/fixed_context_metrics.parquet",
}


def as_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.asarray(x)
    if isinstance(x, str):
        # Parquet list columns should not arrive as strings, but this keeps the
        # script robust to CSV debugging exports.
        try:
            return np.asarray(json.loads(x))
        except Exception:
            return np.asarray([])
    return np.asarray([])


def robust_rank(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x), dtype=float), index=s.index)
    ranks = x.rank(method="average", pct=True)
    return ranks.fillna(ranks.median()).clip(0.0, 1.0)


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median())
    std = float(x.std(ddof=0))
    if not math.isfinite(std) or std < 1e-12:
        return pd.Series(np.zeros(len(x), dtype=float), index=s.index)
    return (x - float(x.mean())) / std


def ols(y: pd.Series, x: pd.DataFrame) -> tuple[float, dict[str, float]]:
    yv = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mat = x.copy()
    mat.insert(0, "intercept", 1.0)
    joined = pd.concat([yv.rename("y"), mat], axis=1).dropna()
    if len(joined) < len(mat.columns) + 3:
        return float("nan"), {c: float("nan") for c in mat.columns}
    y_np = joined["y"].to_numpy(dtype=float)
    x_np = joined.drop(columns=["y"]).to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x_np, y_np, rcond=None)
    pred = x_np @ beta
    ss_res = float(np.sum((y_np - pred) ** 2))
    ss_tot = float(np.sum((y_np - y_np.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return r2, dict(zip(joined.drop(columns=["y"]).columns, beta))


def bootstrap_group_diff(
    df: pd.DataFrame,
    score_col: str,
    group_col: str = "sample_ordinal",
    gain_col: str = "G",
    n_boot: int = 800,
    seed: int = 17,
) -> tuple[float, float, float, int, int]:
    q3 = df[(df["Hn"] <= df["Hn"].median()) & (df["Dn"] >= df["Dn"].median())].copy()
    q3 = q3.dropna(subset=[score_col, group_col, gain_col])
    if q3.empty:
        return float("nan"), float("nan"), float("nan"), 0, 0

    lo = q3[score_col].quantile(0.25)
    hi = q3[score_col].quantile(0.75)
    low = q3[q3[score_col] <= lo]
    high = q3[q3[score_col] >= hi]
    obs = float(high[gain_col].mean() - low[gain_col].mean())

    # Pre-aggregate to group-level sufficient statistics. This keeps the
    # bootstrap as a prompt-cluster bootstrap while avoiding expensive repeated
    # DataFrame concatenation over token-level rows.
    high_stat = (
        high.groupby(group_col)[gain_col]
        .agg(["sum", "count"])
        .rename(columns={"sum": "high_sum", "count": "high_count"})
    )
    low_stat = (
        low.groupby(group_col)[gain_col]
        .agg(["sum", "count"])
        .rename(columns={"sum": "low_sum", "count": "low_count"})
    )
    stat = high_stat.join(low_stat, how="outer").fillna(0.0)
    groups = stat.index.to_numpy()
    stat_np = stat[["high_sum", "high_count", "low_sum", "low_count"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), size=len(groups))
        sums = stat_np[idx].sum(axis=0)
        if sums[1] > 0 and sums[3] > 0:
            diffs.append(float(sums[0] / sums[1] - sums[2] / sums[3]))
    if not diffs:
        return obs, float("nan"), float("nan"), len(high), len(low)
    ci_low, ci_high = np.quantile(diffs, [0.025, 0.975])
    return obs, float(ci_low), float(ci_high), len(high), len(low)


def topk_proxy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if {"student_top_ids", "teacher_top_ids"}.issubset(df.columns):
        jac = []
        overlap = []
        top1 = []
        for s_ids, t_ids in zip(df["student_top_ids"], df["teacher_top_ids"]):
            s = set(map(int, as_array(s_ids)))
            t = set(map(int, as_array(t_ids)))
            inter = len(s & t)
            union = len(s | t)
            jac.append(inter / union if union else 0.0)
            overlap.append(inter / max(len(t), 1))
            top1.append(1.0 if s and t and next(iter(s)) == next(iter(t)) else 0.0)
        out["topk_jaccard"] = jac
        out["topk_overlap_frac"] = overlap
        out["top1_match"] = top1

    if {"student_top_ids", "teacher_top_ids", "teacher_top_logps"}.issubset(df.columns):
        masses = []
        for s_ids, t_ids, t_logps in zip(df["student_top_ids"], df["teacher_top_ids"], df["teacher_top_logps"]):
            s = set(map(int, as_array(s_ids)))
            tids = as_array(t_ids).astype(int, copy=False)
            tlps = as_array(t_logps).astype(float, copy=False)
            total = 0.0
            for tid, tlp in zip(tids, tlps):
                if int(tid) in s and math.isfinite(float(tlp)):
                    total += math.exp(float(tlp))
            masses.append(total)
        out["shared_teacher_topk_mass"] = masses

    if {"student_top_ids", "teacher_top_ids", "student_top_logps"}.issubset(df.columns):
        masses = []
        for s_ids, t_ids, s_logps in zip(df["student_top_ids"], df["teacher_top_ids"], df["student_top_logps"]):
            t = set(map(int, as_array(t_ids)))
            sids = as_array(s_ids).astype(int, copy=False)
            slps = as_array(s_logps).astype(float, copy=False)
            total = 0.0
            for sid, slp in zip(sids, slps):
                if int(sid) in t and math.isfinite(float(slp)):
                    total += math.exp(float(slp))
            masses.append(total)
        out["shared_student_topk_mass"] = masses
    return out


def analyze_one(name: str, path: Path, n_boot: int) -> list[dict[str, object]]:
    df = pd.read_parquet(path)
    if "G_KLf" not in df.columns:
        raise ValueError(f"{path} does not contain G_KLf")
    df = pd.concat([df, topk_proxy_columns(df)], axis=1)
    df["G"] = pd.to_numeric(df["G_KLf"], errors="coerce")
    d_col = "KLf_full_base" if "KLf_full_base" in df.columns else "KLf_full"
    h_col = "Hs_full_base" if "Hs_full_base" in df.columns else "Hs_full"
    ht_col = "Ht_full_base" if "Ht_full_base" in df.columns else "Ht_full"
    df["Dn"] = robust_rank(df[d_col])
    df["Hn"] = robust_rank(df[h_col])
    df["Ht_n"] = robust_rank(df[ht_col])
    df["pos"] = pd.to_numeric(df.get("pos_norm", df.get("tok_pos", 0)), errors="coerce").fillna(0.0)
    if "tok_pos" in df.columns and df["pos"].max() > 1:
        df["pos"] = robust_rank(df["pos"])

    base_x = pd.DataFrame({
        "Hn": zscore(df["Hn"]),
        "Dn": zscore(df["Dn"]),
        "Hn_x_Dn": zscore(df["Hn"] * df["Dn"]),
        "pos": zscore(df["pos"]),
        "Ht_n": zscore(df["Ht_n"]),
    })
    base_r2, _ = ols(zscore(df["G"]), base_x)

    support_cols = [
        "Cmass_true",
        "Coverlap",
        "CBC",
        "reach_t1",
        "target_in_student_topk",
        "topk_jaccard",
        "topk_overlap_frac",
        "shared_teacher_topk_mass",
        "shared_student_topk_mass",
        "top1_match",
    ]
    rows = []
    for c_col in support_cols:
        if c_col not in df.columns:
            continue
        c_raw = pd.to_numeric(df[c_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if c_raw.notna().sum() < 100:
            continue
        c = robust_rank(c_raw)
        score = df["Dn"] * c
        dincompat = df["Dn"] * (1.0 - c)
        x = pd.DataFrame({
            "Hn": zscore(df["Hn"]),
            "Dlearn": zscore(score),
            "Dincompat": zscore(dincompat),
            "pos": zscore(df["pos"]),
            "Ht_n": zscore(df["Ht_n"]),
        })
        r2, beta = ols(zscore(df["G"]), x)
        diff, ci_low, ci_high, n_hi, n_lo = bootstrap_group_diff(
            pd.DataFrame({
                "G": df["G"],
                "Hn": df["Hn"],
                "Dn": df["Dn"],
                "score": score,
                "sample_ordinal": df.get("sample_ordinal", pd.Series(np.arange(len(df)), index=df.index)),
            }),
            score_col="score",
            n_boot=n_boot,
        )
        rows.append({
            "dataset": name,
            "path": str(path),
            "n_tokens": int(len(df)),
            "n_prompts": int(df["sample_ordinal"].nunique()) if "sample_ordinal" in df.columns else int(len(df)),
            "support_definition": c_col,
            "support_mean": float(c_raw.mean()),
            "support_std": float(c_raw.std(ddof=0)),
            "r2_HD": base_r2,
            "r2_decomp": r2,
            "delta_r2_vs_HD": r2 - base_r2 if math.isfinite(r2) and math.isfinite(base_r2) else float("nan"),
            "beta_Dlearn": float(beta.get("Dlearn", float("nan"))),
            "beta_Dincompat": float(beta.get("Dincompat", float("nan"))),
            "beta_gap": float(beta.get("Dlearn", float("nan")) - beta.get("Dincompat", float("nan"))),
            "q3_high_minus_low_mean_gain": diff,
            "q3_high_minus_low_ci_low": ci_low,
            "q3_high_minus_low_ci_high": ci_high,
            "q3_high_tokens": n_hi,
            "q3_low_tokens": n_lo,
        })
    return rows


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    keep = [
        "dataset",
        "support_definition",
        "n_tokens",
        "n_prompts",
        "r2_decomp",
        "delta_r2_vs_HD",
        "beta_Dlearn",
        "beta_Dincompat",
        "beta_gap",
        "q3_high_minus_low_mean_gain",
        "q3_high_minus_low_ci_low",
        "q3_high_minus_low_ci_high",
    ]
    view = df[keep].copy()
    numeric_cols = [c for c in view.columns if c not in {"dataset", "support_definition"}]
    for c in numeric_cols:
        view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.4g}" if isinstance(x, float) else str(x))
    lines = [
        "# Support-definition robustness",
        "",
        "Fixed-context diagnostic table. `Dlearn = D * C` and `Dincompat = D * (1-C)` are recomputed with each support proxy `C`; coefficients are standardized. Q3 high-minus-low compares top vs bottom quartile of `Dlearn` inside low-entropy/high-divergence tokens with prompt-cluster bootstrap CIs.",
        "",
        view.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/path/to/outputs/slime_opd/analysis/scale_context_robustness_20260517")
    parser.add_argument("--dataset", action="append", default=[], help="NAME=PATH; can be repeated. Defaults to known K4/K8/K16/heldout diagnostics.")
    parser.add_argument("--n-boot", type=int, default=800)
    args = parser.parse_args()

    datasets = dict(DEFAULT_DATASETS)
    for item in args.dataset:
        name, p = item.split("=", 1)
        datasets[name] = p

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for name, p in datasets.items():
        path = Path(p)
        if not path.exists():
            missing.append({"dataset": name, "path": str(path)})
            continue
        rows.extend(analyze_one(name, path, args.n_boot))
    df = pd.DataFrame(rows)
    csv_path = out_dir / "support_definition_robustness.csv"
    md_path = out_dir / "support_definition_robustness.md"
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)
    if missing:
        pd.DataFrame(missing).to_csv(out_dir / "missing_datasets.csv", index=False)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    if missing:
        print(f"missing {len(missing)} datasets; see {out_dir / 'missing_datasets.csv'}")


if __name__ == "__main__":
    main()
