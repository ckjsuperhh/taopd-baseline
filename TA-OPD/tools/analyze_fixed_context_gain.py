#!/usr/bin/env python3
"""Analyze fixed-context token learnability G_t with compatibility controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input: {path}")


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _first(df: pd.DataFrame, cols: list[str]) -> str:
    for col in cols:
        if col in df.columns:
            return col
    raise ValueError(f"Missing any of columns: {cols}")


def _norm(s: pd.Series, q_low: float, q_high: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo = x.quantile(q_low)
    hi = x.quantile(q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.0)


def _prepare(df: pd.DataFrame, args) -> tuple[pd.DataFrame, dict[str, str]]:
    y_col = args.gain_col or _first(df, ["G_KLf", "G_KLr"])
    h_col = args.h_col or _first(df, ["H_norm", "Hs_full", "Hs_topk_norm", "Hs"])
    d_col = args.d_col or _first(df, ["D_norm", "KLf_full_base", "KLf_full", "KLf_union", "KLf"])
    c_col = args.c_col or _first(df, ["Cmass_true", "Cmass", "C_norm", "CBC", "Coverlap"])
    group_col = args.group_col or _first(df, ["sample_index", "prompt_id", "group_index", "sample_ordinal"])
    pos_col = args.pos_col or _first(df, ["pos_norm", "tok_pos"])
    ht_col = args.ht_col if args.ht_col in df.columns else (args.ht_col or None)
    if ht_col is None:
        ht_candidates = [col for col in ["Ht_full_base", "Ht_full", "Ht_topk_norm", "Ht"] if col in df.columns]
        ht_col = ht_candidates[0] if ht_candidates else None

    out = df.copy()
    out["G"] = pd.to_numeric(out[y_col], errors="coerce")
    out["Hn"] = _norm(out[h_col], args.q_low, args.q_high)
    out["Dn"] = _norm(out[d_col], args.q_low, args.q_high)
    out["Cn"] = _norm(out[c_col], args.q_low, args.q_high)
    out["pos"] = _norm(out[pos_col], 0.0, 1.0)
    out["Ht_n"] = _norm(out[ht_col], args.q_low, args.q_high) if ht_col else 0.0
    out["Dlearn"] = out["Dn"] * out["Cn"]
    out["Dincompat"] = out["Dn"] * (1.0 - out["Cn"])
    out["tip_score"] = out["Hn"] + out["Dn"] - out["Hn"] * out["Dn"]
    out["ca_softor_score"] = out["Hn"] + out["Dlearn"] - out["Hn"] * out["Dlearn"]
    out["group"] = out[group_col]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["G", "Hn", "Dn", "Cn", "pos"])
    meta = {
        "gain_col": y_col,
        "h_col": h_col,
        "d_col": d_col,
        "c_col": c_col,
        "group_col": group_col,
        "pos_col": pos_col,
        "ht_col": ht_col or "",
    }
    return out, meta


def _ols(y: np.ndarray, x: np.ndarray) -> dict[str, object]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"r2": r2, "beta": beta.tolist(), "pred": pred}


def _regression_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["G"].to_numpy()
    ones = np.ones(len(df))
    base_x = np.column_stack([ones, df["Hn"], df["Dn"], df["Hn"] * df["Dn"], df["pos"], df["Ht_n"]])
    c_x = np.column_stack([base_x, df["Cn"]])
    split_x = np.column_stack([ones, df["Hn"], df["Dlearn"], df["Dincompat"], df["pos"], df["Ht_n"]])

    base = _ols(y, base_x)
    c = _ols(y, c_x)
    split = _ols(y, split_x)
    rows = [
        {"model": "H_D", "r2": base["r2"], "delta_r2_vs_HD": 0.0, "terms": "1,H,D,H*D,pos,Ht"},
        {"model": "H_D_C", "r2": c["r2"], "delta_r2_vs_HD": c["r2"] - base["r2"], "terms": "HD + C"},
        {
            "model": "H_Dlearn_Dincompat",
            "r2": split["r2"],
            "delta_r2_vs_HD": split["r2"] - base["r2"],
            "terms": "1,H,D*C,D*(1-C),pos,Ht",
        },
    ]
    residual = pd.Series(y - np.asarray(base["pred"]), index=df.index, name="residual_HD")
    return pd.DataFrame(rows), residual


def _q3(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.use_quadrant and "quadrant" in df.columns:
        q3 = df[df["quadrant"].astype(str) == "Q3_lowH_highD"].copy()
        if len(q3):
            return q3
    h_cut = df["Hn"].quantile(args.h_quantile)
    d_cut = df["Dn"].quantile(args.d_quantile)
    return df[(df["Hn"] <= h_cut) & (df["Dn"] >= d_cut)].copy()


def _group_bootstrap_diff(q3: pd.DataFrame, args) -> dict[str, float]:
    if q3.empty:
        return {"mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    low_cut = q3["Cn"].quantile(args.low_c_quantile)
    high_cut = q3["Cn"].quantile(args.high_c_quantile)
    low = q3[q3["Cn"] <= low_cut]
    high = q3[q3["Cn"] >= high_cut]
    mean_diff = float(high["G"].mean() - low["G"].mean())
    groups = q3["group"].dropna().unique()
    rng = np.random.default_rng(args.seed)
    diffs = []
    for _ in range(args.bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        boot = pd.concat([q3[q3["group"] == g] for g in sampled], ignore_index=True)
        b_low = boot[boot["Cn"] <= low_cut]
        b_high = boot[boot["Cn"] >= high_cut]
        if len(b_low) and len(b_high):
            diffs.append(float(b_high["G"].mean() - b_low["G"].mean()))
    if not diffs:
        return {"mean_diff": mean_diff, "ci_low": np.nan, "ci_high": np.nan}
    return {
        "mean_diff": mean_diff,
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }


def _within_group_matching(q3: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for group, sub in q3.groupby("group"):
        if len(sub) < 2:
            continue
        med = sub["Cn"].median()
        high = sub[sub["Cn"] > med]
        low = sub[sub["Cn"] <= med]
        if high.empty or low.empty:
            continue
        low_x = low[["Hn", "Dn", "pos"]].to_numpy()
        for idx, hrow in high.iterrows():
            hx = hrow[["Hn", "Dn", "pos"]].to_numpy(dtype=float)
            dist = ((low_x - hx) ** 2).sum(axis=1)
            j = int(dist.argmin())
            lrow = low.iloc[j]
            rows.append(
                {
                    "group": group,
                    "high_index": idx,
                    "low_index": lrow.name,
                    "G_high": hrow["G"],
                    "G_low": lrow["G"],
                    "diff": hrow["G"] - lrow["G"],
                    "C_high": hrow["Cn"],
                    "C_low": lrow["Cn"],
                    "distance": float(dist[j]),
                }
            )
    return pd.DataFrame(rows)


def _plot(df: pd.DataFrame, q3: pd.DataFrame, residual: pd.Series, outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hexbin(df["Hn"], df["Dn"], C=df["Cn"], gridsize=35, reduce_C_function=np.mean, mincnt=1)
    plt.xlabel("H normalized")
    plt.ylabel("D normalized")
    plt.colorbar(label="mean C")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_H_D_meanC.png", dpi=170)
    plt.close()

    if len(q3):
        bins = pd.qcut(q3["Cn"], q=min(3, q3["Cn"].nunique()), duplicates="drop")
        grouped = q3.groupby(bins, observed=False)["G"].mean()
        plt.figure(figsize=(6, 4))
        grouped.plot(kind="bar")
        plt.ylabel("mean G")
        plt.xlabel("Q3 C bins")
        plt.tight_layout()
        plt.savefig(outdir / "q3_gain_by_C_bin.png", dpi=170)
        plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(df["Cn"], residual, s=4, alpha=0.25)
    plt.xlabel("C normalized")
    plt.ylabel("Residual after H,D")
    plt.tight_layout()
    plt.savefig(outdir / "residual_HD_vs_C.png", dpi=170)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gain-col", default=None)
    parser.add_argument("--h-col", default=None)
    parser.add_argument("--d-col", default=None)
    parser.add_argument("--c-col", default=None)
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--pos-col", default=None)
    parser.add_argument("--ht-col", default=None)
    parser.add_argument("--q-low", type=float, default=0.05)
    parser.add_argument("--q-high", type=float, default=0.95)
    parser.add_argument("--h-quantile", type=float, default=0.5)
    parser.add_argument("--d-quantile", type=float, default=0.5)
    parser.add_argument("--low-c-quantile", type=float, default=1 / 3)
    parser.add_argument("--high-c-quantile", type=float, default=2 / 3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-quadrant", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df_raw = _read(Path(args.input))
    df, meta = _prepare(df_raw, args)
    q3 = _q3(df, args)
    reg, residual = _regression_summary(df)
    df["residual_HD"] = residual
    boot = _group_bootstrap_diff(q3, args)
    match = _within_group_matching(q3, args)
    matching_summary = {
        "pairs": int(len(match)),
        "mean_diff": float(match["diff"].mean()) if len(match) else np.nan,
        "median_diff": float(match["diff"].median()) if len(match) else np.nan,
    }

    reg.to_csv(outdir / "regression_summary.csv", index=False)
    summary_row = {
        **{f"bootstrap_{k}": v for k, v in boot.items()},
        **{f"matching_{k}": v for k, v in matching_summary.items()},
    }
    pd.DataFrame([summary_row]).to_csv(outdir / "q3_bootstrap_matching_summary.csv", index=False)
    _write(match, outdir / "q3_matching_pairs.parquet")
    _write(df, outdir / "analysis_tokens.parquet")
    _plot(df, q3, residual, outdir)
    (outdir / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n")
    print(reg.to_string(index=False))
    print(pd.DataFrame([summary_row]).to_string(index=False))
    print(f"tokens={len(df)} q3_tokens={len(q3)} saved={outdir}")


if __name__ == "__main__":
    main()
