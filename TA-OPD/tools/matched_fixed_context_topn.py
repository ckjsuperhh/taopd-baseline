#!/usr/bin/env python3
"""Exact top-N fixed-context selector comparison for OPD teachability.

This analysis is intentionally offline and GPU-free.  It uses the shared
fixed-context bank and asks: if every selector is allowed to keep exactly the
same number of direct supervision targets, which selector has higher
fixed-context gain?

The output is meant to address the keep-ratio confound in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_ROOT = Path("/path/to/outputs/slime_opd/decomposition_common_context_diag_k4ckpt_20260511")
DEFAULT_OUT = Path("/path/to/outputs/slime_opd/analysis/matched_fixed_context_topn")


def norm(series: pd.Series, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo = x.quantile(q_low)
    hi = x.quantile(q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.0)


def winsor(series: pd.Series, low: float = 0.01, high: float = 0.99) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x.clip(x.quantile(low), x.quantile(high))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["G"] = winsor(out["G_KLf"])
    out["G_raw"] = pd.to_numeric(out["G_KLf"], errors="coerce")
    out["Hn"] = norm(out["Hs_full_base"] if "Hs_full_base" in out else out["Hs_full"])
    out["Dn"] = norm(out["KLf_full_base"] if "KLf_full_base" in out else out["KLf_full"])
    out["Cn"] = norm(out["Cmass_true"])
    out["pos_n"] = norm(out["pos_norm"] if "pos_norm" in out else out["tok_pos"], 0.0, 1.0)
    out["Dlearn"] = out["Dn"] * out["Cn"]
    out["Dincompat"] = out["Dn"] * (1.0 - out["Cn"])
    out["tip_softor"] = out["Hn"] + out["Dn"] - out["Hn"] * out["Dn"]
    out["ca_softor"] = out["Hn"] + out["Dlearn"] - out["Hn"] * out["Dlearn"]
    h_cut = out["Hn"].quantile(0.5)
    d_cut = out["Dn"].quantile(0.5)
    out["is_q3"] = (out["Hn"] <= h_cut) & (out["Dn"] >= d_cut)
    out["q3_highc"] = out["is_q3"].astype(float) * 2.0 + out["Cn"]
    out["q3_dlearn"] = out["is_q3"].astype(float) * 2.0 + out["Dlearn"]
    out["q3_divergence"] = out["is_q3"].astype(float) * 2.0 + out["Dn"]
    out["random_score"] = np.arange(len(out), dtype=float)
    out["group"] = out["sample_index"] if "sample_index" in out else out["group_index"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["G", "Hn", "Dn", "Cn", "group"])
    return out


def cluster_ci(selected: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    mean = float(selected["G"].mean()) if len(selected) else float("nan")
    groups = selected["group"].dropna().unique()
    if len(groups) <= 1 or n_boot <= 0:
        return mean, float("nan"), float("nan")
    by_group = {g: selected.loc[selected["group"] == g, "G"].to_numpy() for g in groups}
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        vals.append(float(np.concatenate([by_group[g] for g in sampled]).mean()))
    return mean, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def select_topn(df: pd.DataFrame, score_col: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if score_col == "random":
        return df.iloc[rng.choice(np.arange(len(df)), size=n, replace=False)].copy()
    return df.sort_values(score_col, ascending=False).head(n).copy()


def summarize_selection(df: pd.DataFrame, k_value: int, ratio: float, selector: str, score_col: str, n_boot: int, seed: int) -> dict:
    n_keep = max(1, int(round(len(df) * ratio)))
    rng = np.random.default_rng(seed + k_value * 1000 + int(ratio * 100000) + abs(hash(selector)) % 1000)
    selected = select_topn(df, score_col, n_keep, rng)
    mean, lo, hi = cluster_ci(selected, rng, n_boot)
    return {
        "K": k_value,
        "ratio": ratio,
        "selector": selector,
        "n_tokens_total": len(df),
        "n_keep": len(selected),
        "keep_ratio_exact": len(selected) / len(df),
        "mean_G": mean,
        "ci_low": lo,
        "ci_high": hi,
        "gain_per_keep_ratio": mean / (len(selected) / len(df)) if len(selected) else float("nan"),
        "selected_q3_frac": float(selected["is_q3"].mean()),
        "selected_mean_Hn": float(selected["Hn"].mean()),
        "selected_mean_Dn": float(selected["Dn"].mean()),
        "selected_mean_Cn": float(selected["Cn"].mean()),
        "selected_mean_Dlearn": float(selected["Dlearn"].mean()),
        "selected_mean_Dincompat": float(selected["Dincompat"].mean()),
        "score_col": score_col,
    }


def write_markdown(summary: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Exact Top-N Fixed-Context Selector Comparison",
        "",
        "Every selector keeps exactly the same number of tokens for a given K and ratio. The outcome is winsorized fixed-context `G_KLf` with prompt-cluster bootstrap CIs.",
        "",
    ]
    for k_value in sorted(summary["K"].unique()):
        for ratio in sorted(summary.loc[summary["K"] == k_value, "ratio"].unique()):
            sub = summary[(summary["K"] == k_value) & (summary["ratio"] == ratio)].sort_values("mean_G", ascending=False)
            lines.append(f"## K={k_value}, ratio={ratio:.2f}")
            lines.append("")
            lines.append("| selector | mean G | 95% CI | gain / keep | Q3 frac | mean Dlearn | mean Dincompat |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for row in sub.to_dict("records"):
                lines.append(
                    f"| `{row['selector']}` | {row['mean_G']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
                    f"{row['gain_per_keep_ratio']:.2f} | {row['selected_q3_frac']:.3f} | "
                    f"{row['selected_mean_Dlearn']:.3f} | {row['selected_mean_Dincompat']:.3f} |"
                )
            lines.append("")
    (out_dir / "matched_fixed_context_topn_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ratios", default="0.01,0.03,0.05,0.10")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    selectors = {
        "dlearn_high": "Dlearn",
        "divergence": "Dn",
        "tip_softor": "tip_softor",
        "ca_softor": "ca_softor",
        "entropy": "Hn",
        "highC": "Cn",
        "dincompat_high": "Dincompat",
        "q3_highC": "q3_highc",
        "q3_dlearn": "q3_dlearn",
        "q3_divergence": "q3_divergence",
        "random": "random",
    }
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.input_root.glob("k*/fixed_context_metrics.parquet")):
        k_value = int(path.parent.name.lstrip("k"))
        df = prepare(pd.read_parquet(path))
        for ratio in ratios:
            for selector, score_col in selectors.items():
                rows.append(summarize_selection(df, k_value, ratio, selector, score_col, args.bootstrap, args.seed))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "matched_fixed_context_topn_summary.csv", index=False)
    write_markdown(summary, args.out_dir)
    manifest = {
        "input_root": str(args.input_root),
        "ratios": ratios,
        "selectors": selectors,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "outputs": [
            str(args.out_dir / "matched_fixed_context_topn_summary.csv"),
            str(args.out_dir / "matched_fixed_context_topn_summary.md"),
        ],
    }
    (args.out_dir / "matched_fixed_context_topn_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(args.out_dir / "matched_fixed_context_topn_summary.csv")
    print(args.out_dir / "matched_fixed_context_topn_summary.md")


if __name__ == "__main__":
    main()

