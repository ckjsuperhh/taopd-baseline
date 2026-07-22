#!/usr/bin/env python3
"""Stage-0 Q3 reachability probe for OPD compatibility research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_any(path: Path) -> pd.DataFrame:
    if path.is_dir():
        parts = []
        for file in sorted(path.glob("rollout_*.csv")) + sorted(path.glob("rollout_*.jsonl")):
            parts.append(_read_any(file))
        if not parts:
            raise FileNotFoundError(f"No rollout_*.csv/jsonl files found in {path}")
        return pd.concat(parts, ignore_index=True, sort=False)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type: {path}")


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of these columns exist: {candidates}")


def _finite_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)


def _auc_score(y: np.ndarray, scores: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _make_splits(y: np.ndarray, groups: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 3:
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        folds = np.array_split(shuffled, min(5, len(shuffled)))
        for fold in folds:
            test = np.isin(groups, fold)
            yield np.where(~test)[0], np.where(test)[0]
        return

    class_counts = np.bincount(y.astype(int), minlength=2)
    min_class = int(class_counts.min())
    if min_class < 2:
        return
    n_splits = min(5, min_class)
    folds = [[] for _ in range(n_splits)]
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, chunk in enumerate(np.array_split(idx, n_splits)):
            folds[i].extend(chunk.tolist())
    all_idx = np.arange(len(y))
    for fold in folds:
        test = np.array(sorted(fold), dtype=int)
        train = np.setdiff1d(all_idx, test, assume_unique=False)
        yield train, test


def _predict_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    steps: int = 700,
    lr: float = 0.08,
    l2: float = 1e-4,
) -> np.ndarray:
    del seed
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_train = np.concatenate([np.ones((len(x_train), 1)), x_train], axis=1)
    x_test = np.concatenate([np.ones((len(x_test), 1)), x_test], axis=1)

    y_train = y_train.astype(float)
    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    weights = np.where(y_train > 0.5, len(y_train) / (2.0 * pos), len(y_train) / (2.0 * neg))

    beta = np.zeros(x_train.shape[1], dtype=float)
    for _ in range(steps):
        logits = np.clip(x_train @ beta, -40, 40)
        prob = 1.0 / (1.0 + np.exp(-logits))
        grad = (x_train.T @ ((prob - y_train) * weights)) / len(y_train)
        grad[1:] += l2 * beta[1:]
        beta -= lr * grad

    logits = np.clip(x_test @ beta, -40, 40)
    return 1.0 / (1.0 + np.exp(-logits))


def _cv_auc_brier(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    aucs = []
    briers = []
    for train_idx, test_idx in _make_splits(y, groups, seed):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        prob = _predict_logistic(x[train_idx], y_train, x[test_idx], seed)
        aucs.append(_auc_score(y_test, prob))
        briers.append(float(np.mean((prob - y_test) ** 2)))

    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.mean(briers))


def _q3_subset(df: pd.DataFrame, h_col: str, d_col: str, h_quantile: float, d_quantile: float, use_quadrant: bool):
    if use_quadrant and "quadrant" in df.columns:
        q3 = df[df["quadrant"].astype(str) == "Q3_lowH_highD"].copy()
        if len(q3):
            return q3
    h_cut = df[h_col].quantile(h_quantile)
    d_cut = df[d_col].quantile(d_quantile)
    return df[(df[h_col] <= h_cut) & (df[d_col] >= d_cut)].copy()


def _tertile_rates(q3: pd.DataFrame, proxy: str, target_col: str) -> pd.DataFrame:
    out = q3[[proxy, target_col]].dropna().copy()
    if out.empty:
        return pd.DataFrame(columns=["bin", "mean", "count"])
    if out[proxy].nunique() < 3:
        out["bin"] = "all"
    else:
        try:
            codes = pd.qcut(out[proxy], q=3, labels=False, duplicates="drop")
            n_bins = int(codes.max()) + 1 if not codes.isna().all() else 1
            names = {1: ["all"], 2: ["low", "high"], 3: ["low", "mid", "high"]}.get(
                n_bins, [f"bin_{idx}" for idx in range(n_bins)]
            )
            out["bin"] = codes.map(lambda idx: names[int(idx)] if pd.notna(idx) else None)
        except ValueError:
            out["bin"] = "all"
    return out.groupby("bin", observed=False)[target_col].agg(["mean", "count"]).reset_index()


def _safe_bin_value(bins: pd.DataFrame, name: str) -> float:
    rows = bins[bins["bin"].astype(str) == name]
    if rows.empty:
        return float("nan")
    return float(rows["mean"].iloc[0])


def _maybe_plot(bins: pd.DataFrame, proxy: str, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(5.5, 3.5))
    plt.bar(bins["bin"].astype(str), bins["mean"])
    plt.ylabel("reachability")
    plt.xlabel(proxy)
    plt.tight_layout()
    plt.savefig(output_dir / f"{proxy}_q3_reachability.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Merged token bank or token-bank directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--h-col", default=None, help="Entropy column. Auto: H_norm, Hs_topk_norm, Hs.")
    parser.add_argument("--d-col", default=None, help="Divergence column. Auto: D_norm, KLf_union, KLf.")
    parser.add_argument("--target-col", default=None, help="Reachability label. Auto: reach_t1, reach_t1_16.")
    parser.add_argument("--group-col", default=None, help="Prompt/sample grouping column.")
    parser.add_argument("--compat-cols", nargs="*", default=None)
    parser.add_argument("--h-quantile", type=float, default=0.5)
    parser.add_argument("--d-quantile", type=float, default=0.5)
    parser.add_argument("--use-quadrant", action="store_true", help="Prefer existing Q3_lowH_highD quadrant labels.")
    parser.add_argument("--min-q3-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = _read_any(Path(args.input))
    h_col = args.h_col or _first_existing(df, ["H_norm", "Hs_topk_norm", "Hs"])
    d_col = args.d_col or _first_existing(df, ["D_norm", "KLf_union", "KLf"])
    target_col = args.target_col or _first_existing(df, ["reach_t1", "reach_t1_16"])
    group_col = args.group_col or _first_existing(df, ["sample_index", "group_index", "prompt_id", "sample_ordinal"])
    compat_cols = args.compat_cols or [col for col in ["Cmass", "Coverlap", "CBC", "Cprefix_ema", "C_norm"] if col in df.columns]
    if not compat_cols:
        raise ValueError("No compatibility columns found. Pass --compat-cols explicitly.")

    needed = [h_col, d_col, target_col, group_col, *compat_cols]
    df = _finite_frame(df, needed)
    df[target_col] = df[target_col].astype(int)

    q3 = _q3_subset(df, h_col, d_col, args.h_quantile, args.d_quantile, args.use_quadrant)
    if len(q3) < args.min_q3_size:
        raise ValueError(f"Q3 too small: {len(q3)} rows; lower --min-q3-size for smoke tests.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y = q3[target_col].astype(int).to_numpy()
    groups = q3[group_col].to_numpy()
    base_x = q3[[h_col, d_col]].to_numpy(dtype=float)

    rows = []
    for proxy in compat_cols:
        full_x = q3[[h_col, d_col, proxy]].to_numpy(dtype=float)
        auc_hd, brier_hd = _cv_auc_brier(base_x, y, groups, args.seed)
        auc_hdc, brier_hdc = _cv_auc_brier(full_x, y, groups, args.seed)
        bins = _tertile_rates(q3, proxy, target_col)
        bins.to_csv(output_dir / f"{proxy}_reach_bins.csv", index=False)
        _maybe_plot(bins, proxy, output_dir)
        rows.append(
            {
                "proxy": proxy,
                "rows_total": int(len(df)),
                "q3_rows": int(len(q3)),
                "target_rate": float(q3[target_col].mean()),
                "auc_hd": auc_hd,
                "auc_hdc": auc_hdc,
                "delta_auc": auc_hdc - auc_hd if np.isfinite(auc_hdc) and np.isfinite(auc_hd) else float("nan"),
                "brier_hd": brier_hd,
                "brier_hdc": brier_hdc,
                "delta_brier": brier_hd - brier_hdc
                if np.isfinite(brier_hdc) and np.isfinite(brier_hd)
                else float("nan"),
                "reach_low": _safe_bin_value(bins, "low"),
                "reach_mid": _safe_bin_value(bins, "mid"),
                "reach_high": _safe_bin_value(bins, "high"),
                "h_col": h_col,
                "d_col": d_col,
                "target_col": target_col,
                "group_col": group_col,
            }
        )

    summary = pd.DataFrame(rows).sort_values("delta_auc", ascending=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    metadata = {
        "input": args.input,
        "h_col": h_col,
        "d_col": d_col,
        "target_col": target_col,
        "group_col": group_col,
        "compat_cols": compat_cols,
        "h_quantile": args.h_quantile,
        "d_quantile": args.d_quantile,
        "use_quadrant": args.use_quadrant,
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n")
    print(summary.to_string(index=False))
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
