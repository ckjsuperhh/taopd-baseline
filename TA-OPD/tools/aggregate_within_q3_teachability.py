#!/usr/bin/env python3
"""Aggregate within-Q3 teachability intervention runs.

This script is intentionally read-only over run directories. It collects the
training token-bank diagnostics and fixed-context gain summaries into a stable
storyline directory so later paper figures do not depend on scattered run paths.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("/path/to/outputs/slime_opd")
DEFAULT_TAG_GLOB = "within_q3_teachability_seed*_20260514"
DEFAULT_OUT = DEFAULT_ROOT / "analysis" / "within_q3_teachability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tag-glob", default=DEFAULT_TAG_GLOB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[warn] failed to read {path}: {exc}")
        return pd.DataFrame()


def _ratio_from_name(name: str) -> float | None:
    m = re.search(r"_ratio(\d+)", name)
    if not m:
        return None
    # Existing slime OPD scripts encode 3% as ratio003, 5% as ratio005,
    # and 10% as ratio010.
    return int(m.group(1)) / 100.0


def _seed_from_tag(tag: str) -> str:
    m = re.search(r"(seed\d+)", tag)
    return m.group(1) if m else ""


def _mask_from_eval_name(name: str) -> str:
    m = re.match(r"(.+)_ratio\d+$", name)
    return m.group(1) if m else name


SAVE_RE = re.compile(
    r"^qwen3_1_7b_dapo_budget_k16_ratio(?P<ratio>\d+)_(?P<mask>.+)_max64_"
    r"(?P<seed>seed\d+)_(?P<tag>within_q3_teachability_seed\d+_20260514)$"
)


def _parse_save_dir(path: Path) -> dict[str, object] | None:
    m = SAVE_RE.match(path.name)
    if not m:
        return None
    return {
        "tag": m.group("tag"),
        "seed": m.group("seed"),
        "mask": m.group("mask"),
        "ratio": int(m.group("ratio")) / 100.0,
        "save_dir": path,
    }


def _save_dir_for(root: Path, tag: str, seed: str, mask: str, ratio: float | None) -> Path | None:
    if ratio is None:
        return None
    ratio_token = f"ratio{int(round(ratio * 100)):03d}"
    pat = f"qwen3_1_7b_dapo_budget_k16_{ratio_token}_{mask}_max64_{seed}_{tag}"
    path = root / pat
    if path.exists():
        return path
    hits = sorted(root.glob(f"*{ratio_token}_{mask}_max64_{seed}_{tag}"))
    return hits[0] if hits else path


def _read_token_bank(token_dir: Path) -> pd.DataFrame:
    if not token_dir.exists():
        return pd.DataFrame()
    parquet = sorted(token_dir.glob("*.parquet"))
    if parquet:
        parts = []
        for path in parquet:
            try:
                parts.append(pd.read_parquet(path))
            except Exception as exc:
                print(f"[warn] failed to read {path}: {exc}")
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    csvs = sorted(token_dir.glob("rollout_*.csv"))
    parts = []
    for path in csvs:
        try:
            parts.append(pd.read_csv(path))
        except Exception as exc:
            print(f"[warn] failed to read {path}: {exc}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _mean(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce")
    if vals.notna().sum() == 0:
        return None
    return float(vals.mean())


def _sum(df: pd.DataFrame, col: str) -> int | None:
    if df.empty or col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return int(vals.sum())


def _token_stats(df: pd.DataFrame) -> dict[str, float | int | None]:
    if df.empty:
        return {
            "token_rows": 0,
            "valid_tokens": None,
            "kept_tokens": None,
            "keep_ratio": None,
            "q3_tokens": None,
            "q3_kept_tokens": None,
            "kept_q3_frac": None,
        }

    valid = df[df["loss_mask_original"].astype(int) == 1] if "loss_mask_original" in df.columns else df
    kept = valid[valid["budget_keep"].astype(int) == 1] if "budget_keep" in valid.columns else pd.DataFrame()
    q3 = valid[valid["quadrant"] == "Q3_lowH_highD"] if "quadrant" in valid.columns else pd.DataFrame()
    q3_kept = kept[kept["quadrant"] == "Q3_lowH_highD"] if "quadrant" in kept.columns else pd.DataFrame()

    stats: dict[str, float | int | None] = {
        "token_rows": int(len(df)),
        "valid_tokens": int(len(valid)),
        "kept_tokens": int(len(kept)),
        "keep_ratio": len(kept) / max(len(valid), 1),
        "q3_tokens": int(len(q3)),
        "q3_kept_tokens": int(len(q3_kept)),
        "kept_q3_frac": len(q3_kept) / max(len(kept), 1) if len(kept) else None,
    }

    for prefix, subset in (("valid", valid), ("kept", kept), ("q3_valid", q3), ("q3_kept", q3_kept)):
        for col in (
            "H_norm",
            "D_norm",
            "C_norm",
            "DC_norm",
            "Dlearn",
            "Dincompat",
            "tip_score",
            "ca_softor_score",
            "split_ca_score",
            "KLf_union",
            "Cmass",
            "Cmass_true",
            "pos_norm",
            "resp_len",
        ):
            stats[f"{prefix}_mean_{col}"] = _mean(subset, col)

    return stats


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    tags = {p.name for p in args.root.glob(args.tag_glob) if p.is_dir()}
    save_entries = []
    for save_candidate in sorted(args.root.glob("qwen3_1_7b_dapo_budget_k16_ratio*_max64_seed*_within_q3_teachability_seed*_20260514")):
        parsed = _parse_save_dir(save_candidate)
        if parsed is None:
            continue
        tags.add(str(parsed["tag"]))
        save_entries.append(parsed)

    for tag in sorted(tags):
        tag_dir = args.root / tag
        seed = _seed_from_tag(tag)
        entries: dict[tuple[str, float | None], dict[str, object]] = {}

        if tag_dir.is_dir():
            for eval_dir in sorted(tag_dir.glob("*_ratio*")):
                if not eval_dir.is_dir():
                    continue
                mask = _mask_from_eval_name(eval_dir.name)
                ratio = _ratio_from_name(eval_dir.name)
                entries[(mask, ratio)] = {"mask": mask, "ratio": ratio, "eval_dir": eval_dir}

        for parsed in save_entries:
            if parsed["tag"] != tag:
                continue
            key = (str(parsed["mask"]), parsed["ratio"])
            entries.setdefault(key, {"mask": parsed["mask"], "ratio": parsed["ratio"]})
            entries[key]["save_dir"] = parsed["save_dir"]

        for entry in entries.values():
            mask = str(entry["mask"])
            ratio = entry["ratio"]
            eval_dir = Path(entry.get("eval_dir", tag_dir / f"{mask}_ratio{int(round(float(ratio) * 100)):03d}"))
            save_dir = entry.get("save_dir") or _save_dir_for(args.root, tag, seed, mask, float(ratio) if ratio is not None else None)

            gain_path = eval_dir / "gain" / "q3_bootstrap_matching_summary.csv"
            gain = _safe_read_csv(gain_path)
            gain_row = gain.iloc[0].to_dict() if not gain.empty else {}

            token_dir = save_dir / "token_bank" if save_dir is not None else Path()
            token_df = _read_token_bank(token_dir)
            token_stats = _token_stats(token_df)

            summary_path = token_dir / "summary.csv"
            rollout_summary = _safe_read_csv(summary_path)
            rollout_summary_path = ""
            if not rollout_summary.empty:
                rollout_summary_path = str(summary_path)
                token_stats["rollout_count"] = int(len(rollout_summary))
                token_stats["summary_keep_ratio_mean"] = float(
                    pd.to_numeric(rollout_summary.get("keep_ratio"), errors="coerce").mean()
                )
            else:
                token_stats["rollout_count"] = 0
                token_stats["summary_keep_ratio_mean"] = None

            row = {
                "tag": tag,
                "seed": seed,
                "mask": mask,
                "ratio": ratio,
                "eval_dir": str(eval_dir),
                "save_dir": str(save_dir) if save_dir is not None else "",
                "gain_summary": str(gain_path) if gain_path.exists() else "",
                "token_bank_dir": str(token_dir) if token_dir.exists() else "",
                "rollout_summary": rollout_summary_path,
                **gain_row,
                **token_stats,
            }
            rows.append(row)

            if not gain_path.exists() or token_df.empty:
                missing.append(
                    {
                        "tag": tag,
                        "seed": seed,
                        "mask": mask,
                        "ratio": ratio,
                        "missing_gain": int(not gain_path.exists()),
                        "missing_token_bank": int(token_df.empty),
                        "eval_dir": str(eval_dir),
                        "save_dir": str(save_dir) if save_dir is not None else "",
                    }
                )

    detail = pd.DataFrame(rows)
    detail_path = args.out_dir / "within_q3_teachability_detail_current.csv"
    detail.to_csv(detail_path, index=False)

    if not detail.empty:
        numeric = [
            "bootstrap_mean_diff",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "matching_mean_diff",
            "matching_median_diff",
            "keep_ratio",
            "kept_mean_Dlearn",
            "kept_mean_Dincompat",
            "kept_mean_C_norm",
            "kept_mean_D_norm",
            "q3_kept_mean_Dlearn",
            "q3_kept_mean_Dincompat",
            "q3_kept_mean_C_norm",
        ]
        for col in numeric:
            if col in detail.columns:
                detail[col] = pd.to_numeric(detail[col], errors="coerce")
        agg = (
            detail.groupby(["mask", "ratio"], dropna=False)
            .agg(
                seeds=("seed", "nunique"),
                runs=("seed", "count"),
                bootstrap_mean_diff=("bootstrap_mean_diff", "mean"),
                bootstrap_std=("bootstrap_mean_diff", "std"),
                matching_mean_diff=("matching_mean_diff", "mean"),
                keep_ratio=("keep_ratio", "mean"),
                kept_mean_Dlearn=("kept_mean_Dlearn", "mean"),
                kept_mean_Dincompat=("kept_mean_Dincompat", "mean"),
                kept_mean_C_norm=("kept_mean_C_norm", "mean"),
                q3_kept_mean_Dlearn=("q3_kept_mean_Dlearn", "mean"),
                q3_kept_mean_Dincompat=("q3_kept_mean_Dincompat", "mean"),
                q3_kept_mean_C_norm=("q3_kept_mean_C_norm", "mean"),
            )
            .reset_index()
            .sort_values(["ratio", "mask"])
        )
    else:
        agg = pd.DataFrame()

    agg_path = args.out_dir / "within_q3_teachability_aggregate_current.csv"
    agg.to_csv(agg_path, index=False)

    missing_path = args.out_dir / "within_q3_teachability_missing_current.csv"
    pd.DataFrame(missing).to_csv(missing_path, index=False)

    manifest = {
        "root": str(args.root),
        "tag_glob": args.tag_glob,
        "detail_csv": str(detail_path),
        "aggregate_csv": str(agg_path),
        "missing_csv": str(missing_path),
        "num_runs_indexed": int(len(detail)),
        "num_missing_or_partial": int(len(missing)),
    }
    manifest_path = args.out_dir / "within_q3_teachability_manifest_current.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
