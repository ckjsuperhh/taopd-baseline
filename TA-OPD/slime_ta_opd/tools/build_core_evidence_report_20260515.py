#!/usr/bin/env python3
"""Build a compact paper-facing evidence report for OPD teachability."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path("/path/to/outputs/slime_opd/storyline_20260513")
OUT = ROOT / "core_finding_evidence_20260515.md"


def fmt(x: object, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def table_from_df(df: pd.DataFrame, cols: list[str], n: int | None = None) -> list[str]:
    if df.empty:
        return ["_Missing._", ""]
    use = df[cols].copy()
    if n is not None:
        use = use.head(n)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in use.to_dict("records"):
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |")
    lines.append("")
    return lines


def main() -> None:
    decomp = read_csv(ROOT / "decomposition_paper_readable.csv")
    baselines = read_csv(ROOT / "p1_p2_tip_baselines/p1_p2_tip_baseline_aggregate_current.csv")
    budget = read_csv(ROOT / "budget_intervention_aggregate_current.csv")
    within_live = read_csv(ROOT / "within_q3_teachability/within_q3_teachability_aggregate_current.csv")
    within_bucket = read_csv(ROOT / "p0_within_q3_teachability/within_q3_high_low_summary.csv")
    matched = read_csv(ROOT / "matched_fixed_context_topn/matched_fixed_context_topn_summary.csv")
    math = read_csv(ROOT / "downstream_smoke_20260515/math_hard/math_hard_summary.csv")

    lines: list[str] = [
        "# OPD Teachability Core Evidence",
        "",
        "This file is a paper-facing index of the current evidence. It separates the scientific finding from engineering-level compute claims: the current method is selective direct supervision via a loss mask, not sequence-level compute pruning.",
        "",
        "## Core Claim",
        "",
        "Not all teacher-student disagreement is learnable. Useful OPD supervision is concentrated in tokens where teacher corrections are both nontrivial and compatible with the student's local support. This yields a small direct-supervision budget that preserves most downstream performance without claiming proportional wall-clock savings.",
        "",
        "## 1. Decomposition",
        "",
        "Dlearn has a consistently larger standardized coefficient than Dincompat across K, and the Dlearn-Dincompat gap is prompt-cluster bootstrap positive.",
        "",
    ]
    lines += table_from_df(
        decomp,
        [
            "K",
            "beta_Dlearn",
            "beta_Dincompat",
            "beta_gap",
            "beta_gap_ci_low",
            "beta_gap_ci_high",
            "delta_r2_Ddecomp_Hint_vs_HD",
            "delta_r2_GAM_Ddecomp_vs_GAM_HD",
        ],
    )

    lines += [
        "## 2. Selective OPD Baselines",
        "",
        "At equal nominal budget, Dlearn-high is the strongest live selective-OPD baseline in the current P1/P2 aggregate. This is the main intervention evidence.",
        "",
    ]
    if not baselines.empty:
        sub = baselines.copy()
        sub["ratio"] = pd.to_numeric(sub["ratio"], errors="coerce")
        sub = sub[sub["ratio"].isin([0.03, 0.05])].sort_values(["ratio", "bootstrap_mean"], ascending=[True, False])
        lines += table_from_df(
            sub,
            ["mask", "ratio", "n", "keep_ratio_mean", "bootstrap_mean", "matching_mean", "gain_per_keep_mean"],
            n=30,
        )
    else:
        lines += ["_Missing._", ""]

    lines += [
        "## 3. Budget Curve",
        "",
        "The current budget sweep supports a low-budget sweet spot: Dlearn-high improves from 1% to 3%-5%, then does not keep increasing at 10%.",
        "",
    ]
    if not budget.empty:
        sub = budget[budget["mask"].astype(str) == "dlearn_high"].sort_values("ratio")
        lines += table_from_df(sub, ["mask", "ratio", "n", "keep_ratio_mean", "bootstrap_mean", "matching_mean"])
    else:
        lines += ["_Missing._", ""]

    lines += [
        "## 4. Within-Q3 Evidence",
        "",
        "There are two complementary results. Offline fixed-context buckets show TIP-Q3 is heterogeneous and that compatibility/shared support is the clean separator inside Q3. Live within-Q3 intervention shows Q3 teachability-high is positive while teachability-low is negative and Dincompat-high is weak.",
        "",
        "### Offline Q3 Bucket Contrasts",
        "",
    ]
    if not within_bucket.empty:
        lines += table_from_df(
            within_bucket,
            ["K", "feature", "mean_low", "mean_high", "high_minus_low", "ci_low", "ci_high"],
        )
    else:
        lines += ["_Missing._", ""]

    lines += ["### Live Within-Q3 Intervention", ""]
    if not within_live.empty:
        lines += table_from_df(
            within_live.sort_values("bootstrap_mean_diff", ascending=False),
            ["mask", "ratio", "seeds", "bootstrap_mean_diff", "matching_mean_diff", "keep_ratio"],
        )
    else:
        lines += ["_Missing._", ""]

    lines += [
        "## 5. Exact Top-N Fixed-Context Matched Analysis",
        "",
        "Every selector keeps exactly the same number of fixed-context tokens. This is an offline confound check, not a live training result.",
        "",
    ]
    if not matched.empty:
        sub = matched[(pd.to_numeric(matched["K"], errors="coerce") == 16) & (pd.to_numeric(matched["ratio"], errors="coerce").isin([0.03, 0.05]))]
        sub = sub.sort_values(["ratio", "mean_G"], ascending=[True, False])
        lines += table_from_df(
            sub,
            ["K", "ratio", "selector", "mean_G", "ci_low", "ci_high", "gain_per_keep_ratio", "selected_q3_frac"],
            n=30,
        )
    else:
        lines += ["_Missing._", ""]

    lines += [
        "## 6. Downstream",
        "",
        "- GSM8K full supports the story: Dlearn-high ratio=0.03 seed2 is strongest among base, Q3-highC, and TIP.",
        "- AIME capped-4096 is a low-resolution smoke check: neutral overall, with TIP +1/30 on AIME24 only.",
        "- MATH held-out is currently the active higher-resolution downstream check.",
        "",
    ]
    if not math.empty:
        lines += table_from_df(math, ["model", "leaderboard_math_hard", "metric", "mtime_utc"])
    else:
        lines += ["MATH held-out summary is not available yet.", ""]

    lines += [
        "## Writing Guardrails",
        "",
        "- Use `target-token efficient`, `direct-supervision budget`, or `selective OPD supervision`.",
        "- Do not claim 90% wall-clock compute savings or peak-memory reduction for the current loss-mask implementation.",
        "- State that compute-sparse OPD via teachability-centered windows is future work or an engineering extension.",
        "- Keep the main story centered on learnable vs incompatible disagreement, not generic token importance.",
        "",
        "## Source Paths",
        "",
        f"- Decomposition: `{ROOT / 'decomposition_paper_readable.csv'}`",
        f"- P1/P2 baselines: `{ROOT / 'p1_p2_tip_baselines/p1_p2_tip_baseline_aggregate_current.csv'}`",
        f"- Budget curve: `{ROOT / 'budget_intervention_aggregate_current.csv'}`",
        f"- Within-Q3 offline: `{ROOT / 'p0_within_q3_teachability/within_q3_high_low_summary.csv'}`",
        f"- Within-Q3 live: `{ROOT / 'within_q3_teachability/within_q3_teachability_aggregate_current.csv'}`",
        f"- Exact top-N matched: `{ROOT / 'matched_fixed_context_topn/matched_fixed_context_topn_summary.csv'}`",
        f"- Downstream MATH held-out: `{ROOT / 'downstream_smoke_20260515/math_hard/math_hard_summary.csv'}`",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()

