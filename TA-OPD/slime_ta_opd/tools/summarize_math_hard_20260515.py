from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


STORY_ROOT = Path("/path/to/outputs/slime_opd/storyline_20260513")
OUT_ROOT = STORY_ROOT / "downstream_smoke_20260515" / "math_hard"
SUMMARY_CSV = OUT_ROOT / "math_hard_summary.csv"
SUMMARY_MD = OUT_ROOT / "math_hard_summary.md"
LIVE_STATUS = STORY_ROOT / "live_queue_status_20260515.md"
CURRENT_STATUS = STORY_ROOT / "current_research_status_20260515.md"
ARCHIVE_SCRIPT = STORY_ROOT / "collect_opd_research_assets_20260513.py"

MODEL_ORDER = [
    "base_qwen3_1p7b",
    "dlearn_high_ratio003_seed2",
    "q3_highc_ratio003_seed2",
    "tip_ratio003_seed2",
    "divergence_ratio003_seed2",
]


def _metric_from_result(result: dict, preferred: tuple[str, ...]) -> tuple[float | None, str]:
    for key in preferred:
        value = result.get(key)
        if isinstance(value, (int, float)):
            return float(value), key
    for key, value in result.items():
        if isinstance(value, (int, float)) and not key.endswith("_stderr"):
            return float(value), key
    return None, ""


def _latest_json(model_dir: Path) -> Path | None:
    jsons = sorted(model_dir.glob("**/results_*.json"), key=lambda path: path.stat().st_mtime)
    return jsons[-1] if jsons else None


def collect_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not OUT_ROOT.exists():
        return rows

    for model_dir in sorted(path for path in OUT_ROOT.iterdir() if path.is_dir()):
        result_path = _latest_json(model_dir)
        if result_path is None:
            continue

        data = json.loads(result_path.read_text())
        results = data.get("results", {})
        group_result = results.get("leaderboard_math_hard", {})
        exact, exact_metric = _metric_from_result(
            group_result,
            ("exact_match,none", "exact_match", "exact_match_original,none", "exact_match_original"),
        )

        row = {
            "model": model_dir.name,
            "leaderboard_math_hard": "" if exact is None else f"{exact:.4f}",
            "metric": exact_metric,
            "mtime_utc": datetime.fromtimestamp(result_path.stat().st_mtime, timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "json": str(result_path),
        }

        for task_name, task_result in sorted(results.items()):
            if not task_name.startswith("leaderboard_math_") or task_name == "leaderboard_math_hard":
                continue
            value, metric = _metric_from_result(
                task_result,
                ("exact_match,none", "exact_match", "exact_match_original,none", "exact_match_original"),
            )
            short_name = task_name.removeprefix("leaderboard_math_").removesuffix("_hard")
            row[f"{short_name}"] = "" if value is None else f"{value:.4f}"
            row[f"{short_name}_metric"] = metric

        rows.append(row)

    rows.sort(key=lambda row: (MODEL_ORDER.index(row["model"]) if row["model"] in MODEL_ORDER else 999, row["model"]))
    return rows


def write_summary(rows: list[dict[str, str]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "leaderboard_math_hard",
        "metric",
        "algebra",
        "counting_and_prob",
        "geometry",
        "intermediate_algebra",
        "num_theory",
        "prealgebra",
        "precalculus",
        "mtime_utc",
        "json",
    ]
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    base = next((row for row in rows if row["model"] == "base_qwen3_1p7b"), None)
    lines = [
        "# MATH Held-Out Summary",
        "",
        "Task: `leaderboard_math_hard` from lm-eval. This is a reproducible MATH held-out proxy on the A800 server, not the external Math500 table.",
        "",
        "| model | score | delta vs base | algebra | counting | geometry | int. algebra | num theory | prealgebra | precalc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        score = row.get("leaderboard_math_hard", "")
        delta = ""
        if base and score and base.get("leaderboard_math_hard"):
            delta = f"{float(score) - float(base['leaderboard_math_hard']):+.4f}"
        lines.append(
            "| `{model}` | {score} | {delta} | {algebra} | {counting} | {geometry} | {intermediate} | {num} | {prealg} | {precalc} |".format(
                model=row["model"],
                score=score,
                delta=delta,
                algebra=row.get("algebra", ""),
                counting=row.get("counting_and_prob", ""),
                geometry=row.get("geometry", ""),
                intermediate=row.get("intermediate_algebra", ""),
                num=row.get("num_theory", ""),
                prealg=row.get("prealgebra", ""),
                precalc=row.get("precalculus", ""),
            )
        )

    lines.extend(["", "## Result Files", ""])
    for row in rows:
        lines.append(f"- `{row['model']}`: `{row['json']}`")
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def active_math_processes() -> str:
    try:
        return subprocess.check_output(
            "ps -eo pid,etime,cmd | grep -E 'run_downstream_math_hard|leaderboard_math_hard|lm-eval run' | grep -v grep",
            shell=True,
            text=True,
            timeout=10,
        ).strip()
    except Exception:
        return ""


def update_status(rows: list[dict[str, str]]) -> None:
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except Exception:
        gpu = "unavailable"

    procs = active_math_processes()
    collected = ", ".join(row["model"] for row in rows) if rows else "none"
    LIVE_STATUS.write_text(
        "\n".join(
            [
                "# Live Queue Status - 2026-05-15",
                "",
                f"Last update UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Downstream Eval",
                "",
                "- Full GSM8K_COT: complete.",
                "- AIME24/25 capped-4096: complete; smoke-only, neutral/low-resolution.",
                f"- MATH held-out (`leaderboard_math_hard`): {'active' if procs else 'no active process found'}.",
                f"- MATH collected models: {collected}.",
                f"- MATH summary: `{SUMMARY_MD}`.",
                f"- GPU snapshot: `{gpu.replace(chr(10), '; ')}`.",
                f"- Matching processes: `{procs.replace(chr(10), ' | ') if procs else 'none found'}`.",
                "",
                "## Interpretation",
                "",
                "The MATH held-out run is the next downstream check with better resolution than capped AIME. It compares base, Dlearn-high, Q3-highC, TIP, and high-D/divergence under the same lm-eval task.",
                "",
            ]
        )
    )

    if rows and not procs:
        old = CURRENT_STATUS.read_text(errors="ignore") if CURRENT_STATUS.exists() else ""
        marker = "## Downstream MATH held-out update"
        block = [
            "",
            "",
            f"{marker} ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})",
            "",
            f"Summary: `{SUMMARY_MD}`. Task is `leaderboard_math_hard`, a reproducible MATH held-out proxy on this A800 server. Collected models: {collected}.",
        ]
        if marker not in old:
            CURRENT_STATUS.write_text(old.rstrip() + "\n".join(block) + "\n")


def refresh_archive() -> None:
    if not ARCHIVE_SCRIPT.exists():
        return
    env = os.environ.copy()
    env.update(
        {
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HF_HOME": "/path/to/hf_cache",
            "HF_HUB_CACHE": "/path/to/hf_cache/hub",
            "TRANSFORMERS_CACHE": "/path/to/hf_cache/transformers",
            "HF_DATASETS_CACHE": "/path/to/hf_cache/datasets",
        }
    )
    subprocess.run(["python3", str(ARCHIVE_SCRIPT)], cwd=str(STORY_ROOT), env=env, timeout=120, check=False)


def main() -> None:
    rows = collect_rows()
    write_summary(rows)
    update_status(rows)
    refresh_archive()
    print(SUMMARY_CSV)
    print(SUMMARY_MD)
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()

