#!/usr/bin/env python3
"""Convert a Megatron checkpoint to HF, then run fixed-context OPD metrics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--iteration", default="latest")
    parser.add_argument("--origin-hf-dir", required=True)
    parser.add_argument("--teacher-hf-dir", required=True)
    parser.add_argument("--context-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-metrics", default=None)
    parser.add_argument("--slime-dir", default=None)
    parser.add_argument("--student-device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-response-tokens", type=int, default=None)
    parser.add_argument("--parallel-convert", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    slime_dir = Path(args.slime_dir) if args.slime_dir else Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    hf_dir = output_dir / "student_hf"
    metrics_out = output_dir / "fixed_context_metrics.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_cmd = [
        sys.executable,
        str(slime_dir / "tools" / "convert_megatron_checkpoint_to_hf.py"),
        "--origin-hf-dir",
        args.origin_hf_dir,
        "--output-dir",
        str(hf_dir),
        "--slime-dir",
        str(slime_dir),
    ]
    if args.checkpoint_root:
        convert_cmd += ["--checkpoint-root", args.checkpoint_root, "--iteration", args.iteration]
    if args.input_dir:
        convert_cmd += ["--input-dir", args.input_dir]
    if args.parallel_convert:
        convert_cmd += ["--parallel"]
    if args.force_convert:
        convert_cmd += ["--force"]

    eval_cmd = [
        sys.executable,
        str(slime_dir / "tools" / "eval_fixed_context_bank.py"),
        "--context-bank",
        args.context_bank,
        "--student",
        str(hf_dir),
        "--teacher",
        args.teacher_hf_dir,
        "--output",
        str(metrics_out),
        "--student-device",
        args.student_device,
        "--teacher-device",
        args.teacher_device,
        "--dtype",
        args.dtype,
        "--topk",
        str(args.topk),
    ]
    if args.baseline_metrics:
        eval_cmd += ["--baseline-metrics", args.baseline_metrics]
    if args.max_samples is not None:
        eval_cmd += ["--max-samples", str(args.max_samples)]
    if args.max_response_tokens is not None:
        eval_cmd += ["--max-response-tokens", str(args.max_response_tokens)]

    _run(convert_cmd, cwd=slime_dir, dry_run=args.dry_run)
    _run(eval_cmd, cwd=slime_dir, dry_run=args.dry_run)
    print(f"student_hf={hf_dir}")
    print(f"metrics={metrics_out}")


if __name__ == "__main__":
    main()
