#!/usr/bin/env python3
"""Convert a slime Megatron torch_dist checkpoint iteration to HuggingFace format."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _latest_iteration(root: Path) -> int:
    latest = root / "latest_checkpointed_iteration.txt"
    if not latest.exists():
        raise FileNotFoundError(f"Missing {latest}; pass --iteration or --input-dir explicitly.")
    return int(latest.read_text().strip())


def _resolve_input_dir(args) -> tuple[Path, str]:
    if args.input_dir:
        path = Path(args.input_dir)
        return path, path.name

    root = Path(args.checkpoint_root)
    iteration = _latest_iteration(root) if args.iteration == "latest" else int(args.iteration)
    dirname = f"iter_{iteration:07d}"
    return root / dirname, dirname


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default=None, help="slime --save directory containing iter_XXXXXXX dirs.")
    parser.add_argument("--input-dir", default=None, help="Specific torch_dist iteration directory containing common.pt.")
    parser.add_argument("--iteration", default="latest", help="'latest' or integer iteration under --checkpoint-root.")
    parser.add_argument("--origin-hf-dir", required=True, help="Original HF model dir for tokenizer/config/assets.")
    parser.add_argument("--output-dir", required=True, help="HF output directory.")
    parser.add_argument("--slime-dir", default=None, help="Repo root. Default: parent of this script.")
    parser.add_argument("--parallel", action="store_true", help="Use convert_torch_dist_to_hf_parallel.py.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=5 * 1024**3)
    parser.add_argument("--load-max-workers", type=int, default=2)
    parser.add_argument("--save-max-workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input_dir and not args.checkpoint_root:
        raise ValueError("Pass either --input-dir or --checkpoint-root.")

    slime_dir = Path(args.slime_dir) if args.slime_dir else Path(__file__).resolve().parents[1]
    input_dir, iteration_name = _resolve_input_dir(args)
    output_dir = Path(args.output_dir)

    if not (input_dir / "common.pt").exists():
        raise FileNotFoundError(f"{input_dir} does not look like a torch_dist checkpoint iteration; common.pt missing.")
    if output_dir.exists() and args.force and not args.dry_run:
        shutil.rmtree(output_dir)

    converter = slime_dir / "tools" / (
        "convert_torch_dist_to_hf_parallel.py" if args.parallel else "convert_torch_dist_to_hf.py"
    )
    cmd = [
        sys.executable,
        str(converter),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--origin-hf-dir",
        args.origin_hf_dir,
        "--force",
        "--chunk-size",
        str(args.chunk_size),
    ]
    if args.vocab_size is not None:
        cmd += ["--vocab-size", str(args.vocab_size)]
    if args.parallel:
        cmd += [
            "--load-max-workers",
            str(args.load_max_workers),
            "--save-max-workers",
            str(args.save_max_workers),
        ]

    env_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{slime_dir}:{env_pythonpath}" if env_pythonpath else str(slime_dir)
    print(f"input_dir={input_dir}")
    print(f"output_dir={output_dir}")
    print(f"iteration={iteration_name}")
    _run(cmd, cwd=slime_dir, dry_run=args.dry_run)
    print(f"converted_hf={output_dir}")


if __name__ == "__main__":
    main()
