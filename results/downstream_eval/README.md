# Downstream Evaluation Results — TA-OPD baseline reproduction

This folder holds the downstream-evaluation results of the TA-OPD baseline
reproduction run performed in this environment.

## What was evaluated

- **6 method configurations × 5 eval seeds = 30 runs** (see method table below).
- **5 downloadable benchmarks**: `aime24`, `aime25`, `humaneval`, `ifeval`,
  `hendrycks_math500`.
- `gpqa_diamond` is **not included** — it is a gated HF dataset
  (`Idavidrein/gpqa`) that requires an `HF_TOKEN` from an account with access
  rights. The GPQA-D column is therefore `n/a` in the table below. To fill it,
  provide `HF_TOKEN` (with access granted) and re-run only `--tasks gpqa_diamond`
  for the 30 (model, seed) pairs, then merge into the aggregation.

## Method / budget mapping

| Method (paper alias) | Folder | Budget used | Meaning |
|---|---|---|---|
| Base (no distillation) | `base` | — | Qwen3-1.7B-Base, no OPD supervision |
| Full OPD | `pure_opd` | 10% | Use all valid response tokens as direct OPD targets |
| TA-OPD | `ta_opd` | 0.5% | Select tokens by learnable disagreement (`dlearn_high`) |
| Entropy-only | `entropy` | 1% | Select by student entropy |
| TIP-style | `tip` | 1% | Select by entropy + raw teacher–student divergence |
| TA-OPD + Entropy | `ta_opd_entropy` | 1% | Soft-OR mixture of entropy and learnable disagreement (`ca_softor`) |

> **Budget note:** the original paper reports all methods at a uniform 10%
> budget. In this local reproduction only `pure_opd` reached 10%; the remaining
> methods were only trained up to 0.5–1.0% budget. The table therefore reports
> each method at *its highest locally available budget* and is **not directly
> comparable** to the paper's uniform-budget Table 3.

## Results (mean over 5 eval seeds, in %)

| Method (paper alias) | Budget | AIME24 | AIME25 | HumanEval | IFEval | MATH-500 | GPQA-D |
|---|---|---:|---:|---:|---:|---:|---:|
| Base (no distillation) | — | 0.00 | 3.33 | 51.83 | 22.18 | 0.40 | n/a |
| Full OPD | 10% | 0.00 | 0.00 | 41.46 | 17.56 | 0.00 | n/a |
| TA-OPD | 0.5% | 0.00 | 0.00 | 40.24 | 15.82 | 0.00 | n/a |
| Entropy-only | 1% | 3.33 | 6.67 | 40.24 | 16.52 | 0.00 | n/a |
| TIP-style | 1% | 3.33 | 3.33 | 41.46 | 16.52 | 0.00 | n/a |
| TA-OPD + Entropy | 1% | 3.33 | 6.67 | 38.41 | 16.27 | 0.00 | n/a |

Full mean ± std per seed are in `summary_table3.csv`. GPQA-D is `n/a` (see
above).

## Environment

- Teacher: Qwen3-4B; Student: Qwen3-1.7B-Base.
- Eval harness: `lm-eval` 0.4.12 (conda env `lmeval`).
- 8 × ~48 GB GPUs, 1 lm-eval instance per GPU, `--batch_size 8`,
  `aime24/25` `max_gen_toks=4096`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`.

## How to reproduce

From the repo root:

```bash
# 1) run downstream eval (6 methods x 5 seeds, writes outputs under the EVAL_ROOT)
bash 指令_下游评测_ta_opd.txt
# 2) aggregate into summary_table3.md / .csv (mean±std over 5 seeds)
python3 aggregate_table3.py
```

The evaluation script auto-skips already-completed (model, seed) pairs via a
`DONE` marker, so it is safe to re-run after interruptions.

## File layout

```
results/downstream_eval/
├── README.md                 # this file
├── summary_table3.md         # aggregated table (human-readable)
├── summary_table3.csv        # aggregated table (machine-readable, mean±std)
└── raw/                      # raw lm-eval outputs, one per (model, seed)
    ├── <method>/seed<N>/results_*.json
    └── ...
```

Raw outputs are the verbatim `results_*.json` produced by `lm-eval` (including
per-benchmark metrics, configs, and environment info) and are kept for full
traceability / re-aggregation.
