# Experiment workflows

This note summarizes the scripts extracted from the original TA-OPD
experiments.  The scripts are intentionally written as templates: set model,
data, environment, and output paths explicitly on your machine.

## Training selectors

Use `scripts/train/run_teachability_opd_method_suite.sh` to map paper-facing
method names to the underlying slime budget masks.  The wrapper calls
`scripts/train/run_opd_budget_ratio_sweep.sh`, which performs the actual OPD
training launches.

Common variables:

```bash
export SLIME_DIR=/path/to/slime-main
export MEGATRON_LM_DIR=/path/to/Megatron-LM
export OUTPUT_ROOT=/path/to/outputs/slime_opd
export TEACHER_MODEL=/path/to/teacher-hf
export STUDENT_HF=/path/to/student-hf
export STUDENT_TORCH_DIST=/path/to/student-torch-dist
export PROMPT_DATA=/path/to/data/train.jsonl
export OPD_TOPK_METRICS_K=16
```

## Fixed-context diagnostics

The `scripts/diagnostics/` scripts run the closed-loop diagnostic used in the
paper:

1. train or load a checkpoint,
2. merge token banks,
3. export a fixed context bank,
4. score the same contexts before/after training,
5. compute token-level gain and support-based decomposition.

The analysis tools are:

- `tools/export_fixed_context_bank.py`
- `tools/eval_fixed_context_bank.py`
- `tools/analyze_fixed_context_gain.py`
- `tools/matched_fixed_context_topn.py`
- `tools/support_definition_robustness.py`

## Downstream checks

The scripts in `scripts/eval/` are lightweight wrappers for benchmark smoke
checks.  They are included for reproducibility of the development workflow;
large-scale benchmark evaluation should use your production evaluation stack.
