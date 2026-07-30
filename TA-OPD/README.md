# TA-OPD: Teachability-Aware On-Policy Distillation

This anonymized repository contains the implementation of **Teachability-Aware
On-Policy Distillation (TA-OPD)**, the method introduced in *Not All
Disagreement Is Learnable: Token Teachability in On-Policy Distillation*.
TA-OPD is a token-selection extension for sampled-token on-policy distillation:
it keeps the standard rollout and teacher-scoring pipeline, but applies a hard
target-token mask so that direct KL supervision is concentrated on teacher
signals that are both large and locally reachable by the student.

TA-OPD is implemented on top of public open-source infrastructure, including
[NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and
[THUDM/slime](https://github.com/THUDM/slime). Upstream license and attribution
notices are preserved in this repository.

## Repository layout

- `slime_ta_opd/`: the full modified slime tree used for the TA-OPD training,
  fixed-context diagnostics, and analysis scripts. Use this directory when you
  want a runnable training stack.
- `ta_opd/slime_patch/`: a compact view of the core TA-OPD rollout files. This
  is useful for code review or for manually porting TA-OPD into another slime
  checkout, but it is not a standalone training framework.
- `scripts/train/`: portable launch wrappers for Full OPD, TA-OPD,
  TA-OPD+Entropy, entropy-only, TIP-style, high-divergence, and random
  selectors.
- `scripts/diagnostics/`: fixed-context diagnostic drivers for measuring
  token-level OPD gain and support-based decomposition.
- `scripts/eval/`: lightweight downstream evaluation wrappers used during
  development.
- `tools/`: token-bank merging, fixed-context scoring, regression analysis,
  within-region analysis, support robustness checks, and checkpoint conversion
  utilities.
- `docs/`: method entry points and experiment workflow notes.

Cluster-specific paths from the original experiments have been replaced by
placeholder defaults such as `/path/to/slime-main`, `/path/to/models`, and
`/path/to/outputs`. Override them with environment variables before running.

## Method aliases

| Paper name | Script alias | Underlying mask | Meaning |
|---|---|---|---|
| Full OPD | `pure_opd` | `full` | Use all valid response tokens as direct OPD targets. |
| TA-OPD | `teachability` | `dlearn_high` | Select tokens by learnable disagreement. |
| TA-OPD + Entropy | `teachability_entropy` | `ca_softor` | Soft-OR mixture of entropy and learnable disagreement. |
| Entropy-only | `entropy` | `entropy` | Select by student entropy. |
| TIP-style | `tip` | `tip` | Select by entropy and raw teacher-student divergence. |
| High divergence | `divergence` | `divergence` | Select by raw divergence only. |
| Random | `random` | `random` | Same-budget random token control. |

The released implementation is **supervision-token efficient**. Unselected
response tokens remain in the sequence context, but their direct OPD loss
contribution is masked out. It should not be described as sparse attention,
sequence pruning, or proportional wall-clock compute reduction.

## Key implementation files

The full modified slime tree is the recommended entry point. The TA-OPD changes
are concentrated in:

- `slime_ta_opd/slime/utils/arguments.py`
- `slime_ta_opd/slime/ray/rollout.py`
- `slime_ta_opd/slime/rollout/sglang_rollout.py`
- `slime_ta_opd/slime/rollout/on_policy_distillation.py`
- `slime_ta_opd/slime/rollout/tip_compat.py`

The original slime training entry point remains `slime_ta_opd/train.py`.

## Quick start with the full modified slime tree

```bash
cd slime_ta_opd
export SLIME_DIR=$PWD
export MEGATRON_LM_DIR=/path/to/Megatron-LM
export OUTPUT_ROOT=/path/to/outputs/slime_opd
export TEACHER_MODEL=/path/to/teacher-hf
export STUDENT_HF=/path/to/student-hf
export STUDENT_TORCH_DIST=/path/to/student-torch-dist
export PROMPT_DATA=/path/to/data/train.jsonl

export OPD_TOPK_METRICS_K=16
export METHOD_LIST="pure_opd:1.0:10 teachability:0.10:20 entropy:0.10:30 teachability_entropy:0.10:40 tip:0.10:50"

bash run_teachability_opd_method_suite_20260515.sh
```

For a smaller sanity check, reduce the number of rollouts and methods:

```bash
export NUM_ROLLOUT=50
export METHOD_LIST="pure_opd:1.0:10 teachability:0.10:20 tip:0.10:30"
bash run_teachability_opd_method_suite_20260515.sh
```

## Fixed-context diagnostics

The diagnostic workflow freezes student-generated contexts and re-scores the
same positions before and after OPD training. It is used to measure whether a
token-level teacher signal yields local KL reduction on the same context.

Typical entry points:

```bash
bash scripts/diagnostics/run_4b_to_1p7b_heldout_fixed_context_300.sh
bash scripts/diagnostics/run_8b_to_4b_fixed_context_300.sh
bash scripts/diagnostics/run_14b_to_4b_fixed_context_300.sh
```

The corresponding analysis tools include:

- `tools/export_fixed_context_bank.py`
- `tools/eval_fixed_context_bank.py`
- `tools/analyze_fixed_context_gain.py`
- `tools/matched_fixed_context_topn.py`
- `tools/support_definition_robustness.py`

## Porting only the core patch

If you already have a compatible slime checkout, copy the compact patch files
from `ta_opd/slime_patch/` and make sure the corresponding argument and rollout
call sites are also updated:

```bash
cp ta_opd/slime_patch/slime/rollout/tip_compat.py /path/to/slime-main/slime/rollout/tip_compat.py
cp ta_opd/slime_patch/slime/rollout/on_policy_distillation.py /path/to/slime-main/slime/rollout/on_policy_distillation.py
```

For exact reproducibility, prefer `slime_ta_opd/`, because it contains the full
modified training tree and launch scripts.

## Downstream evaluation results (baseline reproduction)

The TA-OPD baseline was reproduced end-to-end in a local environment and
evaluated on five downloadable benchmarks. Raw outputs and the aggregated
table live under [`results/downstream_eval/`](../results/downstream_eval).

| Method (paper alias) | Budget | AIME24 | AIME25 | HumanEval | IFEval | MATH-500 | GPQA-D |
|---|---|---:|---:|---:|---:|---:|---:|
| Base (no distillation) | — | 0.00 | 3.33 | 51.83 | 22.18 | 0.40 | n/a |
| Full OPD | 10% | 0.00 | 0.00 | 41.46 | 17.56 | 0.00 | n/a |
| TA-OPD | 0.5% | 0.00 | 0.00 | 40.24 | 15.82 | 0.00 | n/a |
| Entropy-only | 1% | 3.33 | 6.67 | 40.24 | 16.52 | 0.00 | n/a |
| TIP-style | 1% | 3.33 | 3.33 | 41.46 | 16.52 | 0.00 | n/a |
| TA-OPD + Entropy | 1% | 3.33 | 6.67 | 38.41 | 16.27 | 0.00 | n/a |

Numbers are percentages, mean over 5 eval seeds; full mean ± std are in
[`results/downstream_eval/summary_table3.csv`](../results/downstream_eval/summary_table3.csv).

**Caveats (read before comparing to the paper):**

- *Budgets are not uniform.* The paper reports all methods at a uniform 10%
  budget; in this run only `pure_opd` reached 10%, while the others were only
  trained to 0.5–1.0%. Each row above uses that method's highest locally
  available budget, so the table is **not directly comparable** to the paper's
  uniform-budget Table 3.
- *GPQA-Diamond is missing* (`n/a`): it is a gated HF dataset
  (`Idavidrein/gpqa`) requiring an `HF_TOKEN` with access granted. Provide the
  token and re-run `--tasks gpqa_diamond` for the 30 (model, seed) pairs to fill
  the column.
- *Setup*: teacher Qwen3-4B → student Qwen3-1.7B-Base, `lm-eval` 0.4.12,
  8 × ~48 GB GPUs, one instance per GPU, `--batch_size 8`.

## License and attribution

This repository includes code derived from slime and interfaces with
Megatron-LM. The relevant upstream license and attribution notices are retained
with the copied source tree. Users should also follow the licenses of any model
checkpoints, datasets, and evaluation suites they use with TA-OPD.
