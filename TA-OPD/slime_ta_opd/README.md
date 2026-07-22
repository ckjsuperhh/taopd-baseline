# Modified slime Tree for TA-OPD

This directory contains the full modified slime tree used to implement
**Teachability-Aware On-Policy Distillation (TA-OPD)**. It is included so the
training and diagnostic scripts can be run without reconstructing the exact
slime-side changes from a patch.

TA-OPD is implemented on top of public open-source infrastructure, including
[NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and
[THUDM/slime](https://github.com/THUDM/slime). Upstream license and attribution
notices are preserved in this repository.

## What changed

The core TA-OPD implementation is concentrated in:

- `slime/utils/arguments.py`
- `slime/ray/rollout.py`
- `slime/rollout/sglang_rollout.py`
- `slime/rollout/on_policy_distillation.py`
- `slime/rollout/tip_compat.py`

The top-level training entry point remains `train.py`, following the original
slime workflow. The added scripts in this directory launch selector sweeps,
budget sweeps, fixed-context diagnostics, and downstream smoke evaluations.

## Method aliases

| Paper name | Script alias | Underlying mask |
|---|---|---|
| Full OPD | `pure_opd` | `full` |
| TA-OPD | `teachability` | `dlearn_high` |
| TA-OPD + Entropy | `teachability_entropy` | `ca_softor` |
| Entropy-only | `entropy` | `entropy` |
| TIP-style | `tip` | `tip` |
| High divergence | `divergence` | `divergence` |
| Random | `random` | `random` |

TA-OPD applies a hard target-token mask after full-sequence rollout and
teacher/student log-probability computation. The method is supervision-token
efficient, not a sparse-attention or sequence-pruning system.

## Quick start

Set local paths explicitly before launching:

```bash
export SLIME_DIR=$PWD
export MEGATRON_LM_DIR=/path/to/Megatron-LM
export OUTPUT_ROOT=/path/to/outputs/slime_opd
export TEACHER_MODEL=/path/to/teacher-hf
export STUDENT_HF=/path/to/student-hf
export STUDENT_TORCH_DIST=/path/to/student-torch-dist
export PROMPT_DATA=/path/to/data/train.jsonl
export OPD_TOPK_METRICS_K=16
```

Run a selector suite:

```bash
export METHOD_LIST="pure_opd:1.0:10 teachability:0.10:20 entropy:0.10:30 teachability_entropy:0.10:40 tip:0.10:50"
bash run_teachability_opd_method_suite_20260515.sh
```

Run fixed-context diagnostics:

```bash
bash run_4b_to_1p7b_heldout_fixed_context_300_20260517.sh
bash run_8b_to_4b_fixed_context_300_20260517.sh
bash run_14b_to_4b_fixed_context_300_20260517.sh
```

Additional method notes are in `TA_OPD_README.md` and
`docs/opd_teachability_method_entrypoints_20260515.md`.

## Upstream documentation

This subtree is derived from slime. For the upstream framework documentation,
see [THUDM/slime](https://github.com/THUDM/slime) and the copied `docs/`
directory. For Megatron-LM, see
[NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
