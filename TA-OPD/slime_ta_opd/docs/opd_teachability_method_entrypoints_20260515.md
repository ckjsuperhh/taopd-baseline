# OPD Teachability Method Entrypoints

This note defines the paper-facing method names and their current slime implementation.

## Writing Position

The current implementation is **selective direct supervision**: it applies a hard loss mask over response tokens after full-sequence rollout and teacher/student log-prob computation. It should be described as target-token efficient or supervision-token efficient, not as compute pruning.

## Method Names

| Paper name | Script method | Underlying mask | Meaning |
|---|---|---|---|
| Full OPD | `pure_opd` | `full` | No OPD budget loss mask; all valid response tokens are direct OPD targets. |
| TA-OPD | `teachability` | `dlearn_high` | Select top tokens by `Dlearn = D_norm * C_norm`. |
| Entropy-only | `entropy` | `entropy` | Select top tokens by normalized student entropy. |
| Entropy + TA | `teachability_entropy` | `ca_softor` | Select by soft-OR of entropy and teachability: `H + Dlearn - H * Dlearn`. |
| Split Entropy + TA | `teachability_entropy_split` | `split_budget_ca` | Allocate `OPD_BUDGET_GAMMA` of the token budget to entropy and the rest to teachability. |
| TIP-style | `tip` | `tip` | Select by soft-OR of entropy and raw divergence. |
| Q3 high-C | `q3_highc` | `q3_highc` | Restrict to TIP-Q3 and rank by compatibility/shared support. |
| High-D | `divergence` | `divergence` | Select by raw normalized teacher-student divergence. |
| Random | `random` | `random` | Same-budget random control. |

## H800 Example

```bash
cd /path/to/slime-main

export TAG=main_methods_k16_ratio10_seed1_20260515
export SEED_LABEL=seed1
export METHOD_LIST="pure_opd:1.0:10 teachability:0.10:20 entropy:0.10:30 teachability_entropy:0.10:40 tip:0.10:50"

# Choose GPUs/resources for the target node before launching.
export TEACHER_GPU=0
export RAY_GPUS=1,2,3
export EVAL_GPUS=0,1
export ACTOR_NUM_GPUS_PER_NODE=2
export ROLLOUT_NUM_GPUS=1
export OPD_TOPK_METRICS_K=16

bash ./run_teachability_opd_method_suite_20260515.sh
```

For a smaller fast check:

```bash
export TAG=main_methods_k16_ratio10_smoke_seed1_20260515
export METHOD_LIST="pure_opd:1.0:10 teachability:0.10:20 entropy:0.10:30 teachability_entropy:0.10:40"
export NUM_ROLLOUT=50
bash ./run_teachability_opd_method_suite_20260515.sh
```

## Paper Guidance

The Method section should present teachability as the primary score and entropy as an explicit comparison or optional mixture:

- Primary: `Dlearn = D_norm * C_norm`.
- Entropy-only baseline: tests whether uncertainty alone explains the gains.
- Entropy+Teachability: tests whether the TIP uncertainty axis is complementary to teachable disagreement.
- Full OPD: establishes the all-token upper/reference point.

Do not claim proportional wall-clock savings from the current method. The correct claim is that a small selected set of direct supervision targets can preserve most full-OPD performance.

