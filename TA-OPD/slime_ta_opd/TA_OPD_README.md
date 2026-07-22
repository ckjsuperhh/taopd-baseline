# TA-OPD full slime tree

This directory contains the anonymized slime checkout used for the TA-OPD experiments. It is included so users can run the released scripts without manually reconstructing the exact slime-side changes.

The primary TA-OPD implementation entry points are:

- `slime/utils/arguments.py`
- `slime/ray/rollout.py`
- `slime/rollout/sglang_rollout.py`
- `slime/rollout/on_policy_distillation.py`
- `slime/rollout/tip_compat.py`

The top-level training entry point remains `train.py`, following the original slime workflow. Cluster-specific paths have been replaced by `/path/to/...` placeholders. Set the corresponding environment variables before launching training or diagnostics.
