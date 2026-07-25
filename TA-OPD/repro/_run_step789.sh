#!/usr/bin/env bash
# 接在 _run_step6_rest.sh 的 sgl-router 失败点, 跑 step 7 尾 + step 8-9
set -eo pipefail
LOG="${HOME}/taopd-faithful-logs/step789.log"
mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1
echo "=== $(date) ==="

export MAMBA_EXE="${HOME}/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
eval "$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}")"
micromamba activate ta_opd_faithful

NVIDIA_PKG="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia"
export CUDA_HOME="${CONDA_PREFIX}"

echo "--- sgl-router (修正: sglang-router 0.3.2 @ PyPI) ---"
pip install sglang-router==0.3.2 2>&1 | tail -5

echo ""
echo "=== [8/9] Megatron-LM @ 3714d81d418c9f1bca4594fc35f9e8289f652862 ==="
BASE_DIR="${HOME}/taopd-faithful"
mkdir -p "${BASE_DIR}"
MEGATRON_DIR="${BASE_DIR}/Megatron-LM"
MEGATRON_LOCAL="${HOME}/taopd-baseline/Megatron-LM"
if [[ ! -d "${MEGATRON_DIR}" ]]; then
  if [[ -d "${MEGATRON_LOCAL}" ]]; then
    echo "从本地 Megatron-LM clone (--dissociate, 干净副本): ${MEGATRON_LOCAL}"
    git clone --dissociate "${MEGATRON_LOCAL}" "${MEGATRON_DIR}"
  else
    git clone https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_DIR}"
  fi
fi
cd "${MEGATRON_DIR}"
# 尝试 fetch; 如果 github 不通, 直接用本地已有的 commit
git fetch --all --quiet 2>/dev/null || echo "⚠ git fetch 失败 (github 不通), 假设目标 commit 已在本地"
if git rev-parse -q --verify 3714d81d418c9f1bca4594fc35f9e8289f652862 >/dev/null; then
  git checkout 3714d81d418c9f1bca4594fc35f9e8289f652862
else
  echo "❌ 本地没有目标 commit, 需要 github 连接"
  exit 128
fi
# Megatron setup.py 调 `python -m pybind11 --includes`
pip install pybind11 2>&1 | tail -3
pip install -e . --no-build-isolation 2>&1 | tail -5

echo ""
echo "=== [9/9] slime + cudnn pin + patches ==="
# slime: 直接用 repo 自带的 slime_ta_opd/ (即 slime fork, 带 v0.5.9 patches)
REPO_ROOT="${HOME}/taopd-baseline/TA-OPD"
export SLIME_DIR="${REPO_ROOT}/slime_ta_opd"
if [[ ! -d "${SLIME_DIR}/slime" ]]; then
  echo "❌ ${SLIME_DIR}/slime 不存在"
  exit 1
fi
cd "${SLIME_DIR}"
pip install -e . 2>&1 | tail -5

pip install "nvidia-cudnn-cu12==9.16.0.29" 2>&1 | tail -3
# 官方 build_conda.sh 写的是 numpy<2, 但 transformers 4.57.1 + scipy 1.18 都要求 numpy>=2.0.
# numpy 2.0.2 与 torch 2.9.1+cu129 兼容 (>=2.0.0).
pip install "numpy==2.0.2" 2>&1 | tail -3

echo "-- apply sglang.patch --"
cd "${BASE_DIR}/sglang"
if git apply --check "${SLIME_DIR}/docker/patch/v0.5.9/sglang.patch" 2>&1; then
  git apply "${SLIME_DIR}/docker/patch/v0.5.9/sglang.patch" && echo "sglang.patch applied"
else
  echo "sglang.patch already applied or conflict (skip)"
fi

echo "-- apply megatron.patch --"
cd "${MEGATRON_DIR}"
if git apply --check "${SLIME_DIR}/docker/patch/v0.5.9/megatron.patch" 2>&1; then
  git apply "${SLIME_DIR}/docker/patch/v0.5.9/megatron.patch" && echo "megatron.patch applied"
else
  echo "megatron.patch already applied or conflict (skip)"
fi

echo ""
echo "=== ALL DONE ==="
python -c "
import transformer_engine as te, sglang, megatron, apex
print('TE', te.__version__)
print('sglang', sglang.__version__)
print('megatron:', megatron.__file__)
print('apex:', apex.__file__)
import sglang_router
print('sglang_router:', sglang_router.__file__)
" 2>&1 | tail -10

echo "=== $(date) exit=$? ==="
