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
if [[ ! -d "${MEGATRON_DIR}" ]]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_DIR}"
fi
cd "${MEGATRON_DIR}"
git fetch --all --quiet
git checkout 3714d81d418c9f1bca4594fc35f9e8289f652862
pip install -e . --no-build-isolation 2>&1 | tail -5

echo ""
echo "=== [9/9] slime + cudnn pin + patches ==="
if [[ ! -d "${BASE_DIR}/slime" ]]; then
  cd "${BASE_DIR}"
  git clone https://github.com/THUDM/slime.git
fi
export SLIME_DIR="${BASE_DIR}/slime"
cd "${SLIME_DIR}"
pip install -e . 2>&1 | tail -5

pip install "nvidia-cudnn-cu12==9.16.0.29" 2>&1 | tail -3
pip install "numpy<2" 2>&1 | tail -3

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
