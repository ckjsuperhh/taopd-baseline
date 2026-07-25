#!/usr/bin/env bash
# 接在 _run_te_only.sh 之后, 续跑 step 6b(apex) + step 7-9
# fla 已装好, TE 已装好
set -eo pipefail
LOG="${HOME}/taopd-faithful-logs/step6_rest.log"
mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1
echo "=== $(date) ==="

export MAMBA_EXE="${HOME}/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
eval "$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}")"
micromamba activate ta_opd_faithful

NVIDIA_PKG="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia"
export C_INCLUDE_PATH="${NVIDIA_PKG}/cusparse/include:${NVIDIA_PKG}/cublas/include:${NVIDIA_PKG}/cufft/include:${NVIDIA_PKG}/cusolver/include:${NVIDIA_PKG}/curand/include:${NVIDIA_PKG}/cudnn/include:${NVIDIA_PKG}/cuda_runtime/include:${NVIDIA_PKG}/nvtx/include:${NVIDIA_PKG}/cuda_nvrtc/include:${NVIDIA_PKG}/nccl/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${C_INCLUDE_PATH}"
export LIBRARY_PATH="${NVIDIA_PKG}/cusparse/lib:${NVIDIA_PKG}/cublas/lib:${NVIDIA_PKG}/cufft/lib:${NVIDIA_PKG}/cusolver/lib:${NVIDIA_PKG}/curand/lib:${NVIDIA_PKG}/cudnn/lib:${NVIDIA_PKG}/cuda_runtime/lib:${NVIDIA_PKG}/nvtx/lib:${NVIDIA_PKG}/cuda_nvrtc/lib:${NVIDIA_PKG}/nccl/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CONDA_PREFIX}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "nvcc at $(which nvcc)"

echo ""
echo "--- [6b] apex (预计 20-30min) ---"
NVCC_APPEND_FLAGS="--threads 4" \
  pip install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4 2>&1 | tail -15

echo ""
echo "=== [7/9] torch_memory_saver / Megatron-Bridge / modelopt / sgl-router ==="
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall 2>&1 | tail -3
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation 2>&1 | tail -3
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation 2>&1 | tail -3
pip install sgl-router==0.3.2 2>&1 | tail -3

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
echo "slime installed at ${SLIME_DIR}"

pip install "nvidia-cudnn-cu12==9.16.0.29" 2>&1 | tail -3
pip install "numpy<2" 2>&1 | tail -3

echo "-- apply sglang.patch to ${BASE_DIR}/sglang --"
cd "${BASE_DIR}/sglang"
if git apply --check "${SLIME_DIR}/docker/patch/v0.5.9/sglang.patch" 2>&1; then
  git apply "${SLIME_DIR}/docker/patch/v0.5.9/sglang.patch" && echo "sglang.patch applied"
else
  echo "sglang.patch already applied or conflict (skip)"
fi

echo "-- apply megatron.patch to ${MEGATRON_DIR} --"
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
print('megatron OK:', megatron.__file__)
print('apex OK:', apex.__file__)
" 2>&1 | tail -10

echo "=== $(date) exit=$? ==="
