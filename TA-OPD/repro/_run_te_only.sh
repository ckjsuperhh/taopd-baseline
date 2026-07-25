#!/usr/bin/env bash
# 只跑 transformer_engine 安装, 完整输出到 log, 方便查错
set -eo pipefail
LOG="${HOME}/taopd-faithful-logs/te_only.log"
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

# Force source build: skip the GitHub Releases wheel download which is blocked in China.
# (TE setup.py checks NVTE_PYTORCH_FORCE_BUILD=="TRUE" to bypass CachedWheelsCommand.)
export NVTE_PYTORCH_FORCE_BUILD=TRUE

echo "CUDA_HOME=${CUDA_HOME}"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "C_INCLUDE_PATH=${C_INCLUDE_PATH}"
which nvcc
nvcc --version | tail -2

echo ""
echo "=== precheck cusparse.h ==="
echo '#include <cusparse.h>' > /tmp/test_cu.c
echo 'int main(){return 0;}' >> /tmp/test_cu.c
${CONDA_PREFIX}/bin/nvcc -c /tmp/test_cu.c -o /tmp/test_cu.o && echo "OK: cusparse.h found" || echo "FAIL: cusparse.h not found"
rm -f /tmp/test_cu.c /tmp/test_cu.o

echo ""
echo "=== pip install transformer_engine[pytorch]==2.10.0 ==="
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
echo "=== DONE exit=$? ==="
