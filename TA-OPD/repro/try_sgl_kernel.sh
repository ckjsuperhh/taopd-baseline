#!/usr/bin/env bash
# 只尝试装 sgl-kernel (sglang 已经装好，只缺 CUDA kernel)
# 用法: bash repro/try_sgl_kernel.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

export CUDA_HOME="${CUDA_HOME:-$(python3 -c 'import sys; print(sys.prefix)')}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

FLASHINFER_INDEX="https://flashinfer.ai/whl/cu124/torch2.5/"

echo "=== 环境 ==="
echo "CUDA_HOME = ${CUDA_HOME}"
which nvcc && nvcc --version 2>/dev/null | tail -1
python3 -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')"
echo ""

echo "=== 现状 ==="
pip show sglang 2>/dev/null | grep -E '^(Name|Version):' || echo "sglang: (未装)"
pip show sgl-kernel 2>/dev/null | grep -E '^(Name|Version):' || echo "sgl-kernel: (未装)"
python3 -c "
try:
    import sgl_kernel
    print(f'sgl_kernel import OK: {sgl_kernel.__file__}')
except Exception as e:
    print(f'sgl_kernel import FAIL: {e}')
"
echo ""

echo "=== 安装 sgl-kernel==0.1.0 (via flashinfer.ai) ==="
pip install "sgl-kernel==0.1.0" --extra-index-url "${FLASHINFER_INDEX}" \
  || pip install "sgl-kernel==0.1.0" \
  || { echo "❌ sgl-kernel==0.1.0 装不上"; exit 1; }
echo "✅ sgl-kernel==0.1.0 装好"

echo ""
echo "=== 最终状态 ==="
pip show sglang 2>/dev/null | grep -E '^(Name|Version):'
pip show sgl-kernel 2>/dev/null | grep -E '^(Name|Version):' || echo "sgl-kernel: (未装)"
python3 -c "
import sglang
print(f'sglang import OK')
try:
    import sgl_kernel
    print(f'sgl_kernel import OK: {sgl_kernel.__file__}')
except Exception as e:
    print(f'sgl_kernel import FAIL: {e}')
"
