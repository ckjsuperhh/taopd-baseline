#!/usr/bin/env bash
# 单独装/编译 sglang，看失败原因
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

export CUDA_HOME="${CUDA_HOME:-$(python3 -c 'import sys; print(sys.prefix)')}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

echo "=== 环境 ==="
echo "CUDA_HOME   = ${CUDA_HOME}"
which nvcc && nvcc --version 2>/dev/null | tail -1
which gcc  && gcc  --version 2>/dev/null | head -1
python3 -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')"
echo ""

# 先看现有 sglang 状态
echo "=== 现有 sglang 状态 ==="
pip show sglang 2>/dev/null | grep -E '^(Name|Version|Location):' || echo "(未装)"
pip show sgl-kernel 2>/dev/null | grep -E '^(Name|Version|Location):' || echo "sgl-kernel: (未装)"
python3 -c "
try:
    import sglang
    print(f'import sglang: OK ({sglang.__file__})')
except Exception as e:
    print(f'import sglang: FAIL ({e})')
try:
    import sgl_kernel
    print(f'import sgl_kernel: OK ({sgl_kernel.__file__})')
except Exception as e:
    print(f'import sgl_kernel: FAIL ({e})')
"
echo ""

read -rp "要重装 sglang 吗? [y/N] " ans
case "$ans" in
  y|Y) ;;
  *) echo "跳过安装，只做诊断。"; exit 0 ;;
esac

echo "=== 卸载旧版 ==="
pip uninstall -y sglang sgl-kernel sglang-router 2>/dev/null || true

echo "=== 安装 sglang==0.4.1 + sgl-kernel==0.1.0 (详细日志) ==="
FLASHINFER_INDEX="https://flashinfer.ai/whl/cu124/torch2.5/"
pip install --verbose "sglang==0.4.1" 2>&1 | tee /tmp/sglang_install.log
pip install --verbose "sgl-kernel==0.1.0" --extra-index-url "${FLASHINFER_INDEX}" 2>&1 | tee -a /tmp/sglang_install.log

echo ""
echo "=== 安装后状态 ==="
pip show sglang 2>/dev/null | grep -E '^(Name|Version):'
pip show sgl-kernel 2>/dev/null | grep -E '^(Name|Version):' || echo "sgl-kernel: (未装)"

python3 -c "
import sglang
print(f'import sglang OK: {sglang.__file__}')
try:
    import sgl_kernel
    print(f'import sgl_kernel OK: {sgl_kernel.__file__}')
except Exception as e:
    print(f'import sgl_kernel FAIL: {e}')
"

echo ""
echo "日志: /tmp/sglang_install.log"
echo "查失败原因: grep -iE 'error|failed|exception' /tmp/sglang_install.log | tail -30"
