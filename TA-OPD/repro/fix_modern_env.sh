#!/usr/bin/env bash
# 修复 modern env 三个问题:
#   1) sgl_kernel libnuma.so.1 缺失
#   2) flash_attn 2.x 和 torch 2.9 ABI 不兼容 → 升级到 4.0.0b23
#   3) sgl_kernel wheel 只含 sm_100, 没 sm_89 (RTX 4090 需要的)
#
# 用法: bash repro/fix_modern_env.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

MODERN_ENV="${MODERN_ENV:-ta_opd_modern}"
eval "$(conda shell.bash hook)"
conda activate "${MODERN_ENV}"
echo "✅ activated: ${MODERN_ENV}"
echo ""

# ── [1/4] libnuma ─────────────────────────────────────────────
echo "=== [1/4] 装 libnuma (sgl_kernel 的 CUDA runtime 需要) ==="
# 先试 conda (用户态)
if ! python3 -c "import ctypes; ctypes.CDLL('libnuma.so.1')" 2>/dev/null; then
  conda install -n "${MODERN_ENV}" -c conda-forge -y libnuma 2>&1 | tail -3 \
    || echo "  ⚠ conda libnuma 失败"
fi
# 验证
if python3 -c "import ctypes; ctypes.CDLL('libnuma.so.1'); print('  ✅ libnuma.so.1 OK')" 2>&1; then
  :
else
  echo "  ⚠ conda 装的 libnuma 找不到, 试系统包"
  echo "    请 root 跑: apt install -y libnuma-dev"
  echo "    或设 LD_LIBRARY_PATH: export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
fi

# ── [2/4] flash-attn 4.0.0b23 (for torch 2.9) ─────────────────
echo ""
echo "=== [2/4] 强制升级到 flash-attn 4.0.0b23 (torch 2.9 ABI) ==="
pip install --force-reinstall --no-deps flash-attn==4.0.0b23 2>&1 | tail -5 \
  || pip install --force-reinstall flash-attn==4.0.0b23 2>&1 | tail -5 \
  || echo "  ⚠ flash-attn 4.0.0b23 装不上"

# 验证
python3 -c "
import flash_attn
print(f'  ✅ flash_attn {flash_attn.__version__}')
" 2>&1 || echo "  ❌ flash_attn import 失败"

# ── [3/4] sgl_kernel sm_89 排查 ──────────────────────────────
echo ""
echo "=== [3/4] sgl_kernel sm_89 排查 ==="
SGL_KERNEL_DIR="$(python3 -c "import os; print(os.path.dirname(__import__('sgl_kernel').__file__))" 2>/dev/null || echo "?")"
echo "sgl_kernel 目录: ${SGL_KERNEL_DIR}"
echo ""
echo "现有 .so 文件:"
find "${SGL_KERNEL_DIR}" -name "*.so" 2>/dev/null | head -30
echo ""
echo "按 arch 分类:"
for D in "${SGL_KERNEL_DIR}/sm"*/; do
  [[ -d "$D" ]] && echo "  $(basename "$D")/: $(ls "$D" | wc -l) files"
done

# 试 import (带详细错误)
echo ""
echo "sgl_kernel import 测试:"
python3 -c "
import sgl_kernel
print(f'  ✅ sgl_kernel {getattr(sgl_kernel, \"__version__\", \"?\")}')
" 2>&1 | head -20 || true

# 如果还是 sm100 only, 提供手动方案
if ! python3 -c "import sgl_kernel" 2>/dev/null; then
  echo ""
  echo "  ❌ sgl_kernel 仍无法加载 sm_89"
  echo ""
  echo "  手动方案 (按优先级):"
  echo "  方案 A: 从 sglang GitHub release 下载 sm89 wheel"
  echo "    pip install 'sglang-kernel' --upgrade --pre \\"
  echo "      --extra-index-url https://sgl-whl.kiud.info/sgl-cu128/"
  echo ""
  echo "  方案 B: 从 sglang 源码编译 (for sm_89)"
  echo "    cd /tmp && git clone https://github.com/sgl-project/sglang.git"
  echo "    cd sglang/sgl-kernel"
  echo "    TORCH_CUDA_ARCH_LIST='8.9' python -m build --wheel"
  echo "    pip install --force-reinstall dist/sgl_kernel-*.whl"
  echo ""
  echo "  方案 C: 设置环境变量允许 sm80 fallback (不一定有效)"
  echo "    export SGL_KERNEL_SM_FALLBACK=80"
fi

# ── [4/4] 完整验证 ────────────────────────────────────────────
echo ""
echo "=== [4/4] 完整验证 ==="
python3 -c "
import sys
checks = [
    ('torch', 'HARD'),
    ('sglang', 'HARD'),
    ('sgl_kernel', 'HARD'),
    ('flashinfer', 'HARD'),
    ('flash_attn', 'HARD'),
    ('ray', 'HARD'),
    ('transformers', 'HARD'),
    ('datasets', 'HARD'),
    ('safetensors', 'HARD'),
    ('megatron', 'HARD'),
    ('mbridge', 'HARD'),
]
n_ok = n_fail = 0
for mod, kind in checks:
    try:
        m = __import__(mod)
        v = getattr(m, '__version__', '?')
        print(f'  ✅ {mod:20s} = {v}')
        n_ok += 1
    except Exception as e:
        print(f'  ❌ {mod:20s}: {str(e).splitlines()[0]}')
        n_fail += 1
print()
print(f'{n_ok} OK / {n_fail} FAIL')
"

echo ""
echo "========================================="
echo " 修复完成"
echo ""
echo " 下一步:"
echo "   bash repro/run_all_modern.sh 3"
echo ""
echo " 如果 sgl_kernel 还是 sm100 only,"
echo " 走方案 B 从源码编译 (最稳)。"
echo "========================================="
