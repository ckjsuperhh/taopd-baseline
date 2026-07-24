#!/usr/bin/env bash
# 降级 sgl-kernel 到匹配 torch 2.5.x 的版本
# 当前 sgl-kernel 0.3.21 是给 torch 2.7+ 编的, c10::ListType::get ABI 不匹配
# 用法: bash repro/fix_sgl_kernel_v2.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

# 安全地显示前 N 行 (避免 pipefail + head SIGPIPE)
head_safe() {
  python3 -c "
import sys
n = int(sys.argv[1])
for i, line in enumerate(sys.stdin):
    if i >= n: break
    sys.stdout.write(line)
" "$1"
}

ENV_PREFIX="$(python3 -c 'import sys; print(sys.prefix)')"
TORCH_LIB="$(python3 -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "=== 环境 ==="
echo "torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "sglang: $(pip show sglang 2>/dev/null | awk '/^Version:/{print $2}')"
echo "sgl-kernel 当前: $(pip show sgl-kernel 2>/dev/null | awk '/^Version:/{print $2}')"
echo ""

echo "=== [1] 卸载当前 sgl-kernel ==="
pip uninstall -y sgl-kernel sgl_kernel 2>/dev/null || true
echo ""

FLASHINFER_INDEX="https://flashinfer.ai/whl/cu124/torch2.5/"

echo "=== [2] 逐版本试装 (sglang 0.4.1 时期的 sgl-kernel) ==="
# 从最可能的匹配版本开始, 逐渐回退
# sglang 0.4.1 大约 2025 年初, sgl-kernel 对应版本约 0.0.x ~ 0.1.x
OK_VER=""
for VER in "0.1.0.post2" "0.1.0.post1" "0.1.0" "0.0.5.post3" "0.0.5.post2" "0.0.5.post1" "0.0.5" "0.0.4.post2" "0.0.4.post1" "0.0.4"; do
  echo "--- 试 sgl-kernel==${VER} ---"
  if pip install "sgl-kernel==${VER}" --extra-index-url "${FLASHINFER_INDEX}" >/tmp/sgl_install.log 2>&1; then
    # 装成功, 测 import
    if python3 -c "import sgl_kernel" 2>/tmp/sgl_import.err; then
      echo "  ✅ sgl-kernel==${VER} 装好且 import OK"
      OK_VER="${VER}"
      break
    else
      echo "  ⚠ sgl-kernel==${VER} 装好但 import 失败:"
      cat /tmp/sgl_import.err | head_safe 3
      pip uninstall -y sgl-kernel >/dev/null 2>&1 || true
    fi
  else
    echo "  ✗ 装不上 (可能 PyPI/flashinfer.ai 上没这个版本)"
  fi
done

if [[ -z "${OK_VER}" ]]; then
  echo ""
  echo "❌ 所有已知版本都失败。列出 PyPI 上所有可用版本:"
  pip index versions sgl-kernel 2>/dev/null | head -3 || pip install "sgl-kernel==99.99.99" 2>&1 | grep -iE 'from versions'
  echo ""
  echo "请贴出可用版本列表, 我帮你选。"
  exit 1
fi

echo ""
echo "=== [3] 最终验证 ==="
python3 -c "
import torch
import sgl_kernel
import importlib.util, pathlib
spec = importlib.util.find_spec('sgl_kernel')
print(f'torch       : {torch.__version__}')
print(f'sgl_kernel  : (pip show below)')
print(f'sgl_kernel path: {spec.origin if spec else None}')
print(f'✅ 全部 OK')
"
pip show sgl-kernel | grep -E '^(Name|Version):'

echo ""
echo "========================================="
echo " ✅ sgl-kernel==${OK_VER} 修复完成。"
echo " 继续: bash repro/run_all.sh 2"
echo "========================================="
