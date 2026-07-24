#!/usr/bin/env bash
# 诊断 + 修复 mbridge 缺失
# step 3 报错: from mbridge.core import register_model → ModuleNotFoundError
# 用法: bash repro/fix_mbridge.sh
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

echo "=== 环境 ==="
echo "SLIME_DIR = ${SLIME_DIR}"
echo ""

echo "=== [1] 看 slime_plugins/mbridge 里都依赖什么 ==="
MBRIDGE_DIR="${SLIME_DIR}/slime_plugins/mbridge"
if [[ -d "${MBRIDGE_DIR}" ]]; then
  echo "--- ls ${MBRIDGE_DIR} ---"
  ls "${MBRIDGE_DIR}"
  echo ""
  echo "--- grep import ${MBRIDGE_DIR}/*.py ---"
  grep -E "^(import|from) " "${MBRIDGE_DIR}"/*.py 2>/dev/null || true
else
  echo "⚠ ${MBRIDGE_DIR} 不存在"
fi
echo ""

echo "=== [2] 看 slime_plugins 其它可能缺的包 ==="
echo "--- grep '^from ' ${SLIME_DIR}/slime_plugins/ -r ---"
grep -hRE "^from [a-zA-Z_][a-zA-Z0-9_]*" "${SLIME_DIR}/slime_plugins/" 2>/dev/null | sort -u | head_safe 30 || true
echo ""

echo "=== [3] pip 上有没有 mbridge ==="
pip index versions mbridge 2>/dev/null | head_safe 3 || true
pip show mbridge 2>/dev/null | grep -E '^(Name|Version|Location):' || echo "mbridge: (未装)"
echo ""

echo "=== [4] 试装 mbridge ==="
# 常见的 mbridge 来源:
#   - PyPI 上的 mbridge (如果存在)
#   - git+https://github.com/...
#   - 项目内 vendor
if pip install mbridge 2>&1 | tee /tmp/mbridge_install.log; then
  echo "  ✅ mbridge 装好 (via PyPI)"
elif pip install m-bridge 2>&1 | tee -a /tmp/mbridge_install.log; then
  echo "  ✅ m-bridge 装好"
else
  echo "  ⚠ PyPI 上没有 mbridge, 看项目里有没有自带:"
  # 看 slime_ta_opd 仓库是否自带 mbridge 子目录或 setup
  find "${SLIME_DIR}" -maxdepth 3 -type d -name "mbridge*" 2>/dev/null
  find "${SLIME_DIR}" -maxdepth 3 -name "setup.py" -o -name "pyproject.toml" 2>/dev/null | head_safe 5 || true
  # 看 slime_ta_opd 的 requirements
  echo ""
  echo "--- ${SLIME_DIR} 下的 requirements*.txt / pyproject.toml ---"
  for f in "${SLIME_DIR}"/requirements*.txt "${SLIME_DIR}"/pyproject.toml "${SLIME_DIR}"/setup.py; do
    if [[ -f "$f" ]]; then
      echo "[$f]"
      cat "$f" | head_safe 30
      echo ""
    fi
  done
fi
echo ""

echo "=== [5] 验证 mbridge import ==="
python3 -c "
try:
    from mbridge.core import register_model
    print('✅ from mbridge.core import register_model OK')
except Exception as e:
    print(f'❌ {type(e).__name__}: {e}')
"
echo ""

echo "=== [6] 重跑 step 3 的关键 import ==="
export PYTHONPATH="${MEGATRON_LM_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
python3 -c "
import sys
sys.path.insert(0, '${SLIME_DIR}')
try:
    import slime_plugins.mbridge
    print('✅ import slime_plugins.mbridge OK')
except Exception as e:
    print(f'❌ {type(e).__name__}: {e}')
"
echo ""

echo "========================================="
echo " 如果 [5] 和 [6] 都 ✅ → bash repro/run_all.sh 3"
echo " 如果 mbridge 装不上 → 把上面所有输出贴给 QoderCN"
echo "========================================="
