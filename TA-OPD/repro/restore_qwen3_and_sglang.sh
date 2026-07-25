#!/usr/bin/env bash
# 还原 Qwen3 模型 config.json + 撤销 sglang qwen2.py monkey-patch
# 幂等: 多次跑都安全。
#
# 用法: bash repro/restore_qwen3_and_sglang.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env 2>/dev/null || true  # 只为拿到 conda env 路径; 失败也无妨

echo "========================================="
echo " 还原 Qwen3 模型 + sglang qwen2.py"
echo "========================================="

# ── 1) 还原 Qwen3 dense 模型 config.json ────────────────────────
echo ""
echo "=== [1/2] 还原 Qwen3 模型 config.json ==="
for M in "${TEACHER_MODEL}" "${STUDENT_HF}"; do
  echo "--- ${M} ---"
  if [[ ! -d "$M" ]]; then
    echo "  ⏭  目录不存在, 跳过"
    continue
  fi
  if [[ -f "${M}/config.json.orig" ]]; then
    mv "${M}/config.json.orig" "${M}/config.json"
    echo "  ✅ config.json 已从 .orig 还原"
  else
    echo "  ⏭  config.json.orig 不存在 (可能从未 patch, 或已还原)"
  fi
  if [[ -f "${M}/config.json" ]]; then
    python3 -c "
import json
c = json.load(open('${M}/config.json'))
print(f'    model_type   = {c.get(\"model_type\")}')
print(f'    architectures = {c.get(\"architectures\")}')
"
  fi
done

# ── 2) 撤销 sglang qwen2.py monkey-patch ────────────────────────
echo ""
echo "=== [2/2] 撤销 sglang qwen2.py monkey-patch ==="
QWEN2_PY="$(python3 -c "import sglang.srt.models.qwen2 as m, os; print(os.path.abspath(m.__file__))" 2>/dev/null || true)"
if [[ -z "$QWEN2_PY" ]]; then
  # 退而求其次, 拼路径
  QWEN2_PY="${CONDA_PREFIX:-/opt/conda/envs/ta_opd}/lib/python3.10/site-packages/sglang/srt/models/qwen2.py"
fi
PATCH_DIR="$(dirname "${QWEN2_PY}")"
HELPER_PY="${PATCH_DIR}/_taopd_qwen3_patch.py"

echo "qwen2.py: ${QWEN2_PY}"
echo "helper:   ${HELPER_PY}"

if [[ -f "${QWEN2_PY}.orig" ]]; then
  mv "${QWEN2_PY}.orig" "${QWEN2_PY}"
  echo "  ✅ qwen2.py 已从 .orig 还原"
elif [[ -f "${QWEN2_PY}" ]] && grep -q "TAOPD_PATCH_V1" "${QWEN2_PY}" 2>/dev/null; then
  echo "  ⚠ qwen2.py 包含 TAOPD_PATCH_V1 但没有 .orig 备份, 无法自动还原"
  echo "    请手动重装 sglang: pip install --force-reinstall sglang==0.4.1"
else
  echo "  ⏭ qwen2.py 未被 patch (或已还原)"
fi

if [[ -f "${HELPER_PY}" ]]; then
  rm "${HELPER_PY}"
  echo "  ✅ helper 已删除: ${HELPER_PY}"
else
  echo "  ⏭ helper 不存在: ${HELPER_PY}"
fi

# 验证
echo ""
echo "=== 验证 ==="
if [[ -f "${QWEN2_PY}" ]]; then
  if python3 -c "import ast; ast.parse(open('${QWEN2_PY}').read())" 2>/dev/null; then
    echo "  ✅ qwen2.py ast.parse OK"
  else
    echo "  ❌ qwen2.py 仍有语法错误 (建议 pip install --force-reinstall sglang==0.4.1)"
  fi
  if python3 -c "import sglang.srt.models.qwen2" 2>/dev/null; then
    echo "  ✅ sglang.srt.models.qwen2 import OK"
  else
    echo "  ⚠ sglang.srt.models.qwen2 import 失败 (建议 pip install --force-reinstall sglang==0.4.1)"
  fi
fi

echo ""
echo "========================================="
echo " 还原完成"
echo ""
echo " ⚠️  注意: sglang 0.4.1 不认识 Qwen3ForCausalLM,"
echo "    还原后 teacher server 会再次报:"
echo "    ValueError: Model architectures ['Qwen3ForCausalLM'] are not supported"
echo "    需要另想办法 (升级 sglang / 换模型 / ...) 才能跑 smoke test。"
echo "========================================="
