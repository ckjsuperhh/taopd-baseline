#!/usr/bin/env bash
# 一键诊断 + 修复 step 1/2 残留问题
# 用法: bash repro/fix_step1_step2.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

export CUDA_HOME="${CUDA_HOME:-$(python3 -c 'import sys; print(sys.prefix)')}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${MEGATRON_LM_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"

# 辅助: 安全地只显示前 N 行 (避免 pipefail + head 导致 SIGPIPE 退出)
head_safe() {
  python3 -c "
import sys
n = int(sys.argv[1])
for i, line in enumerate(sys.stdin):
    if i >= n: break
    sys.stdout.write(line)
" "$1"
}

echo "========================================="
echo " Paths"
echo "========================================="
echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "MEGATRON_LM_DIR = ${MEGATRON_LM_DIR}"
echo "SLIME_DIR = ${SLIME_DIR}"
echo "MODEL_DIR = ${MODEL_DIR}"
echo ""

# ── 1. 诊断 megatron 导入 ───────────────────────────────────────────────
echo "========================================="
echo " [1] Megatron-LM 诊断"
echo "========================================="
echo "--- ls ${MEGATRON_LM_DIR} (前 15 行) ---"
ls "${MEGATRON_LM_DIR}" 2>&1 | head_safe 15
echo ""
echo "--- ls ${MEGATRON_LM_DIR}/megatron ---"
ls "${MEGATRON_LM_DIR}/megatron" 2>&1
echo ""
echo "--- ls ${MEGATRON_LM_DIR}/megatron/__init__.py? ---"
ls -la "${MEGATRON_LM_DIR}/megatron/__init__.py" 2>&1 || true
echo ""

echo "--- python import megatron ---"
python3 -c "
import sys
print('sys.path 前 3 项:')
for p in sys.path[:3]: print(' ', p)
try:
    import megatron
    print(f'import megatron OK, __file__ = {megatron.__file__!r}')
    print(f'      __path__ = {list(getattr(megatron, \"__path__\", []))}')
except Exception as e:
    print(f'import megatron FAIL: {type(e).__name__}: {e}')
"
echo ""

# 如果 __init__.py 缺失 (namespace package, __file__=None), 创建一个空的
if [[ -d "${MEGATRON_LM_DIR}/megatron" ]] && [[ ! -f "${MEGATRON_LM_DIR}/megatron/__init__.py" ]]; then
  echo "⚠ ${MEGATRON_LM_DIR}/megatron/__init__.py 缺失"
  echo "  新版 Megatron-LM 用 implicit namespace package，导致 megatron.__file__=None"
  echo "  创建空 __init__.py 修复..."
  touch "${MEGATRON_LM_DIR}/megatron/__init__.py"
  # 子包也补一下
  for sub in core legacy training rl post_training; do
    if [[ -d "${MEGATRON_LM_DIR}/megatron/${sub}" ]] && [[ ! -f "${MEGATRON_LM_DIR}/megatron/${sub}/__init__.py" ]]; then
      touch "${MEGATRON_LM_DIR}/megatron/${sub}/__init__.py"
      echo "  + megatron/${sub}/__init__.py"
    fi
  done
fi

echo "--- 再次 import megatron ---"
python3 -c "
import megatron
print(f'__file__ = {megatron.__file__!r}')
try:
    from megatron.core import parallel_state
    print('megatron.core.parallel_state import OK')
except Exception as e:
    print(f'megatron.core import issue: {e}')
"
echo ""

# ── 2. 诊断 sgl_kernel 导入 ─────────────────────────────────────────────
echo "========================================="
echo " [2] sgl_kernel 诊断"
echo "========================================="
pip show sgl-kernel 2>/dev/null | grep -E '^(Name|Version|Location):' || echo "sgl-kernel: (未装)"
echo ""
echo "--- python import sgl_kernel ---"
SGL_ERR="$(python3 -c "
try:
    import sgl_kernel
    print(f'OK: {sgl_kernel.__file__}')
except Exception as e:
    print(f'FAIL: {type(e).__name__}: {e}')
" 2>&1)" || true
echo "${SGL_ERR}"
echo ""

# 如果 sgl_kernel import 失败, 看是不是 GLIBCXX / ABI 问题
if echo "${SGL_ERR}" | grep -qE 'GLIBCXX|undefined symbol'; then
  echo "⚠ sgl_kernel 报 GLIBCXX/symbol 错误"
  echo ""
  echo "--- 当前 LD_LIBRARY_PATH ---"
  echo "${LD_LIBRARY_PATH}" | tr ':' '\n'
  echo ""
  echo "--- conda 里的 libstdc++.so.6 ---"
  ls -la "${CUDA_HOME}/lib/libstdc++.so.6"* 2>/dev/null || true
  echo ""
  echo "--- 系统的 libstdc++.so.6 支持的 GLIBCXX 最高版本 ---"
  strings /lib/x86_64-linux-gnu/libstdc++.so.6 2>/dev/null | grep ^GLIBCXX_ | sort -V | tail -3 || true
  echo "--- conda 的 ---"
  strings "${CUDA_HOME}/lib/libstdc++.so.6" 2>/dev/null | grep ^GLIBCXX_ | sort -V | tail -3 || true
fi

echo ""

# ── 3. 验证模型已下载 ──────────────────────────────────────────────────
echo "========================================="
echo " [3] 模型权重检查"
echo "========================================="
echo "Teacher: ${TEACHER_MODEL}"
ls "${TEACHER_MODEL}/config.json" 2>/dev/null && echo "  ✅ config.json 存在" || echo "  ❌ 不存在"
echo "  safetensors 文件数: $(ls -1 "${TEACHER_MODEL}"/*.safetensors 2>/dev/null | wc -l)"
echo "Student: ${STUDENT_HF}"
ls "${STUDENT_HF}/config.json" 2>/dev/null && echo "  ✅ config.json 存在" || echo "  ❌ 不存在"
echo "  safetensors 文件数: $(ls -1 "${STUDENT_HF}"/*.safetensors 2>/dev/null | wc -l)"
echo ""

# ── 4. 总体验证 ────────────────────────────────────────────────────────
echo "========================================="
echo " [4] 总体验证"
echo "========================================="
python3 -c "
import torch
v = torch.__version__
ok = '✅' if v.startswith('2.5.') else '❌'
print(f'  {ok} torch {v} cuda={torch.version.cuda} GPUs={torch.cuda.device_count()}')
try:
    import flash_attn; print(f'  ✅ flash_attn {flash_attn.__version__}')
except Exception as e: print(f'  ❌ flash_attn ({e})')
try:
    import sglang
    import subprocess as _sp
    _r = _sp.run(['pip', 'show', 'sglang'], capture_output=True, text=True)
    _sv = 'unknown'
    for _line in (_r.stdout or '').splitlines():
        if _line.startswith('Version:'):
            _sv = _line.split(':', 1)[1].strip(); break
    _has_kernel = False
    try:
        import sgl_kernel; _has_kernel = True
    except Exception:
        pass
    _mark = chr(9989) if _has_kernel else chr(9888)
    _kernel_msg = 'ok' if _has_kernel else 'MISSING'
    print(f'  {_mark} sglang {_sv} (sgl_kernel={_kernel_msg})')
except Exception as e:
    print(f'  ❌ sglang ({e})')
try:
    import ray; print(f'  ✅ ray {ray.__version__}')
except Exception as e: print(f'  ❌ ray ({e})')
try:
    import transformers; print(f'  ✅ transformers {transformers.__version__}')
except Exception as e: print(f'  ❌ transformers ({e})')
try:
    import pyarrow; print(f'  ✅ pyarrow {pyarrow.__version__}')
except Exception as e: print(f'  ❌ pyarrow ({e})')
try:
    import megatron
    mp = megatron.__file__ or '(namespace package, __file__=None)'
    print(f'  ✅ megatron ({mp})')
except Exception as e:
    print(f'  ❌ megatron ({e})')
try:
    from megatron.core import parallel_state
    print(f'  ✅ megatron.core.parallel_state')
except Exception as e:
    print(f'  ❌ megatron.core.parallel_state ({e})')
"
echo ""
echo "========================================="
echo " Done. 如果 [4] 全部 ✅ → bash repro/run_all.sh 2"
echo "========================================="
