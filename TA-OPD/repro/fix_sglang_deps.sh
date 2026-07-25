#!/usr/bin/env bash
# 完整修复 sglang 0.4.1 server 启动依赖
# 用法: bash repro/fix_sglang_deps.sh
#
# 安装顺序 (严格):
#   1) torch 2.5.1 先立住 (flashinfer 的 wheel 需要 torch 2.5.* 才能解析)
#   2) 常用 server 运行时依赖 (fastapi / orjson / pyzmq / outlines+interegular / ...)
#   3) flashinfer-python (从 flashinfer.ai 官方索引, cu124 + torch2.5)
#   4) vllm (sglang 0.4.1 硬要 >=0.6.3.post1, 但 vllm 会拉低 torch 到 2.4)
#   5) 强制 torch 回退到 2.5.1+cu124 (覆盖 vllm 拉的 2.4)
#   6) 验证 import + 提示 server 用 --attention-backend triton 避开 flashinfer 兼容问题
set -eo pipefail  # 不能用 -u：conda 内部 activate 脚本有 unbound variable (SYS_SYSROOT 等)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

# ─────────────────────────────────────────────────────────────────
echo "=== [1/6] 先立 torch 2.5.1+cu124 ==="
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall --no-deps || echo "  ⚠ torch rollback 失败"

echo ""
echo "=== [2/6] 装 sglang server 运行时依赖 ==="

# ── pyairports (outlines 传递依赖, 关键 HARD, 单独多策略处理) ────
# 这包比较怪: PyPI 上 0.0.1 是 2025-09 才上线, 经常被 resolver 跳过。
# 策略链: PyPI → GitHub 源码 → 替代包 airports-py → 重装 outlines 拉传递依赖
PYAIRPORTS_OK=0

echo "--- [pyairports] 策略 1/4: PyPI ---"
if pip install pyairports 2>&1 | tail -3; then
  python3 -c "import pyairports" 2>/dev/null && PYAIRPORTS_OK=1 || true
fi

if [[ "${PYAIRPORTS_OK}" -eq 0 ]]; then
  echo "--- [pyairports] 策略 2/4: GitHub 源码 (NICTA/pyairports) ---"
  if pip install "pyairports @ git+https://github.com/NICTA/pyairports.git" 2>&1 | tail -3; then
    python3 -c "import pyairports" 2>/dev/null && PYAIRPORTS_OK=1 || true
  fi
fi

if [[ "${PYAIRPORTS_OK}" -eq 0 ]]; then
  echo "--- [pyairports] 策略 3/4: GitHub 源码 (ozeliger/pyairports) ---"
  if pip install "pyairports @ git+https://github.com/ozeliger/pyairports.git" 2>&1 | tail -3; then
    python3 -c "import pyairports" 2>/dev/null && PYAIRPORTS_OK=1 || true
  fi
fi

if [[ "${PYAIRPORTS_OK}" -eq 0 ]]; then
  echo "--- [pyairports] 策略 4/4: 替代包 airports-py + airportsdata ---"
  if pip install airports-py airportsdata 2>&1 | tail -3; then
    python3 -c "import pyairports" 2>/dev/null && PYAIRPORTS_OK=1 || true
  fi
fi

if [[ "${PYAIRPORTS_OK}" -eq 0 ]]; then
  echo "--- [pyairports] 兜底: 重装 outlines 让其拉传递依赖 ---"
  pip install --force-reinstall "outlines>=0.0.44,<0.1.0" 2>&1 | tail -3 || true
  python3 -c "import pyairports" 2>/dev/null && PYAIRPORTS_OK=1 || true
fi

if [[ "${PYAIRPORTS_OK}" -eq 0 ]]; then
  echo "  ❌ pyairports 所有策略都失败"
  echo "     pip show pyairports:"
  pip show pyairports 2>&1 | head -10 || true
  echo "     pip show airports-py:"
  pip show airports-py 2>&1 | head -10 || true
else
  echo "  ✅ pyairports 装上且可 import"
  pip show pyairports 2>/dev/null | head -3 || true
fi

# ── interegular (outlines 直接依赖, 之前漏了) ─────────────────────
pip install interegular 2>&1 | tail -2 || echo "  ⚠ interegular 装不上"

# ── 常用 server 运行时依赖 (一次性 bulk, 单包失败不阻塞) ─────────
# setuptools 必须显式装 (提供 pkg_resources), Python 3.12+ 不预装
for P in \
  "setuptools>=68" \
  orjson fastapi uvicorn uvloop pydantic msgspec python-multipart \
  hf_transfer decord soundfile pillow requests aiohttp psutil \
  "pyzmq>=25.1.2" "outlines>=0.0.44,<0.1.0" "prometheus_client>=0.20.0" \
  setproctitle diskcache cloudpickle tiktoken numba coloredlogs packaging \
  sentencepiece protobuf nvidia-ml-py openai IPython tqdm; do
  pip install "${P}" 2>&1 | tail -2 || echo "  ⚠ ${P} 装不上"
done

# xgrammar: 仅 xgrammar 后端需要, 默认是 outlines, 装不上不致命
pip install "xgrammar>=0.1.6" 2>&1 | tail -2 || echo "  (xgrammar 装不上不致命, 默认走 outlines 后端)"

echo ""
echo "=== [3/6] 装 flashinfer-python (cu124 + torch2.5, 非必需但推荐) ==="
# 注意: sglang 0.4.1 (2024-11) 原本要的是 flashinfer==0.1.6 (已 yanked)
# flashinfer.ai 索引上 cu124+torch2.5 只有 0.2.x, API 和 0.1.6 不兼容
# 装了不一定能用, 但装不上 sglang 可以用 triton 后端 fallback
FLASHINFER_INDEX="https://flashinfer.ai/whl/cu124/torch2.5/"
FLASHINFER_OK=0
for V in "0.2.0.post2" "0.2.1.post1" "0.2.1.post2" "0.2.2.post1" "0.2.2" "0.2.3" "0.2.4" "0.2.5"; do
  echo "--- 试 flashinfer-python==${V} ---"
  if pip install "flashinfer-python==${V}" --extra-index-url "${FLASHINFER_INDEX}" 2>&1 | tail -5; then
    FLASHINFER_OK=1; break
  fi
done
if [[ "${FLASHINFER_OK}" -eq 0 ]]; then
  echo "⚠ 钉版本失败, 试最新 flashinfer-python..."
  pip install flashinfer-python --extra-index-url "${FLASHINFER_INDEX}" 2>&1 | tail -5 \
    && FLASHINFER_OK=1 || true
fi
if [[ "${FLASHINFER_OK}" -eq 0 ]]; then
  pip install flashinfer-python 2>&1 | tail -5 || echo "  ❌ flashinfer 装不上 (server 用 --attention-backend triton 即可)"
fi

echo ""
echo "=== [4/6] 装 vllm (sglang 0.4.1 硬要 >=0.6.3.post1) ==="
# vllm 0.6.3.post1 会把 torch 拉到 2.4.0, 下一步再回退
VLLM_OK=0
for V in "0.6.3.post1" "0.6.3" "0.6.2" "0.6.1.post2"; do
  echo "--- 试 vllm==${V} ---"
  if pip install "vllm==${V}" 2>&1 | tail -5; then
    VLLM_OK=1; break
  fi
done
if [[ "${VLLM_OK}" -eq 0 ]]; then
  echo "⚠ 已知版本都失败, 试最新 vllm..."
  pip install vllm 2>&1 | tail -5 || echo "  ❌ vllm 装不上"
fi

echo ""
echo "=== [5/6] 强制 torch 回退到 2.5.1+cu124 ==="
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall --no-deps || echo "  ⚠ torch rollback 失败"

echo ""
echo "=== [6/6] 验证 import ==="
python3 -c "
mods = [
    # HARD: sglang server 启动必需
    ('sglang', 'HARD'),
    ('sglang.srt', 'HARD'),
    ('sglang.srt.server', 'HARD'),
    ('sglang.launch_server', 'HARD'),
    ('torch', 'HARD'),
    ('vllm', 'HARD'),
    ('sgl_kernel', 'HARD'),
    ('fastapi', 'HARD'),
    ('uvicorn', 'HARD'),
    ('uvloop', 'HARD'),
    ('orjson', 'HARD'),
    ('aiohttp', 'HARD'),
    ('pydantic', 'HARD'),
    ('msgspec', 'HARD'),
    ('multipart', 'HARD'),  # python-multipart
    ('zmq', 'HARD'),
    ('requests', 'HARD'),
    ('numpy', 'HARD'),
    ('setproctitle', 'HARD'),
    ('packaging', 'HARD'),
    ('prometheus_client', 'HARD'),
    ('psutil', 'HARD'),
    ('PIL', 'HARD'),  # pillow
    ('transformers', 'HARD'),
    ('huggingface_hub', 'HARD'),
    ('hf_transfer', 'HARD'),
    ('outlines', 'HARD'),
    ('interegular', 'HARD'),
    ('triton', 'HARD'),
    ('pyairports', 'HARD'),  # 本次修复的关键
    ('pkg_resources', 'HARD'),  # 来自 setuptools
    # SOFT
    ('flashinfer', 'SOFT'),
    ('xgrammar', 'SOFT'),
    ('tiktoken', 'SOFT'),
    ('sentencepiece', 'SOFT'),
    ('ray', 'SOFT'),
]
n_hard_ok = n_hard_fail = n_soft_ok = n_soft_fail = 0
fails = []
for mod, kind in mods:
    try:
        __import__(mod)
        mark = '✅'
        if kind == 'HARD': n_hard_ok += 1
        else: n_soft_ok += 1
    except Exception as e:
        mark = '❌'
        fails.append((mod, kind, type(e).__name__, str(e).splitlines()[0]))
        if kind == 'HARD': n_hard_fail += 1
        else: n_soft_fail += 1
    print(f'  {mark} [{kind:4s}] {mod}')

print()
print(f'HARD: {n_hard_ok} ok / {n_hard_fail} fail')
print(f'SOFT: {n_soft_ok} ok / {n_soft_fail} fail')
if n_hard_fail > 0:
    print()
    print('❌ HARD 依赖缺失 (server 启动会失败):')
    for mod, kind, etype, emsg in fails:
        if kind == 'HARD':
            print(f'   - {mod}: {etype}: {emsg}')

# 打印 torch / vllm / flashinfer 实际版本, 帮助诊断
print()
print('=== 关键版本 ===')
import torch
print(f'  torch         = {torch.__version__}')
print(f'  torch.cuda    = {torch.version.cuda}')
for pkg in ['vllm', 'sglang', 'sgl_kernel', 'flashinfer', 'outlines', 'pyairports', 'pyzmq', 'prometheus_client', 'interegular']:
    try:
        m = __import__(pkg)
        v = getattr(m, '__version__', getattr(m, 'VERSION', '?'))
        print(f'  {pkg:17s} = {v}')
    except Exception:
        print(f'  {pkg:17s} = (not installed)')
"

echo ""
echo "========================================="
echo " 下一步:"
echo "   bash repro/run_all.sh 5"
echo ""
echo " 注: smoke test 里 teacher SGLang 已改用 --attention-backend triton"
echo "     (避开 flashinfer 0.2.x vs sglang 0.4.1 API 不兼容问题)"
echo "========================================="
