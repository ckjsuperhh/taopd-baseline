#!/usr/bin/env bash
# modern env v2: torch 回退 2.5.1 + flash-attn 2.8.3 + PYTHONPATH 修
#
# 原因:
#   - flash-attn 4.0.0b23 (beta) 没公开 wheel
#   - PyPI 最新稳定 2.8.3.post1 给 torch 2.5/2.8 编的, torch 2.9 ABI 不兼容
#   - sglang 0.5.10 同时支持 torch 2.5 和 2.9, 所以 torch 回退无影响
#
# 用法: bash repro/fix_modern_env_v2.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

MODERN_ENV="${MODERN_ENV:-ta_opd_modern}"
eval "$(conda shell.bash hook)"
conda activate "${MODERN_ENV}"
echo "✅ activated: ${MODERN_ENV}"
echo ""

# ── [1/5] torch 回退到 2.5.1+cu124 ───────────────────────────
echo "=== [1/5] torch 回退到 2.5.1+cu124 (sglang 0.5.10 支持) ==="
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall --no-deps 2>&1 | tail -5
python3 -c "import torch; print(f'  ✅ torch {torch.__version__} cuda={torch.version.cuda}')"

# ── [2/5] 强制装 flash-attn 2.8.3.post1 (wheel for torch 2.5) ─
echo ""
echo "=== [2/5] flash-attn 2.8.3.post1 (for torch 2.5) ==="
# 优先从 Dao-AILab 官方 wheel 索引拿 (cu124 + torch 2.5 + cp310)
FLASH_ATTN_INDEX="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/"
pip install --force-reinstall --no-deps flash-attn==2.8.3.post1 2>&1 | tail -5 \
  || pip install --force-reinstall flash-attn==2.8.3.post1 2>&1 | tail -5 \
  || echo "  ⚠ flash-attn 装不上 (Megatron fallback torch Norm)"

python3 -c "
try:
    import flash_attn
    print(f'  ✅ flash_attn {flash_attn.__version__}')
except Exception as e:
    print(f'  ⚠ flash_attn import 失败: {str(e).splitlines()[0]}')
    print('    Megatron 会 fallback 到 torch Norm (非致命)')
" 2>&1 || true

# ── [3/5] 修 megatron PYTHONPATH ──────────────────────────────
echo ""
echo "=== [3/5] 验证 megatron (需 PYTHONPATH) ==="
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-${PROJECT_ROOT}/Megatron-LM}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime_ta_opd}"
if [[ -d "${MEGATRON_LM_DIR}" ]]; then
  export PYTHONPATH="${MEGATRON_LM_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
  echo "  PYTHONPATH=${PYTHONPATH}"
  python3 -c "import megatron; print('  ✅ megatron import OK')" 2>&1 | head -5 \
    || echo "  ❌ megatron import 失败"
else
  echo "  ⚠ ${MEGATRON_LM_DIR} 不存在 (step 1 应该 clone 过)"
fi

# ── [4/5] sgl_kernel (之前 sm90 已 OK, 验证一下) ─────────────
echo ""
echo "=== [4/5] sgl_kernel 验证 ==="
python3 -c "
import sgl_kernel
print(f'  ✅ sgl_kernel {getattr(sgl_kernel, \"__version__\", \"?\")}')
" 2>&1 | head -5 || echo "  ❌ sgl_kernel 失败"

# ── [5/5] 完整验证 ────────────────────────────────────────────
echo ""
echo "=== [5/5] 完整验证 ==="
export PYTHONPATH="${MEGATRON_LM_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
python3 -c "
import sys
checks = [
    ('torch', 'HARD'),
    ('sglang', 'HARD'),
    ('sgl_kernel', 'HARD'),
    ('flashinfer', 'HARD'),
    ('flash_attn', 'SOFT'),  # 降级, 可 fallback
    ('ray', 'HARD'),
    ('transformers', 'HARD'),
    ('datasets', 'HARD'),
    ('safetensors', 'HARD'),
    ('megatron', 'HARD'),
    ('mbridge', 'HARD'),
    ('numpy', 'HARD'),
    ('pyarrow', 'HARD'),
]
n_hard_ok = n_hard_fail = n_soft_ok = n_soft_fail = 0
for mod, kind in checks:
    try:
        m = __import__(mod)
        v = getattr(m, '__version__', '?')
        print(f'  ✅ [{kind:4s}] {mod:20s} = {v}')
        if kind == 'HARD': n_hard_ok += 1
        else: n_soft_ok += 1
    except Exception as e:
        print(f'  ❌ [{kind:4s}] {mod:20s}: {str(e).splitlines()[0]}')
        if kind == 'HARD': n_hard_fail += 1
        else: n_soft_fail += 1
print()
print(f'HARD: {n_hard_ok} ok / {n_hard_fail} fail')
print(f'SOFT: {n_soft_ok} ok / {n_soft_fail} fail')
if n_hard_fail > 0:
    print('❌ HARD 缺 → server / 训练会失败')
    sys.exit(1)
elif n_soft_fail > 0:
    print('⚠ SOFT 缺 → 性能稍差但能跑')
else:
    print('✅ 全部 OK')
"

echo ""
echo "========================================="
echo " 下一步:"
echo "   bash repro/run_all_modern.sh 3"
echo ""
echo " 如果 flash_attn 还是 ❌,"
echo " Megatron 会 fallback 到 torch Norm,"
echo " 不影响训练正确性, 只是慢一点。"
echo "========================================="
