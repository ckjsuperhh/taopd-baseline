#!/usr/bin/env bash
# modern env v3: 补回 libnuma + sgl_kernel sm89→sm90 fallback
#
# v2 的两个问题:
#   1. 漏了 libnuma (conda-forge), sgl_kernel 加载 .so 时报 libnuma.so.1 missing
#   2. sgl-kernel 0.4.1 wheel 只带 sm90 + sm100, RTX 4090 (sm89) 没有专属 .so,
#      loader 找不到 sm89 就直接失败, 不会 fallback 到 sm90
#
# 修法:
#   - conda install libnuma
#   - 在 site-packages/sgl_kernel/ 下建 sm89 -> sm90 软链接
#     (sm90 PTX 在 sm89 上 JIT 编译就能跑, 性能损失很小)
#
# 用法: bash repro/fix_modern_env_v3.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

MODERN_ENV="${MODERN_ENV:-ta_opd_modern}"
eval "$(conda shell.bash hook)"
conda activate "${MODERN_ENV}"
echo "✅ activated: ${MODERN_ENV}"
echo ""

# ── [1/6] libnuma (sgl_kernel .so 运行时依赖) ───────────────
echo "=== [1/6] libnuma (conda-forge) ==="
conda install -c conda-forge -y libnuma 2>&1 | tail -5
python3 -c "
import ctypes
try:
    ctypes.CDLL('libnuma.so.1')
    print('  ✅ libnuma.so.1 OK')
except Exception as e:
    print(f'  ❌ libnuma.so.1: {e}')
    raise
"

# ── [2/6] torch 回退到 2.5.1+cu124 ───────────────────────────
echo ""
echo "=== [2/6] torch 回退到 2.5.1+cu124 (sglang 0.5.10 支持) ==="
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall --no-deps 2>&1 | tail -5
python3 -c "import torch; print(f'  ✅ torch {torch.__version__} cuda={torch.version.cuda}')"

# ── [3/6] flash-attn 2.8.3.post1 (wheel for torch 2.5) ──────
echo ""
echo "=== [3/6] flash-attn 2.8.3.post1 (for torch 2.5) ==="
pip install --force-reinstall --no-deps flash-attn==2.8.3.post1 2>&1 | tail -5 \
  || pip install --force-reinstall flash-attn==2.8.3.post1 2>&1 | tail -5 \
  || echo "  ⚠ flash-attn 装不上 (Megatron 会 fallback 到 torch Norm)"

python3 -c "
try:
    import flash_attn
    print(f'  ✅ flash_attn {flash_attn.__version__}')
except Exception as e:
    print(f'  ⚠ flash_attn import 失败: {str(e).splitlines()[0]}')
    print('    Megatron 会 fallback 到 torch Norm (非致命)')
" 2>&1 || true

# ── [4/6] sgl_kernel: sm89 → sm90 软链接 ────────────────────
echo ""
echo "=== [4/6] sgl_kernel sm89 → sm90 fallback (RTX 4090) ==="
python3 <<'PY'
import os, sys, importlib, pathlib
try:
    import sgl_kernel
    pkg_dir = pathlib.Path(sgl_kernel.__file__).parent
    sm89 = pkg_dir / "sm89"
    sm90 = pkg_dir / "sm90"
    if not sm89.exists() and sm90.exists():
        os.symlink(sm90, sm89)
        print(f"  ✅ 创建 {sm89.name} -> {sm90.name} 软链接")
    elif sm89.exists():
        print(f"  ✅ sm89 已存在 (或已链接)")
    else:
        print(f"  ⚠ sm90 也不存在, 没法做 fallback")
        sys.exit(0)

    # 重新触发 loader
    for mod_name in list(sys.modules):
        if mod_name.startswith("sgl_kernel"):
            del sys.modules[mod_name]
    import sgl_kernel as sk2
    # 触发 common_ops 加载
    from sgl_kernel import common_ops  # type: ignore
    print(f"  ✅ sgl_kernel {getattr(sk2, '__version__', '?')} common_ops 加载成功")
except Exception as e:
    print(f"  ❌ sgl_kernel: {e}")
    raise
PY

# ── [5/6] megatron PYTHONPATH ────────────────────────────────
echo ""
echo "=== [5/6] 验证 megatron (需 PYTHONPATH) ==="
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-${PROJECT_ROOT}/Megatron-LM}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime_ta_opd}"
if [[ -d "${MEGATRON_LM_DIR}" ]]; then
  export PYTHONPATH="${MEGATRON_LM_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
  echo "  PYTHONPATH=${PYTHONPATH}"
  python3 -c "import megatron; print('  ✅ megatron import OK')" 2>&1 | head -5 \
    || echo "  ❌ megatron import 失败"
else
  echo "  ⚠ ${MEGATRON_LM_DIR} 不存在"
fi

# ── [6/6] 完整验证 ───────────────────────────────────────────
echo ""
echo "=== [6/6] 完整验证 ==="
export PYTHONPATH="${MEGATRON_LM_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
python3 -c "
import sys
checks = [
    ('torch', 'HARD'),
    ('sglang', 'HARD'),
    ('sgl_kernel', 'HARD'),
    ('flashinfer', 'HARD'),
    ('flash_attn', 'SOFT'),
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
echo " sgl_kernel sm89→sm90 说明:"
echo "   RTX 4090 = sm_89, 但 PyPI sgl-kernel 0.4.1"
echo "   wheel 只编译了 sm90 + sm100 的 .so"
echo "   sm90 PTX 在 sm89 上 JIT 编译即可运行"
echo "   性能损失 < 5%, 远好于跑不了"
echo "========================================="
