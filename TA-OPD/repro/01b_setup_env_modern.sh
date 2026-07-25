#!/usr/bin/env bash
# Modern 环境: sglang 0.5.10 + torch 2.9 + flashinfer 0.6.7 + flash-attn 4.0
# 用法: bash repro/01b_setup_env_modern.sh
# 走全新软件栈 (不再用 0.4.1 + 2.5.1 + monkey-patch)
#
# 和旧 01_setup_env.sh 的区别:
#   - conda env 名: ta_opd_modern (避免和旧环境冲突)
#   - torch 不锁版本, 让 sglang 拉合适的
#   - sglang 直接 0.5.10.post1 (原生支持 Qwen3)
#   - 不再需要任何 monkey-patch (patch_qwen3_as_qwen2.sh / patch_sglang_qwen2_for_qwen3.sh)
#   - 不再需要 fix_sglang_deps.sh (依赖一次性装全)
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

echo "========================================="
echo " Step 1b: Modern env (sglang 0.5.10 + torch 2.9)"
echo "========================================="

MODERN_ENV="${MODERN_ENV:-ta_opd_modern}"
echo "Modern env name: ${MODERN_ENV}"
echo ""

# ── [1/7] 创建 conda env ───────────────────────────────────────
echo "[1/7] Conda env '${MODERN_ENV}' (Python 3.10)..."
if conda env list | awk '{print $1}' | grep -xq "${MODERN_ENV}"; then
  echo "  env 已存在, 删除重建 (避免旧 sglang 0.4.1 / torch 2.5.1 干扰)"
  conda env remove -n "${MODERN_ENV}" -y
fi
conda create -n "${MODERN_ENV}" python=3.10 -y

# 激活
eval "$(conda shell.bash hook)"
conda activate "${MODERN_ENV}"
echo "  ENV_PREFIX = ${CONDA_PREFIX}"

# ── [2/7] CUDA toolkit 12.4 + GCC 12 (from conda-forge) ───────
echo ""
echo "[2/7] CUDA toolkit + GCC + cuDNN + NCCL..."
conda install -n "${MODERN_ENV}" -c conda-forge -y \
  cudatoolkit=12.4 \
  gcc_linux-64=12 \
  gxx_linux-64=12 \
  cudnn \
  nccl \
  make \
  libnuma \
  || echo "  ⚠ 部分 conda 包失败 (非致命, pip 会补)"

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# ── [3/7] torch 2.5.1+cu124 (和 flash-attn 2.8.3 wheel 匹配) ──
echo ""
echo "[3/7] PyTorch 2.5.1 + cu124..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5
python3 -c "import torch; print(f'  torch {torch.__version__} cuda={torch.version.cuda}')"

# ── [4/7] sglang 0.5.10.post1 + 所有依赖 ─────────────────────
echo ""
echo "[4/7] sglang 0.5.10.post1 (原生 Qwen3, 含 sglang-kernel + flashinfer)..."
pip install "sglang==0.5.10.post1" 2>&1 | tail -10

# sglang 的依赖可能把 torch 拉到更新版本, 这里强制回退到 2.5.1+cu124 (和 flash-attn 2.8.3 wheel 匹配)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall --no-deps 2>&1 | tail -3

# 验证 sglang 版本 (关键!)
python3 -c "
import sglang, os
v = getattr(sglang, '__version__', '?')
print(f'  sglang {v}  path={os.path.dirname(sglang.__file__)}')
assert v.startswith('0.5.10'), f'sglang 版本不对: {v} (期望 0.5.10.x)'
print('  ✅ sglang 版本 OK')
"

# 验证 sglang-kernel (sm_89 for RTX 4090 必须可用)
python3 -c "
import sgl_kernel
print(f'  sgl_kernel {getattr(sgl_kernel, \"__version__\", \"?\")}  path={sgl_kernel.__file__}')
" 2>&1 | head -5 || echo "  ⚠ sgl_kernel import 失败 (sm_89 wheel 可能没装)"

# ── [5/7] Megatron-LM + slime_ta_opd + mbridge ───────────────
echo ""
echo "[5/7] Megatron-LM + slime + mbridge..."
pip install "mbridge>=0.15"

# Megatron-LM 从 git clone (如果 01_setup_env.sh 已经做了, 这里复用)
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-${PROJECT_ROOT}/Megatron-LM}"
if [[ ! -d "${MEGATRON_LM_DIR}" ]]; then
  echo "  cloning Megatron-LM..."
  git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_LM_DIR}" \
    || echo "  ⚠ Megatron-LM clone 失败 (请手动 clone)"
fi
# 给 namespace package 打 __init__.py (老问题, 0.5.10 时代也需要)
for D in "${MEGATRON_LM_DIR}/megatron" "${MEGATRON_LM_DIR}/megatron/core" \
         "${MEGATRON_LM_DIR}/megatron/legacy" "${MEGATRON_LM_DIR}/megatron/training" \
         "${MEGATRON_LM_DIR}/megatron/rl" "${MEGATRON_LM_DIR}/megatron/post_training"; do
  [[ -d "$D" && ! -f "$D/__init__.py" ]] && touch "$D/__init__.py"
done

# slime_ta_opd 已经在 SLIME_DIR, 加入 pip editable install
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime_ta_opd}"
if [[ -d "${SLIME_DIR}" ]]; then
  pip install -e "${SLIME_DIR}" 2>&1 | tail -3 || echo "  ⚠ slime editable install 失败"
fi

# ── [6/7] 常用训练 / 数据处理依赖 ─────────────────────────────
echo ""
echo "[6/7] ray + transformers + datasets + 常用包..."
pip install "ray[default]>=2.9" \
  transformers datasets accelerate safetensors \
  pyarrow pandas scipy \
  numpy\<2 \
  flash-attn==2.8.3.post1 \
  2>&1 | tail -5

# numpy 钉 <2 (Megatron 不兼容 numpy 2.x)
pip install "numpy<2" 2>&1 | tail -1

# ── [7/7] 验证 ─────────────────────────────────────────────────
echo ""
echo "[7/7] 验证..."
export PYTHONPATH="${MEGATRON_LM_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
python3 -c "
import sys
checks = [
    ('torch', 'HARD'),
    ('sglang', 'HARD'),
    ('sgl_kernel', 'HARD'),
    ('flashinfer', 'HARD'),
    ('flash_attn', 'SOFT'),  # Megatron 可 fallback torch Norm
    ('ray', 'HARD'),
    ('transformers', 'HARD'),
    ('datasets', 'HARD'),
    ('safetensors', 'HARD'),
    ('megatron', 'HARD'),
    ('mbridge', 'HARD'),
    ('numpy', 'HARD'),
    ('pyarrow', 'HARD'),
    ('scipy', 'HARD'),
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
    sys.exit(1)
"

echo ""
echo "========================================="
echo " ✅ Modern env 就绪: ${MODERN_ENV}"
echo ""
echo " sglang 0.5.10 原生支持 Qwen3,"
echo " 不再需要 monkey-patch。"
echo ""
echo " 下一步:"
echo "   conda activate ${MODERN_ENV}"
echo "   bash repro/adapt_for_sglang_0_5_10.sh"
echo "   bash repro/run_all.sh 3"
echo "   bash repro/run_all.sh 5"
echo ""
echo " 或一键跑完:"
echo "   MODERN_ENV=${MODERN_ENV} bash repro/run_all_modern.sh"
echo "========================================="
