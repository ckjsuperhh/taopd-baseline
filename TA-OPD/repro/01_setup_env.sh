#!/usr/bin/env bash
set -eo pipefail  # 不能用 -u：conda 内部 deactivate 脚本有 unbound variable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

echo "========================================="
echo " Step 1: Setup conda environment"
echo " (完全复制 apex 的方案)"
echo "========================================="

# ── 1. Conda env ─────────────────────────────────────────────────────────
echo "[1/6] Conda env '${CONDA_ENV}' (Python 3.10)..."
if conda env list | grep -qw "${CONDA_ENV}"; then
  echo "  Removing existing..."
  conda env remove -n "${CONDA_ENV}" -y
fi
conda create -n "${CONDA_ENV}" python=3.10 pip -y
activate_env

ENV_PREFIX="$(python3 -c 'import sys; print(sys.prefix)')"
echo "  ENV_PREFIX = ${ENV_PREFIX}"

# ── 2. conda-forge: CUDA toolkit 12.4 + GCC 12 + cuDNN + NCCL ──────────
# apex 上就是这么装的，nvcc 直接在 conda env 的 bin/ 里
echo "[2/6] conda install CUDA toolkit 12.4 + GCC 12 + cuDNN + NCCL..."
conda install -c conda-forge -y \
  cuda-toolkit=12.4 \
  gcc=12.4 \
  gxx=12.4 \
  cudnn \
  nccl \
  curl \
  gdb \
  tmux

# 验证 nvcc
export CUDA_HOME="${ENV_PREFIX}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
echo "  nvcc: $(which nvcc) — $(nvcc --version 2>/dev/null | tail -1)"
echo "  gcc:  $(which gcc) — $(gcc --version 2>/dev/null | head -1)"
echo "  CUDA_HOME = ${CUDA_HOME}"

# ── 3. PyTorch 2.5.1 + cu124 ────────────────────────────────────────────
echo "[3/6] pip install torch 2.5.1 + cu124..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

python3 -c "
import torch
print(f'  torch={torch.__version__} cuda={torch.version.cuda}')
assert torch.__version__.startswith('2.5.'), f'Wrong torch: {torch.__version__}'
assert torch.cuda.is_available(), 'CUDA not available'
"

# ── 4. SGLang + flash-attn + 其他依赖 ──────────────────────────────────
echo "[4/6] pip install sglang + flash-attn + deps..."
pip install "sglang[all]>=0.4.0" || pip install sglang
# sglang 可能拉高 torch，回退
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall --no-deps

MAX_JOBS=4 pip install flash-attn --no-build-isolation

pip install "ray[default]>=2.9"
pip install transformers datasets accelerate safetensors
pip install pyarrow pandas numpy scipy scikit-learn
pip install matplotlib seaborn
pip install einops tiktoken sentencepiece protobuf
pip install wandb tensorboard
pip install lm-eval>=0.4.5

# 最终检查 torch 没被升级
CURRENT="$(python3 -c 'import torch; print(torch.__version__)')"
if [[ ! "${CURRENT}" == 2.5.* ]]; then
  echo "  WARNING: torch 被升级到 ${CURRENT}，回退中..."
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
fi

# ── 5. Megatron-LM + DCP patch ──────────────────────────────────────────
echo "[5/6] Megatron-LM..."
if [[ ! -d "${MEGATRON_LM_DIR}" ]]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_LM_DIR}"
fi
cd "${MEGATRON_LM_DIR}"
PATCH="${REPO_ROOT}/docs/megatron_apex_patches.patch"
if [[ -f "${PATCH}" ]] && git apply --check "${PATCH}" 2>/dev/null; then
  git apply "${PATCH}"
  echo "  Applied DCP patches"
fi
cd "${PROJECT_ROOT}"

# ── 6. 代码修复（和 apex 完全一样）─────────────────────────────────────
echo "[6/6] Code fixes..."

# pyarrow SIGSEGV fix
SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
cat > "${SITE_PACKAGES}/sitecustomize.py" << 'PYEOF'
import os
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"
try:
    import pyarrow
    pool = pyarrow.system_memory_pool()
    pyarrow.set_memory_pool(pool)
except Exception:
    pass
PYEOF

# LD_LIBRARY_PATH — conda lib 在前（和 apex 一样）
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# ProcessGroup ctor
RPG="${SLIME_DIR}/slime/utils/reloadable_process_group.py"
if [[ -f "${RPG}" ]] && grep -q 'super().__init__(rank=' "${RPG}" 2>/dev/null; then
  sed -i 's/super().__init__(rank=[^)]*)/super().__init__(0, 0)/' "${RPG}"
  echo "  ProcessGroup ctor fixed"
fi

# RolloutManager
PG="${SLIME_DIR}/slime/ray/placement_group.py"
if [[ -f "${PG}" ]] && grep -q 'RolloutManager\.options(' "${PG}" 2>/dev/null; then
  sed -i 's/RolloutManager\.options(/ray.remote(RolloutManager).options(/' "${PG}"
  echo "  RolloutManager fixed"
fi

RL="${SLIME_DIR}/slime/ray/rollout.py"
if [[ -f "${RL}" ]] && grep -q '@ray\.remote' "${RL}" 2>/dev/null; then
  sed -i '/@ray\.remote/d' "${RL}"
  if ! grep -q 'import faulthandler' "${RL}"; then
    sed -i '1i import faulthandler as _fh\nimport sys as _sys\n_fh.enable(file=_sys.stderr, all_threads=True)\n' "${RL}"
  fi
  echo "  rollout.py fixed"
fi

# curl wrapper（conda curl 已经装了，但保险起见还是加 wrapper）
mkdir -p "${HOME}/.local/bin"
cat > "${HOME}/.local/bin/curl" << 'EOF'
#!/usr/bin/env bash
exec env -u LD_LIBRARY_PATH /usr/bin/curl "$@"
EOF
chmod +x "${HOME}/.local/bin/curl"
export PATH="${HOME}/.local/bin:${PATH}"

# ── 验证 ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
python3 -c "
import torch
v = torch.__version__
ok = '✅' if v.startswith('2.5.') else '❌'
print(f'  {ok} torch {v} cuda={torch.version.cuda} GPUs={torch.cuda.device_count()}')
try:
    import flash_attn; print(f'  ✅ flash_attn {flash_attn.__version__}')
except: print('  ❌ flash_attn')
try:
    import sglang; print(f'  ✅ sglang {sglang.__version__}')
except: print('  ❌ sglang')
try:
    import ray; print(f'  ✅ ray {ray.__version__}')
except: print('  ❌ ray')
import transformers; print(f'  ✅ transformers {transformers.__version__}')
import pyarrow; print(f'  ✅ pyarrow {pyarrow.__version__}')
import megatron; print(f'  ✅ megatron')
"
echo ""
echo "========================================="
echo " Done! conda env: ${CONDA_ENV}"
echo " CUDA_HOME: ${CUDA_HOME}"
echo "========================================="
