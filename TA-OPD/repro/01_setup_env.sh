#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

echo "========================================="
echo " Step 1: Setup conda environment"
echo " (apex 验证过的方案)"
echo "========================================="

# ── 1. Conda env ─────────────────────────────────────────────────────────
echo "[1/7] Conda env '${CONDA_ENV}' (Python 3.10)..."
if conda env list | grep -qw "${CONDA_ENV}"; then
  echo "  Removing existing env..."
  conda env remove -n "${CONDA_ENV}" -y
fi
conda create -n "${CONDA_ENV}" python=3.10 pip -y
activate_env

ENV_PREFIX="$(python3 -c 'import sys; print(sys.prefix)')"
echo "  ENV_PREFIX = ${ENV_PREFIX}"

# ── 2. PyTorch 2.5.1 + CUDA 12.4 ────────────────────────────────────────
# apex 用的就是这个版本，4090 sm_89 也完全兼容 cu124
echo "[2/7] PyTorch 2.5.1 + cu124..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

python3 -c "
import torch
print(f'  torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}')
assert torch.__version__.startswith('2.5.'), f'Wrong torch: {torch.__version__}'
"

# ── 3. 基础依赖（不会动 torch 的包）────────────────────────────────────
echo "[3/7] Basic deps..."
pip install transformers datasets accelerate safetensors
pip install pyarrow pandas numpy scipy scikit-learn
pip install matplotlib seaborn
pip install einops tiktoken sentencepiece protobuf
pip install wandb tensorboard
pip install "ray[default]>=2.9"
pip install lm-eval>=0.4.5

# ── 4. SGLang（安装后强制回退 torch）──────────────────────────────────
echo "[4/7] SGLang..."
pip install "sglang[all]>=0.4.0" || pip install sglang

# sglang 可能拉高 torch，立刻回退
echo "  Re-pinning torch..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall --no-deps

python3 -c "import torch; print(f'  torch={torch.__version__} (must be 2.5.x)'); assert torch.__version__.startswith('2.5.')"

# ── 5. flash-attn ────────────────────────────────────────────────────────
echo "[5/7] flash-attn..."
# 找 CUDA_HOME
if [[ -z "${CUDA_HOME:-}" ]] || [[ ! -f "${CUDA_HOME}/bin/nvcc" ]]; then
  for d in /usr/local/cuda /usr/local/cuda-12* /opt/cuda "${ENV_PREFIX}"; do
    if [[ -f "$d/bin/nvcc" ]]; then
      export CUDA_HOME="$d"
      break
    fi
  done
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

if [[ -f "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "  CUDA_HOME=${CUDA_HOME}, nvcc=$(nvcc --version | tail -1)"
  MAX_JOBS=4 pip install flash-attn --no-build-isolation 2>&1 || \
    echo "  WARNING: flash-attn 编译失败，训练时可用 --attention-backend eager 替代"
else
  echo "  WARNING: nvcc 找不到。手动设置 CUDA_HOME 后重新运行此脚本。"
  echo "  或: conda install -c nvidia cuda-toolkit"
fi

# ── 6. Megatron-LM + DCP patch ──────────────────────────────────────────
echo "[6/7] Megatron-LM..."
if [[ ! -d "${MEGATRON_LM_DIR}" ]]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_LM_DIR}"
fi

cd "${MEGATRON_LM_DIR}"
PATCH_FILE="${REPO_ROOT}/docs/megatron_apex_patches.patch"
if [[ -f "${PATCH_FILE}" ]] && git apply --check "${PATCH_FILE}" 2>/dev/null; then
  git apply "${PATCH_FILE}"
  echo "  Applied DCP patches"
fi
cd "${PROJECT_ROOT}"

# ── 7. 代码修复 + pyarrow fix（和 apex 完全一致）───────────────────────
echo "[7/7] Code fixes (apex 同款)..."

# pyarrow SIGSEGV fix — 和 apex 一模一样的 sitecustomize.py
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

# LD_LIBRARY_PATH — 和 apex 一样：conda lib 在前
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# ProcessGroup 构造修复
RPG="${SLIME_DIR}/slime/utils/reloadable_process_group.py"
if [[ -f "${RPG}" ]] && grep -q 'super().__init__(rank=' "${RPG}" 2>/dev/null; then
  sed -i 's/super().__init__(rank=[^)]*)/super().__init__(0, 0)/' "${RPG}"
  echo "  ProcessGroup ctor fixed"
fi

# placement_group.py — RolloutManager
PG="${SLIME_DIR}/slime/ray/placement_group.py"
if [[ -f "${PG}" ]] && grep -q 'RolloutManager\.options(' "${PG}" 2>/dev/null; then
  sed -i 's/RolloutManager\.options(/ray.remote(RolloutManager).options(/' "${PG}"
  echo "  RolloutManager ray.remote fixed"
fi

# rollout.py — @ray.remote + faulthandler
RL="${SLIME_DIR}/slime/ray/rollout.py"
if [[ -f "${RL}" ]] && grep -q '@ray\.remote' "${RL}" 2>/dev/null; then
  sed -i '/@ray\.remote/d' "${RL}"
  if ! grep -q 'import faulthandler' "${RL}"; then
    sed -i '1i import faulthandler as _fh\nimport sys as _sys\n_fh.enable(file=_sys.stderr, all_threads=True)\n' "${RL}"
  fi
  echo "  rollout.py fixed"
fi

# curl wrapper（和 apex 一样）
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
print(f'  {ok} torch {v} (need 2.5.x)')
print(f'     CUDA={torch.version.cuda}, GPUs={torch.cuda.device_count() if torch.cuda.is_available() else 0}')
try:
    import flash_attn; print(f'  ✅ flash_attn {flash_attn.__version__}')
except: print('  ⚠️  flash_attn not installed')
try:
    import sglang; print(f'  ✅ sglang {sglang.__version__}')
except: print('  ❌ sglang not installed')
try:
    import ray; print(f'  ✅ ray {ray.__version__}')
except: print('  ❌ ray not installed')
import transformers; print(f'  ✅ transformers {transformers.__version__}')
import pyarrow; print(f'  ✅ pyarrow {pyarrow.__version__}')
import megatron; print(f'  ✅ megatron')
"

echo ""
echo "========================================="
echo " Done! Conda env: ${CONDA_ENV}"
echo "========================================="
