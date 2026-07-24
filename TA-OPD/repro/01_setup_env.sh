#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

echo "========================================="
echo " Step 1: Setup conda environment & deps"
echo "========================================="

# ── 0. Detect CUDA ────────────────────────────────────────────────────────
echo "[0/9] Detecting CUDA installation..."
if [[ -z "${CUDA_HOME:-}" ]]; then
  # Try common locations
  for candidate in /usr/local/cuda /usr/local/cuda-12.4 /usr/local/cuda-12 \
                   /opt/cuda /opt/cuda-12.4 "${CONDA_PREFIX:-}/targets/x86_64-linux" \
                   "${HOME}/miniconda3/envs/${CONDA_ENV}"; do
    if [[ -f "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      break
    fi
  done
fi

if [[ -n "${CUDA_HOME:-}" ]] && [[ -f "${CUDA_HOME}/bin/nvcc" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  echo "  CUDA_HOME = ${CUDA_HOME}"
  echo "  nvcc = $(nvcc --version 2>/dev/null | tail -1 || echo 'not found')"
else
  echo "  WARNING: nvcc not found. Will try to install via conda."
  echo "  If flash-attn fails, install CUDA toolkit manually:"
  echo "    conda install -c nvidia cuda-toolkit"
fi

# ── 1. Create conda environment ───────────────────────────────────────────
echo "[1/9] Creating conda environment '${CONDA_ENV}' (Python 3.10)..."
if conda env list | grep -qw "${CONDA_ENV}"; then
  echo "  Environment '${CONDA_ENV}' already exists. Removing..."
  conda env remove -n "${CONDA_ENV}" -y
fi
conda create -n "${CONDA_ENV}" python=3.10 -y
activate_env

# ── 2. Install PyTorch 2.5.1 + CUDA 12.4 (FIRST, pinned) ────────────────
echo "[2/9] Installing PyTorch 2.5.1 + CUDA 12.4..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

# Verify torch version BEFORE installing anything else
python3 -c "
import torch
v = torch.__version__
print(f'  Installed torch {v}')
assert v.startswith('2.5.'), f'Expected torch 2.5.x, got {v}'
assert torch.cuda.is_available(), 'CUDA not available after torch install'
print(f'  CUDA: {torch.version.cuda}, GPU count: {torch.cuda.device_count()}')
"

# ── 3. Install SGLang WITHOUT upgrading torch ─────────────────────────────
echo "[3/9] Installing SGLang (pinned torch)..."
# Install sglang dependencies without pulling in a newer torch
pip install "sglang[all]>=0.4.0" || true
# Force torch back to 2.5.1 in case sglang upgraded it
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
echo "  Re-pinned torch to 2.5.1"

# ── 4. Install Ray and ML dependencies ───────────────────────────────────
echo "[4/9] Installing Ray and ML dependencies..."
pip install "ray[default]>=2.9"
pip install transformers datasets accelerate safetensors
pip install pyarrow pandas numpy scipy scikit-learn
pip install matplotlib seaborn
pip install einops tiktoken sentencepiece protobuf
pip install wandb tensorboard

# ── 5. Install flash-attn ────────────────────────────────────────────────
echo "[5/9] Installing flash-attn..."

# Re-detect CUDA_HOME after conda env activation
if [[ -z "${CUDA_HOME:-}" ]] || [[ ! -f "${CUDA_HOME}/bin/nvcc" ]]; then
  # Try conda env path
  CONDA_PREFIX="$(python3 -c 'import sys; print(sys.prefix)')"
  if [[ -f "${CONDA_PREFIX}/bin/nvcc" ]]; then
    export CUDA_HOME="${CONDA_PREFIX}"
  else
    # Install CUDA toolkit via conda
    echo "  Installing CUDA toolkit via conda..."
    conda install -c nvidia cuda-toolkit -y || conda install -c conda-forge cudatoolkit -y || true
    if [[ -f "${CONDA_PREFIX}/bin/nvcc" ]]; then
      export CUDA_HOME="${CONDA_PREFIX}"
    fi
  fi
fi

# Set CUDA_HOME for flash-attn compilation
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

if command -v nvcc &>/dev/null; then
  echo "  nvcc found at $(which nvcc), building flash-attn from source..."
  pip install flash-attn --no-build-isolation 2>&1 || {
    echo "  flash-attn build failed, trying pre-built wheel..."
    pip install flash-attn --no-build-isolation --find-links \
      "https://github.com/Dao-AILab/flash-attention/releases" 2>&1 || {
      echo "  WARNING: flash-attn installation failed."
      echo "  Training may still work without it (--attention-backend flash will be unavailable)."
      echo "  Try: MAX_JOBS=4 pip install flash-attn --no-build-isolation"
    }
  }
else
  echo "  nvcc not found, trying pre-built flash-attn wheel..."
  pip install flash-attn 2>&1 || {
    echo "  WARNING: flash-attn installation failed."
    echo "  Install CUDA toolkit: conda install -c nvidia cuda-toolkit"
    echo "  Then: MAX_JOBS=4 pip install flash-attn --no-build-isolation"
  }
fi

# ── 6. Install lm-eval ───────────────────────────────────────────────────
echo "[6/9] Installing lm-eval..."
pip install lm-eval>=0.4.5

# ── 7. Final torch pin (safety net) ──────────────────────────────────────
echo "[7/9] Final torch version check..."
CURRENT_TORCH="$(python3 -c 'import torch; print(torch.__version__)')"
if [[ ! "${CURRENT_TORCH}" == 2.5.* ]]; then
  echo "  WARNING: torch was upgraded to ${CURRENT_TORCH}, re-pinning to 2.5.1..."
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
fi

# ── 8. Clone and patch Megatron-LM ───────────────────────────────────────
echo "[8/9] Setting up Megatron-LM..."
if [[ ! -d "${MEGATRON_LM_DIR}" ]]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_LM_DIR}"
  echo "  Cloned Megatron-LM"
else
  echo "  Megatron-LM already exists at ${MEGATRON_LM_DIR}"
fi

cd "${MEGATRON_LM_DIR}"
PATCH_FILE="${REPO_ROOT}/docs/megatron_apex_patches.patch"
if [[ -f "${PATCH_FILE}" ]]; then
  if git apply --check "${PATCH_FILE}" 2>/dev/null; then
    git apply "${PATCH_FILE}"
    echo "  Applied DCP patches"
  else
    echo "  Patches already applied or not applicable"
  fi
else
  echo "  WARNING: patch file not found at ${PATCH_FILE}"
fi
cd "${PROJECT_ROOT}"

# ── 9. Code patches + pyarrow fix + curl wrapper ─────────────────────────
echo "[9/9] Applying code patches and fixes..."

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
echo "  pyarrow SIGSEGV fix installed"

# Fix 1: reloadable_process_group.py
RPG_FILE="${SLIME_DIR}/slime/utils/reloadable_process_group.py"
if [[ -f "${RPG_FILE}" ]] && grep -q 'super().__init__(rank=' "${RPG_FILE}" 2>/dev/null; then
  python3 -c "
import re, pathlib
p = pathlib.Path('${RPG_FILE}')
t = p.read_text()
t = re.sub(r'super\(\)\.__init__\(rank=.*?,\s*size=.*?\)', 'super().__init__(0, 0)', t)
p.write_text(t)
print('  Fixed ProcessGroup constructor')
"
fi

# Fix 2: placement_group.py
PG_FILE="${SLIME_DIR}/slime/ray/placement_group.py"
if [[ -f "${PG_FILE}" ]] && grep -q 'RolloutManager\.options(' "${PG_FILE}" 2>/dev/null; then
  python3 -c "
import pathlib
p = pathlib.Path('${PG_FILE}')
t = p.read_text()
t = t.replace('RolloutManager.options(', 'ray.remote(RolloutManager).options(')
p.write_text(t)
print('  Fixed RolloutManager ray.remote')
"
fi

# Fix 3: rollout.py
ROLL_FILE="${SLIME_DIR}/slime/ray/rollout.py"
if [[ -f "${ROLL_FILE}" ]] && grep -q '@ray\.remote' "${ROLL_FILE}" 2>/dev/null; then
  python3 -c "
import pathlib
p = pathlib.Path('${ROLL_FILE}')
t = p.read_text()
t = t.replace('@ray.remote\nclass RolloutManager', 'class RolloutManager')
if 'import faulthandler' not in t:
    t = 'import faulthandler as _fh\nimport sys as _sys\n_fh.enable(file=_sys.stderr, all_threads=True)\n\n' + t
p.write_text(t)
print('  Fixed rollout.py')
"
fi

# curl wrapper
CURL_BIN_DIR="${HOME}/.local/bin"
mkdir -p "${CURL_BIN_DIR}"
cat > "${CURL_BIN_DIR}/curl" << 'CURLEOF'
#!/usr/bin/env bash
exec env -u LD_LIBRARY_PATH /usr/bin/curl "$@"
CURLEOF
chmod +x "${CURL_BIN_DIR}/curl"

# ── Verify ────────────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'  torch {torch.__version__} (must be 2.5.x)')
assert torch.__version__.startswith('2.5.'), f'WRONG torch version: {torch.__version__}'
print(f'  CUDA available: {torch.cuda.is_available()}, version: {torch.version.cuda}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
try:
    import flash_attn; print(f'  flash_attn {flash_attn.__version__}')
except ImportError:
    print('  flash_attn: NOT INSTALLED (training may still work)')
try:
    import sglang; print(f'  sglang {sglang.__version__}')
except ImportError:
    print('  sglang: NOT INSTALLED')
try:
    import ray; print(f'  ray {ray.__version__}')
except ImportError:
    print('  ray: NOT INSTALLED')
import transformers; print(f'  transformers {transformers.__version__}')
import pyarrow; print(f'  pyarrow {pyarrow.__version__}')
import megatron; print(f'  megatron OK')
"

echo ""
echo "========================================="
echo " Environment setup complete!"
echo " Conda env: ${CONDA_ENV}"
echo " CUDA_HOME: ${CUDA_HOME:-NOT SET}"
echo "========================================="
