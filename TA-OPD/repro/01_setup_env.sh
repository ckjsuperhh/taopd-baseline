#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

echo "========================================="
echo " Step 1: Setup conda environment & deps"
echo "========================================="

# ── 1. Create conda environment ───────────────────────────────────────────
echo "[1/8] Creating conda environment '${CONDA_ENV}' (Python 3.10)..."
if conda env list | grep -qw "${CONDA_ENV}"; then
  echo "  Environment '${CONDA_ENV}' already exists. Removing..."
  conda env remove -n "${CONDA_ENV}" -y
fi
conda create -n "${CONDA_ENV}" python=3.10 -y
activate_env

# ── 2. Install PyTorch 2.5.1 + CUDA 12.4 ─────────────────────────────────
echo "[2/8] Installing PyTorch 2.5.1 + CUDA 12.4..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

# ── 3. Install core ML dependencies ───────────────────────────────────────
echo "[3/8] Installing SGLang, Ray, and ML dependencies..."
pip install "sglang[all]>=0.4.0"
pip install "ray[default]>=2.9"
pip install transformers datasets accelerate safetensors
pip install pyarrow pandas numpy scipy scikit-learn
pip install matplotlib seaborn
pip install flash-attn --no-build-isolation
pip install einops tiktoken sentencepiece protobuf
pip install wandb tensorboard
pip install lm-eval>=0.4.5

# ── 4. Clone and patch Megatron-LM ───────────────────────────────────────
echo "[4/8] Setting up Megatron-LM..."
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

# ── 5. Fix pyarrow SIGSEGV (jemalloc/mimalloc conflict) ──────────────────
echo "[5/8] Installing pyarrow SIGSEGV fix (sitecustomize.py)..."
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
echo "  Written sitecustomize.py to ${SITE_PACKAGES}/"

# ── 6. Patch slime code for torch 2.5.1 compatibility ────────────────────
echo "[6/8] Patching slime code for torch 2.5.1..."

# Fix 1: reloadable_process_group.py — ProcessGroup positional args
RPG_FILE="${SLIME_DIR}/slime/utils/reloadable_process_group.py"
if [[ -f "${RPG_FILE}" ]]; then
  if grep -q 'super().__init__(rank=' "${RPG_FILE}" 2>/dev/null; then
    python3 -c "
import re, pathlib
p = pathlib.Path('${RPG_FILE}')
t = p.read_text()
t = re.sub(
    r'super\(\)\.__init__\(rank=.*?,\s*size=.*?\)',
    'super().__init__(0, 0)',
    t
)
p.write_text(t)
print('  Fixed ProcessGroup constructor (positional args)')
"
  else
    echo "  ProcessGroup constructor already fixed"
  fi
fi

# Fix 2: placement_group.py — ray.remote() wrapping
PG_FILE="${SLIME_DIR}/slime/ray/placement_group.py"
if [[ -f "${PG_FILE}" ]]; then
  if grep -q 'RolloutManager\.options(' "${PG_FILE}" 2>/dev/null; then
    python3 -c "
import pathlib
p = pathlib.Path('${PG_FILE}')
t = p.read_text()
t = t.replace('RolloutManager.options(', 'ray.remote(RolloutManager).options(')
p.write_text(t)
print('  Fixed RolloutManager ray.remote wrapping')
"
  else
    echo "  RolloutManager ray.remote already fixed"
  fi
fi

# Fix 3: rollout.py — remove @ray.remote decorator, add faulthandler
ROLL_FILE="${SLIME_DIR}/slime/ray/rollout.py"
if [[ -f "${ROLL_FILE}" ]]; then
  if grep -q '@ray\.remote' "${ROLL_FILE}" 2>/dev/null; then
    python3 -c "
import pathlib
p = pathlib.Path('${ROLL_FILE}')
t = p.read_text()
t = t.replace('@ray.remote\nclass RolloutManager', 'class RolloutManager')
if 'import faulthandler' not in t:
    t = 'import faulthandler as _fh\nimport sys as _sys\n_fh.enable(file=_sys.stderr, all_threads=True)\n\n' + t
p.write_text(t)
print('  Fixed RolloutManager: removed @ray.remote, added faulthandler')
"
  else
    echo "  rollout.py already fixed"
  fi
fi

# ── 7. Install curl wrapper (LD_LIBRARY_PATH conflict) ───────────────────
echo "[7/8] Installing curl wrapper..."
CURL_BIN_DIR="${HOME}/.local/bin"
mkdir -p "${CURL_BIN_DIR}"
cat > "${CURL_BIN_DIR}/curl" << 'CURLEOF'
#!/usr/bin/env bash
exec env -u LD_LIBRARY_PATH /usr/bin/curl "$@"
CURLEOF
chmod +x "${CURL_BIN_DIR}/curl"
export PATH="${CURL_BIN_DIR}:${PATH}"
echo "  curl wrapper installed at ${CURL_BIN_DIR}/curl"

# ── 8. Verify installation ───────────────────────────────────────────────
echo "[8/8] Verifying installation..."
python3 -c "
import torch; print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB)')
import sglang; print(f'  sglang {sglang.__version__}')
import ray; print(f'  ray {ray.__version__}')
import transformers; print(f'  transformers {transformers.__version__}')
import pyarrow; print(f'  pyarrow {pyarrow.__version__}')
import megatron; print(f'  megatron OK')
"

echo ""
echo "========================================="
echo " Environment setup complete!"
echo " Conda env: ${CONDA_ENV}"
echo "========================================="
