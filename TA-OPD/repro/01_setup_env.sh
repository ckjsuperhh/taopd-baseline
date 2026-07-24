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
  echo "  ✅ env '${CONDA_ENV}' 已存在，跳过创建 (若要重建: conda env remove -n ${CONDA_ENV})"
else
  echo "  创建 env '${CONDA_ENV}'..."
  conda create -n "${CONDA_ENV}" python=3.10 pip -y
fi
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
# 已验证可行的组合 (2026-07 Inspur):
#   sglang==0.4.1 + sgl-kernel==0.1.0 + torch==2.5.1+cu124
# 注意: sgl-kernel 不能用最新版 (0.3.21 给 torch 2.7+ 编的, c10 ABI 不匹配)
FLASHINFER_INDEX="https://flashinfer.ai/whl/cu124/torch2.5/"

pip install "sglang==0.4.1" || { echo "  ❌ sglang install failed"; false; }
pip install "sgl-kernel==0.1.0" --extra-index-url "${FLASHINFER_INDEX}" \
  || pip install "sgl-kernel==0.1.0" \
  || { echo "  ❌ sgl-kernel==0.1.0 装不上 (rollout 引擎必需)"; false; }

# bare sglang 不带 [all] extras 的运行时依赖 (teacher/rollout SGLang server 要用)
pip install \
  orjson fastapi uvicorn uvloop pydantic msgspec python-multipart \
  hf_transfer decord soundfile pillow requests aiohttp psutil

# sglang 可能拉高 torch，回退
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall --no-deps

# flash-attn 编译非常慢且容易失败；允许失败（训练时如果不用 flash-attn，Megatron 会 fallback）
MAX_JOBS=4 pip install flash-attn --no-build-isolation || echo "  ⚠ flash-attn failed (non-fatal; Megatron can fall back)"

pip install "ray[default]>=2.9"
pip install transformers datasets accelerate safetensors
pip install pyarrow pandas "numpy<2" scipy scikit-learn
pip install matplotlib seaborn
pip install einops tiktoken sentencepiece protobuf
pip install wandb tensorboard
pip install lm-eval>=0.4.5
pip install mbridge

# 最终检查 torch 没被升级
CURRENT="$(python3 -c 'import torch; print(torch.__version__)')"
if [[ ! "${CURRENT}" == 2.5.* ]]; then
  echo "  WARNING: torch 被升级到 ${CURRENT}，回退中..."
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
fi

# ── 5. Megatron-LM + DCP patch ──────────────────────────────────────────
echo "[5/6] Megatron-LM..."
if [[ ! -d "${MEGATRON_LM_DIR}/megatron" ]]; then
  echo "  Cloning Megatron-LM to ${MEGATRON_LM_DIR}..."
  rm -rf "${MEGATRON_LM_DIR}"
  git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_LM_DIR}" \
    || { echo "  ❌ Megatron-LM clone failed (GitHub 不可达?)"; false; }
fi
if [[ ! -d "${MEGATRON_LM_DIR}/megatron" ]]; then
  echo "  ❌ ${MEGATRON_LM_DIR}/megatron 不存在，clone 失败"
  false
fi
echo "  Megatron-LM at ${MEGATRON_LM_DIR}"

cd "${MEGATRON_LM_DIR}"
PATCH="${REPO_ROOT}/docs/megatron_apex_patches.patch"
if [[ -f "${PATCH}" ]] && git apply --check "${PATCH}" 2>/dev/null; then
  git apply "${PATCH}"
  echo "  Applied DCP patches"
else
  echo "  (no patch to apply or already applied)"
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
# 验证时必须把 Megatron-LM 和 slime_ta_opd 加入 PYTHONPATH，否则 import megatron 会失败
export PYTHONPATH="${MEGATRON_LM_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
echo "  PYTHONPATH=${PYTHONPATH}"
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
    # 用 pip show 取版本最可靠（sglang 本体不一定暴露 __version__）
    import subprocess as _sp
    _r = _sp.run(['pip', 'show', 'sglang'], capture_output=True, text=True)
    _sv = 'unknown'
    for _line in (_r.stdout or '').splitlines():
        if _line.startswith('Version:'):
            _sv = _line.split(':', 1)[1].strip(); break
    # 检查 sgl-kernel 是否在（[all] extras 的核心）
    _has_kernel = False
    try:
        import sgl_kernel; _has_kernel = True
    except Exception:
        pass
    _mark = chr(9989) if _has_kernel else chr(9888)
    _kernel_msg = 'ok' if _has_kernel else 'MISSING'
    print(f'  {_mark} sglang {_sv} (sgl_kernel={_kernel_msg})')
    if not _has_kernel:
        print('    ⚠ sgl_kernel 缺失: rollout 不可用。')
        print('    原因: pip install sglang[all]==0.4.1 因 flashinfer==0.1.6 (yanked) 失败。')
        print('    建议单独装: pip install sgl-kernel==0.4.1 --find-links https://flashinfer.ai/whl/cu124/torch2.5/')
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
    import os
    mp = os.path.dirname(megatron.__file__)
    print(f'  ✅ megatron ({mp})')
except Exception as e:
    print(f'  ❌ megatron ({e})')
"
echo ""
echo "========================================="
echo " Done! conda env: ${CONDA_ENV}"
echo " CUDA_HOME: ${CUDA_HOME}"
echo "========================================="
