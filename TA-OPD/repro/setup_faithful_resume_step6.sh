#!/usr/bin/env bash
# 续跑 setup_faithful.sh 的 step 6-9 (step 1-5 已成功完成)
#
# 原因: step 6 transformer_engine 编译失败 (cusparse.h 缺失)
# 修法: 补装全套 CUDA dev headers, 然后跑 step 6-9
#
# 用法: bash repro/setup_faithful_resume_step6.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_DIR="${BASE_DIR:-${HOME}/taopd-faithful}"
MICROMAMBA_ROOT="${MICROMAMBA_ROOT:-${HOME}/micromamba}"
ENV_NAME="${ENV_NAME:-ta_opd_faithful}"
export SGLANG_COMMIT="bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"
export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

# micromamba hook
MM="${HOME}/.local/bin/micromamba"
[[ -x "${MM}" ]] || { echo "❌ micromamba 不在 ${MM}"; exit 1; }
export MAMBA_EXE="${MM}"
export MAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT}"
micromamba() { "${MAMBA_EXE}" "$@"; }
export -f micromamba

eval "$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}")"
micromamba activate "${ENV_NAME}"
export CUDA_HOME="$CONDA_PREFIX"
echo "✅ activated: ${ENV_NAME}"

# ── 补装 CUDA dev headers ──────────────────────────────────────────────
echo ""
echo "=== [R1] 补装 CUDA dev headers ==="
CUDA_CHANNEL="nvidia/label/cuda-12.9.1"
if [[ ! -f "${CONDA_PREFIX}/include/cusparse.h" ]]; then
  micromamba install -n "${ENV_NAME}" -c "${CUDA_CHANNEL}" -y \
    libcusparse-dev libcublas-dev libcufft-dev libcurand-dev libcusolver-dev \
    cuda-nvrtc-dev cuda-cccl \
    || {
      echo "⚠ ${CUDA_CHANNEL} 失败, 试 conda-forge"
      micromamba install -n "${ENV_NAME}" -c conda-forge -y \
        cusparse cusolver cublas cufft curand
    }
fi
ls "${CONDA_PREFIX}/include/cusparse.h" "${CONDA_PREFIX}/include/cublas_v2.h" 2>&1 | head
ls "${CONDA_PREFIX}/targets/x86_64-linux/include/cusparse.h" 2>&1 | head

# 确保 include path 能找到 (有些 conda 包装在 targets/x86_64-linux/include)
if [[ ! -f "${CONDA_PREFIX}/include/cusparse.h" ]] && \
   [[ -f "${CONDA_PREFIX}/targets/x86_64-linux/include/cusparse.h" ]]; then
  export C_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${C_INCLUDE_PATH:-}"
  export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
  echo "  set C_INCLUDE_PATH/CPLUS_INCLUDE_PATH"
fi

# ── Step 6: mbridge / transformer_engine / flash-linear-attention / apex ─
echo ""
echo "=== [6/9] mbridge / transformer_engine / flash-linear-attention / apex ==="
pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps 2>&1 | tail -3
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0" 2>&1 | tail -5
pip install flash-linear-attention==0.4.1 2>&1 | tail -3
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4 2>&1 | tail -5

# ── Step 7-9: 同 setup_faithful.sh ───────────────────────────────────────
echo ""
echo "=== [7/9] torch_memory_saver / Megatron-Bridge / modelopt / sgl-router ==="
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall 2>&1 | tail -3
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation 2>&1 | tail -3
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation 2>&1 | tail -3
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall 2>&1 | tail -3

echo ""
echo "=== [8/9] Megatron-LM @ ${MEGATRON_COMMIT} ==="
cd "${BASE_DIR}"
if [[ ! -d "${BASE_DIR}/Megatron-LM" ]]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd Megatron-LM
git fetch --all 2>/dev/null || true
git checkout "${MEGATRON_COMMIT}"
pip install -e . 2>&1 | tail -3

echo ""
echo "=== [9/9] slime + cudnn pin + numpy pin + apply patches ==="
if [[ ! -d "${BASE_DIR}/slime" ]]; then
  if [[ -d "${REPO_ROOT}/slime_ta_opd" ]]; then
    ln -sfn "${REPO_ROOT}/slime_ta_opd" "${BASE_DIR}/slime"
  else
    cd "${BASE_DIR}"
    git clone https://github.com/THUDM/slime.git
  fi
fi
export SLIME_DIR="${BASE_DIR}/slime"
cd "${SLIME_DIR}"
pip install -e . 2>&1 | tail -3

pip install nvidia-cudnn-cu12==9.16.0.29 2>&1 | tail -2
pip install "numpy<2" 2>&1 | tail -2

echo ""
echo "=== Apply patches ==="
cd "${BASE_DIR}/sglang"
if ! git apply --check "${SLIME_DIR}/docker/patch/v0.5.9/sglang.patch" 2>/dev/null; then
  echo "⚠ sglang.patch 已 apply 或冲突, 跳过"
else
  git apply "${SLIME_DIR}/docker/patch/v0.5.9/sglang.patch"
  echo "✅ sglang.patch applied"
fi

cd "${BASE_DIR}/Megatron-LM"
if ! git apply --check "${SLIME_DIR}/docker/patch/v0.5.9/megatron.patch" 2>/dev/null; then
  echo "⚠ megatron.patch 已 apply 或冲突, 跳过"
else
  git apply "${SLIME_DIR}/docker/patch/v0.5.9/megatron.patch"
  echo "✅ megatron.patch applied"
fi

echo ""
echo "=== 最终验证 ==="
export PYTHONPATH="${BASE_DIR}/Megatron-LM:${SLIME_DIR}:${PYTHONPATH:-}"
python3 -c "
import sys
checks = [
    ('torch', 'HARD'),
    ('sglang', 'HARD'),
    ('sgl_kernel', 'HARD'),
    ('flashinfer', 'HARD'),
    ('flash_attn', 'HARD'),
    ('ray', 'HARD'),
    ('transformers', 'HARD'),
    ('datasets', 'HARD'),
    ('safetensors', 'HARD'),
    ('megatron', 'HARD'),
    ('mbridge', 'HARD'),
    ('transformer_engine', 'HARD'),
    ('apex', 'HARD'),
    ('flash_linear_attention', 'HARD'),
    ('torch_memory_saver', 'HARD'),
    ('nvidia_modelopt', 'HARD'),
    ('sglang_router', 'HARD'),
    ('numpy', 'HARD'),
    ('cuda', 'HARD'),
]
n_hard_ok = n_hard_fail = 0
for mod, kind in checks:
    try:
        m = __import__(mod)
        v = getattr(m, '__version__', '?')
        print(f'  ✅ [{kind:4s}] {mod:25s} = {v}')
        n_hard_ok += 1
    except Exception as e:
        print(f'  ❌ [{kind:4s}] {mod:25s}: {str(e).splitlines()[0]}')
        n_hard_fail += 1
print()
print(f'HARD: {n_hard_ok} ok / {n_hard_fail} fail')
if n_hard_fail > 0:
    print('❌ 有 HARD 包缺失')
    sys.exit(1)
print('✅ 全部 OK')
"
