#!/usr/bin/env bash
# Faithful reproduction of slime_ta_opd/build_conda.sh
#
# 100% 按官方脚本走, 不做任何版本回退/替换.
# 唯一变化:
#   - micromamba 装在 ${HOME}/micromamba (不动 /root)
#   - BASE_DIR 默认 ~/taopd-faithful (不动 /root)
#   - sglang/Megatron/slime clone 到 BASE_DIR 下
#
# 用法:
#   bash repro/setup_faithful.sh                 # 在 apex 服务器上跑
#   BASE_DIR=/data/xxx bash repro/setup_faithful.sh   # 换挂载点
#
# 预计耗时: 2-4 小时 (flash-attn + transformer_engine + apex 都是源码编)
# 预计磁盘: ~50 GB
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── 配置 ──────────────────────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-${HOME}/taopd-faithful}"
MICROMAMBA_ROOT="${MICROMAMBA_ROOT:-${HOME}/micromamba}"
ENV_NAME="${ENV_NAME:-ta_opd_faithful}"
export SGLANG_COMMIT="bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"
export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

echo "========================================="
echo " Faithful build (复刻 slime/build_conda.sh)"
echo " BASE_DIR         = ${BASE_DIR}"
echo " MICROMAMBA_ROOT  = ${MICROMAMBA_ROOT}"
echo " ENV_NAME         = ${ENV_NAME}"
echo " SGLANG_COMMIT    = ${SGLANG_COMMIT}"
echo " MEGATRON_COMMIT  = ${MEGATRON_COMMIT}"
echo "========================================="

# ── 预检: 磁盘空间 ──────────────────────────────────────────────────────
AVAIL_GB=$(df -BG "${BASE_DIR%/*}" 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
echo "可用磁盘: ${AVAIL_GB:-?} G (在 ${BASE_DIR%/*})"
if [[ -n "${AVAIL_GB}" ]] && [[ "${AVAIL_GB}" -lt 50 ]]; then
  echo "⚠ 磁盘 < 50G, 大概率编到一半会失败"
  echo "  请设置 BASE_DIR 到一个有 ≥50G 的挂载点:"
  echo "    BASE_DIR=/data/xxx bash repro/setup_faithful.sh"
  read -p "  继续? [y/N] " -n 1 -r
  echo
  [[ ! "${REPLY}" =~ ^[Yy]$ ]] && exit 1
fi

# ── Step 1: micromamba (build_conda.sh 用的是 micromamba) ──────────────
echo ""
echo "=== [1/9] 安装 micromamba ==="
mkdir -p "${MICROMAMBA_ROOT}"
if [[ ! -x "${MICROMAMBA_ROOT}/bin/micromamba" ]]; then
  yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh) 2>&1 | tail -10
  # 装完 source 一下让当前 shell 能用
  if [[ -f "${HOME}/.bashrc" ]]; then source "${HOME}/.bashrc"; fi
fi
# 兜底: 如果 ~/.bashrc 没注入, 直接加到 PATH
if ! command -v micromamba >/dev/null 2>&1; then
  export PATH="${MICROMAMBA_ROOT}/bin:${PATH}"
fi
# 再兜底: apex 上 micromamba 默认装到 ~/.local/bin/micromamba, env 在 ~/micromamba/
if ! command -v micromamba >/dev/null 2>&1; then
  if [[ -x "${HOME}/.local/bin/micromamba" ]]; then
    export MAMBA_EXE="${HOME}/.local/bin/micromamba"
    export MAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT}"
    eval "$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}" 2>/dev/null)"
  fi
fi
# 再再兜底: source ~/.bashrc 让 micromamba shell init 生效
if ! command -v micromamba >/dev/null 2>&1; then
  [[ -f "${HOME}/.bashrc" ]] && source "${HOME}/.bashrc"
fi
command -v micromamba >/dev/null 2>&1 || { echo "❌ micromamba 装不上"; exit 1; }
micromamba --version

# ── Step 2: 创建 env + 装 CUDA 12.9 + cudnn ────────────────────────────
echo ""
echo "=== [2/9] conda env + CUDA 12.9 + cudnn ==="
if ! micromamba env list | awk '{print $1}' | grep -xq "${ENV_NAME}"; then
  micromamba create -n "${ENV_NAME}" python=3.12 pip -c conda-forge -y
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"
export CUDA_HOME="$CONDA_PREFIX"
echo "✅ activated: ${ENV_NAME}"
echo "   CUDA_HOME = ${CUDA_HOME}"

# 注: 整包 `cuda` 在 nvidia/label/cuda-12.9.1 channel 用 libmamba 解算会冲突 (__win marker)
# 只装 nvcc + headers (编译 flash-attn/TE/apex 必需), 其他 runtime libs 由 torch wheel 提供
micromamba install -n "${ENV_NAME}" cuda-nvcc cuda-cudart-dev cuda-nvtx cuda-nvtx-dev -c nvidia/label/cuda-12.9.1 -y \
  || micromamba install -n "${ENV_NAME}" cuda-nvcc cuda-cudart-dev -c nvidia/label/cuda-12.9.1 -y \
  || {
    echo "⚠ nvidia/label/cuda-12.9.1 解算失败, 试 conda-forge cuda-nvcc"
    micromamba install -n "${ENV_NAME}" cuda-nvcc cuda-cudart-dev cuda-nvtx -c conda-forge -y
  }
micromamba install -n "${ENV_NAME}" -c conda-forge cudnn -y

# ── Step 3: cuda-python + torch 2.9.1+cu129 ────────────────────────────
echo ""
echo "=== [3/9] cuda-python + torch 2.9.1+cu129 ==="
pip install cuda-python==12.9
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu129
python3 -c "
import torch
print(f'  torch {torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}')
print(f'  cuda available: {torch.cuda.is_available()}  device_count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'    GPU{i}: {torch.cuda.get_device_name(i)} (sm_{torch.cuda.get_device_capability(i)[0]}{torch.cuda.get_device_capability(i)[1]})')
"

# ── Step 4: sglang (从源码, 指定 commit) ────────────────────────────────
echo ""
echo "=== [4/9] sglang @ ${SGLANG_COMMIT} ==="
mkdir -p "${BASE_DIR}"
cd "${BASE_DIR}"
if [[ ! -d "${BASE_DIR}/sglang" ]]; then
  git clone https://github.com/sgl-project/sglang.git
fi
cd sglang
git fetch --all 2>/dev/null || true
git checkout "${SGLANG_COMMIT}"
pip install -e "python[all]"

# ── Step 5: flash-attn 2.7.4.post1 (源码编, Megatron 最高支持) ────────
echo ""
echo "=== [5/9] flash-attn 2.7.4.post1 (源码) ==="
pip install cmake ninja
MAX_JOBS="${MAX_JOBS:-8}" pip -v install flash-attn==2.7.4.post1 --no-build-isolation

# ── Step 6: mbridge + transformer_engine + flash-linear-attention + apex
echo ""
echo "=== [6/9] mbridge / transformer_engine / flash-linear-attention / apex ==="
pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install flash-linear-attention==0.4.1
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

# ── Step 7: torch_memory_saver / Megatron-Bridge / nvidia-modelopt / sgl-router
echo ""
echo "=== [7/9] torch_memory_saver / Megatron-Bridge / modelopt / sgl-router ==="
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall

# ── Step 8: Megatron-LM (指定 commit) ──────────────────────────────────
echo ""
echo "=== [8/9] Megatron-LM @ ${MEGATRON_COMMIT} ==="
cd "${BASE_DIR}"
if [[ ! -d "${BASE_DIR}/Megatron-LM" ]]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd Megatron-LM
git fetch --all 2>/dev/null || true
git checkout "${MEGATRON_COMMIT}"
pip install -e .

# ── Step 9: slime (用本地 repo, 不打 patch 先) + cudnn pin + numpy pin + apply patches
echo ""
echo "=== [9/9] slime + cudnn pin + numpy pin + apply patches ==="
# slime: 用本地 REPO_ROOT/slime_ta_opd 作为 slime 源码 (避免再 clone 一份)
# 但 build_conda.sh 期望 slime 是独立目录, 用软链做兼容
if [[ ! -d "${BASE_DIR}/slime" ]]; then
  if [[ -d "${REPO_ROOT}/slime_ta_opd" ]]; then
    ln -sfn "${REPO_ROOT}/slime_ta_opd" "${BASE_DIR}/slime"
    echo "  slime → ${REPO_ROOT}/slime_ta_opd (软链)"
  else
    cd "${BASE_DIR}"
    git clone https://github.com/THUDM/slime.git
  fi
fi
export SLIME_DIR="${BASE_DIR}/slime"
cd "${SLIME_DIR}"
pip install -e .

# cudnn + numpy pin (https://github.com/pytorch/pytorch/issues/168167)
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"

# Apply patches
echo ""
echo "=== Apply patches ==="
cd "${BASE_DIR}/sglang"
if ! git apply --check "${SLIME_DIR}/docker/patch/v0.5.9/sglang.patch" 2>/dev/null; then
  echo "⚠ sglang.patch 已经 apply 过或冲突, 跳过 (用 --check 失败)"
else
  git apply "${SLIME_DIR}/docker/patch/v0.5.9/sglang.patch"
  echo "✅ sglang.patch applied"
fi

cd "${BASE_DIR}/Megatron-LM"
if ! git apply --check "${SLIME_DIR}/docker/patch/v0.5.9/megatron.patch" 2>/dev/null; then
  echo "⚠ megatron.patch 已经 apply 过或冲突, 跳过"
else
  git apply "${SLIME_DIR}/docker/patch/v0.5.9/megatron.patch"
  echo "✅ megatron.patch applied"
fi

# ── 最终验证 ──────────────────────────────────────────────────────────────
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

echo ""
echo "========================================="
echo " Faithful build 完成"
echo ""
echo " 下一步:"
echo "   bash repro/run_smoke_test_faithful.sh"
echo "========================================="
