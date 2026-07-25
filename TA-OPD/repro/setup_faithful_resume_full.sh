#!/usr/bin/env bash
# 续跑 setup_faithful step 6-9, 100% faithful (含 transformer_engine + apex)
#
# 根因: transformer_engine 编译时找不到 cusparse.h
# 修法: torch wheel 自带全套 NVIDIA headers (nvidia/cusparse/include 等),
#       设 C_INCLUDE_PATH + LIBRARY_PATH 让 nvcc/linker 能找到,
#       然后跑完整 step 6-9 (含 TE + apex + flash-linear-attention + ...)
#
# 这是 100% faithful 路线, 不删任何官方依赖
#
# 用法: bash repro/setup_faithful_resume_full.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_DIR="${BASE_DIR:-${HOME}/taopd-faithful}"
MICROMAMBA_ROOT="${MICROMAMBA_ROOT:-${HOME}/micromamba}"
ENV_NAME="${ENV_NAME:-ta_opd_faithful}"
export SGLANG_COMMIT="bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"
export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

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

# ── [R0] 设 NVIDIA header / lib paths (torch wheel 自带) ─────────────
echo ""
echo "=== [R0] NVIDIA headers from torch wheel ==="
NVIDIA_PKG="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia"
if [[ ! -d "${NVIDIA_PKG}" ]]; then
  echo "❌ ${NVIDIA_PKG} 不存在"
  exit 1
fi
# 关键 headers
for pkg in cusparse cublas cufft cusolver curand cudnn nvtx cuda_nvrtc; do
  if [[ -d "${NVIDIA_PKG}/${pkg}/include" ]]; then
    echo "  ✅ ${pkg}/include ($(ls ${NVIDIA_PKG}/${pkg}/include/*.h 2>/dev/null | wc -l) headers)"
  fi
done
# 关键 libs
for pkg in cusparse cublas cufft cusolver curand cudnn cuda_runtime nvtx cuda_nvrtc; do
  if [[ -d "${NVIDIA_PKG}/${pkg}/lib" ]]; then
    echo "  ✅ ${pkg}/lib ($(ls ${NVIDIA_PKG}/${pkg}/lib/lib*.so* 2>/dev/null | wc -l) libs)"
  fi
done

# 收集所有 include / lib 路径
ALL_INC=""
ALL_LIB=""
for pkg in cusparse cublas cufft cusolver curand cudnn cuda_runtime nvtx cuda_nvrtc; do
  if [[ -d "${NVIDIA_PKG}/${pkg}/include" ]]; then
    ALL_INC="${ALL_INC}:${NVIDIA_PKG}/${pkg}/include"
  fi
  if [[ -d "${NVIDIA_PKG}/${pkg}/lib" ]]; then
    ALL_LIB="${ALL_LIB}:${NVIDIA_PKG}/${pkg}/lib"
  fi
done
export C_INCLUDE_PATH="${ALL_INC#:}:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${ALL_INC#:}:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${ALL_LIB#:}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${ALL_LIB#:}:${LD_LIBRARY_PATH:-}"
echo "  C_INCLUDE_PATH=${C_INCLUDE_PATH}"
echo "  LIBRARY_PATH=${LIBRARY_PATH}"

# 验证 cusparse.h 能找到
echo '#include <cusparse.h>
int main(){return 0;}' > /tmp/test_cusparse.c
${CONDA_PREFIX}/bin/nvcc -c /tmp/test_cusparse.c -o /tmp/test_cusparse.o 2>&1 && echo "✅ nvcc 能找到 cusparse.h" || { echo "❌ 还是找不到"; exit 1; }
rm -f /tmp/test_cusparse.c /tmp/test_cusparse.o

# ── [R1] Step 6: mbridge + transformer_engine + flash-linear-attention + apex
echo ""
echo "=== [6/9] mbridge / transformer_engine / flash-linear-attention / apex ==="
pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps 2>&1 | tail -3
echo "--- transformer_engine (预计 30-45min) ---"
# 强制源码编译: 中国网络无法下载 GitHub Releases 上的预编译 wheel
export NVTE_PYTORCH_FORCE_BUILD=TRUE
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
echo "--- flash-linear-attention ---"
pip install flash-linear-attention==0.4.1 2>&1 | tail -3
echo "--- apex (预计 30min) ---"
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

# ── [7-9] 同 minimal 脚本 ────────────────────────────────────────────────
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
    echo "  slime → ${REPO_ROOT}/slime_ta_opd (软链)"
  else
    cd "${BASE_DIR}"
    git clone https://github.com/THUDM/slime.git
  fi
fi
export SLIME_DIR="${BASE_DIR}/slime"
cd "${SLIME_DIR}"
pip install -e . 2>&1 | tail -5

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

# ── 最终验证 ──────────────────────────────────────────────────────────────
echo ""
echo "=== 最终验证 (100% faithful) ==="
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
print('✅ 全部 OK (100% faithful)')
"

echo ""
echo "========================================="
echo " 🎉 Faithful build (FULL) 完成"
echo ""
echo " 下一步:"
echo "   bash repro/run_apex_faithful.sh"
echo "========================================="
