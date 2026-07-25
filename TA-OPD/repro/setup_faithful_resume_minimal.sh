#!/usr/bin/env bash
# 续跑 setup_faithful.sh: 跳过 transformer_engine/apex 等非必需包
#
# 原因:
#   - conda cuda dev headers (libcusparse-dev 等) 在 nvidia/label/cuda-12.9.1 和
#     conda-forge 都解算失败, 导致 transformer_engine 编不出来
#   - 但 smoke test 用 --transformer-impl local, 完全不用 TE
#   - apex / flash-linear-attention / torch_memory_saver / nvidia-modelopt
#     也不是 smoke test 必需
#
# 本脚本装 smoke test 真正需要的:
#   mbridge (必需, checkpoint 转换)
#   Megatron-Bridge (可选但论文有用)
#   Megatron-LM @ MEGATRON_COMMIT (必需)
#   slime + apply patches (必需)
#   numpy<2 + nvidia-cudnn-cu12 pin (必需)
#
# 用法: bash repro/setup_faithful_resume_minimal.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_DIR="${BASE_DIR:-${HOME}/taopd-faithful}"
MICROMAMBA_ROOT="${MICROMAMBA_ROOT:-${HOME}/micromamba}"
ENV_NAME="${ENV_NAME:-ta_opd_faithful}"
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

# ── [R1] mbridge (checkpoint 转换必需) ────────────────────────────────
echo ""
echo "=== [R1] mbridge ==="
pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps 2>&1 | tail -3
python3 -c "import mbridge; print(f'  ✅ mbridge {getattr(mbridge, \"__version__\", \"?\")}')"

# ── [R2] 跳过 transformer_engine / apex / flash-linear-attention / torch_memory_saver / nvidia-modelopt ─
echo ""
echo "=== [R2] 跳过 (非 smoke test 必需) ==="
echo "  - transformer_engine[pytorch]==2.10.0  ❌ 需要 cusparse.h (conda 装不上)"
echo "  - apex (NVIDIA git)                    ❌ 同上"
echo "  - flash-linear-attention==0.4.1        ❌ (非 Qwen3 dense 必需)"
echo "  - torch_memory_saver                   ❌ (megatron patch 显式 disable)"
echo "  - nvidia-modelopt                      ❌ (非 smoke test 必需)"
echo "  (smoke test 用 --transformer-impl local + --no-persist-layer-norm)"

# ── [R3] Megatron-Bridge (rl 训练可能用) ────────────────────────────────
echo ""
echo "=== [R3] Megatron-Bridge (fzyzcjy dev_rl) ==="
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation 2>&1 | tail -3 \
  || echo "  ⚠ Megatron-Bridge 装不上 (非必需, 跳过)"

# ── [R4] sgl-router (slime requirements.txt 列了 sglang-router>=0.2.3) ──
echo ""
echo "=== [R4] sgl-router 0.3.2 ==="
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall 2>&1 | tail -3 \
  || echo "  ⚠ sgl-router 装不上 (非必需)"

# ── [R5] Megatron-LM (必需, 指定 commit) ────────────────────────────────
echo ""
echo "=== [R5] Megatron-LM @ ${MEGATRON_COMMIT} ==="
cd "${BASE_DIR}"
if [[ ! -d "${BASE_DIR}/Megatron-LM" ]]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd Megatron-LM
git fetch --all 2>/dev/null || true
git checkout "${MEGATRON_COMMIT}" 2>&1 | tail -3
pip install -e . 2>&1 | tail -3

# ── [R6] slime + cudnn pin + numpy pin ──────────────────────────────────
echo ""
echo "=== [R6] slime + cudnn pin + numpy pin ==="
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

# ── [R7] Apply patches ─────────────────────────────────────────────────
echo ""
echo "=== [R7] Apply slime v0.5.9 patches ==="
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

# ── [R8] 验证 (HARD 只列 smoke test 必需包) ────────────────────────────
echo ""
echo "=== [R8] 最终验证 ==="
export PYTHONPATH="${BASE_DIR}/Megatron-LM:${SLIME_DIR}:${PYTHONPATH:-}"
python3 -c "
import sys
hard = [
    'torch', 'sglang', 'sgl_kernel', 'flashinfer', 'flash_attn',
    'ray', 'transformers', 'datasets', 'safetensors',
    'megatron', 'mbridge', 'numpy',
]
soft = [
    'transformer_engine', 'apex', 'flash_linear_attention',
    'torch_memory_saver', 'nvidia_modelopt', 'sglang_router',
    'cuda',
]
h_ok = h_fail = s_ok = s_fail = 0
for mod in hard:
    try:
        m = __import__(mod)
        print(f'  ✅ [HARD] {mod:25s} = {getattr(m, \"__version__\", \"?\")}')
        h_ok += 1
    except Exception as e:
        print(f'  ❌ [HARD] {mod:25s}: {str(e).splitlines()[0]}')
        h_fail += 1
for mod in soft:
    try:
        m = __import__(mod)
        print(f'  ✅ [SOFT] {mod:25s} = {getattr(m, \"__version__\", \"?\")}')
        s_ok += 1
    except Exception as e:
        print(f'  ⚠ [SOFT] {mod:25s}: {str(e).splitlines()[0]}')
        s_fail += 1
print()
print(f'HARD: {h_ok} ok / {h_fail} fail')
print(f'SOFT: {s_ok} ok / {s_fail} fail (允许)')
if h_fail > 0:
    print('❌ HARD 缺')
    sys.exit(1)
print('✅ smoke test 必需包全部 OK')
"

echo ""
echo "========================================="
echo " Faithful build (minimal) 完成"
echo ""
echo " 下一步:"
echo "   bash repro/run_apex_faithful.sh"
echo "========================================="
