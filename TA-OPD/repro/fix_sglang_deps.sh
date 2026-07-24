#!/usr/bin/env bash
# 装 sglang 裸包漏掉的依赖 (orjson 等)
# 用法: bash repro/fix_sglang_deps.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

echo "=== 装 sglang[all] 通常会带上的常见依赖 ==="
# sglang==0.4.1 bare 不带这些, 但 server / rollout 都要
# 一次性补齐, 不再一个个打地鼠

# pyairports (outlines 传递依赖) 单独 force 一次, 避免之前跑过旧版脚本漏掉
pip install --force-reinstall --no-deps pyairports 2>/dev/null || pip install pyairports

pip install \
  orjson \
  fastapi \
  uvicorn \
  uvloop \
  pydantic \
  msgspec \
  python-multipart \
  hf_transfer \
  decord \
  soundfile \
  pillow \
  requests \
  aiohttp \
  psutil \
  pyzmq \
  outlines \
  prometheus_client \
  setproctitle \
  diskcache \
  cloudpickle \
  tiktoken \
  numba \
  coloredlogs \
  packaging \
  sentencepiece \
  protobuf \
  nvidia-ml-py \
  openai \
  pyairports

# flashinfer: sglang 0.4.1 的 attention backend。
#   - 默认 PyPI 的 flashinfer==0.1.6 已被 yanked, 直接 pip install 会失败
#   - wheel 包名叫 flashinfer-python (不是 flashinfer), 走 flashinfer.ai 官方索引
#     cu124 + torch2.5 可用: 0.2.0.post2, 0.2.1.post1/2, 0.2.2(.post1), 0.2.3, 0.2.4, 0.2.5
echo ""
echo "=== 装 flashinfer-python (从 flashinfer.ai wheel 索引) ==="
FLASHINFER_INDEX="https://flashinfer.ai/whl/cu124/torch2.5/"
FLASHINFER_OK=0
for V in "0.2.1.post2" "0.2.1.post1" "0.2.2.post1" "0.2.2" "0.2.0.post2" "0.2.3" "0.2.4" "0.2.5"; do
  echo "--- 试 flashinfer-python==${V} ---"
  if pip install "flashinfer-python==${V}" --extra-index-url "${FLASHINFER_INDEX}" 2>&1 | tail -5; then
    FLASHINFER_OK=1; break
  fi
done
if [[ "${FLASHINFER_OK}" -eq 0 ]]; then
  echo "⚠ 钉版本失败, 试最新 flashinfer-python (从 flashinfer.ai 索引)..."
  if pip install flashinfer-python --extra-index-url "${FLASHINFER_INDEX}" 2>&1 | tail -5; then
    FLASHINFER_OK=1
  fi
fi
if [[ "${FLASHINFER_OK}" -eq 0 ]]; then
  echo "⚠ flashinfer.ai 索引也失败, 试 PyPI 上的 flashinfer-python (latest)..."
  pip install flashinfer-python 2>&1 | tail -5 \
    || echo "  ❌ flashinfer 装不上 (sglang 可 fallback 到 triton, 不一定致命)"
fi

# vllm: sglang server 依赖但 vllm 可能拉高 torch。策略:
#   - 钉 vllm 到已知与 torch 2.5.1+cu124 兼容的版本
#   - 装完回退 torch 到 2.5.1
echo ""
echo "=== 装 vllm (保持 torch 2.5.1+cu124) ==="
VLLM_OK=0
for V in "0.6.3.post1" "0.6.3" "0.6.2" "0.6.1.post2"; do
  echo "--- 试 vllm==${V} ---"
  if pip install "vllm==${V}" 2>&1 | tail -5; then
    VLLM_OK=1; break
  fi
done
if [[ "${VLLM_OK}" -eq 0 ]]; then
  echo "⚠ 已知版本都失败, 试最新 vllm..."
  pip install vllm || echo "  ❌ vllm 装不上"
fi

# 强制回退 torch (vllm 经常拉高)
echo "--- 回退 torch 到 2.5.1+cu124 ---"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall --no-deps

echo ""
echo "=== 测试 sglang.srt.server 关键 import ==="
python3 -c "
for mod in [
    'sglang',
    'sglang.srt',
    'sglang.srt.server',
    'sglang.launch_server',
    'orjson',
    'fastapi',
    'uvicorn',
    'uvloop',
    'zmq',
    'outlines',
    'prometheus_client',
    'setproctitle',
    'vllm',
    'flashinfer',
    'pyairports',
]:
    try:
        __import__(mod)
        print(f'  ✅ {mod}')
    except Exception as e:
        print(f'  ❌ {mod}: {type(e).__name__}: {e}')
"

echo ""
echo "========================================="
echo " 继续: bash repro/run_all.sh 5"
echo "========================================="
