#!/usr/bin/env bash
# 装 sglang 裸包漏掉的依赖 (orjson 等)
# 用法: bash repro/fix_sglang_deps.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

echo "=== 装 sglang[all] 通常会带上的常见依赖 ==="
# sglang==0.4.1 bare 不带这些, 但 server / rollout 都要
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
  psutil

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
