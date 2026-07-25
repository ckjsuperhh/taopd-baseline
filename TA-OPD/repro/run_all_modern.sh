#!/usr/bin/env bash
# Modern pipeline 一键入口:
#   1) conda activate ta_opd_modern (如果 01b 已经建好 env)
#   2) 验证 sglang 0.5.10
#   3) 运行 adapt 脚本 (patch 05/06/07)
#   4) 跑 step 3 (student conversion) + step 5 (smoke test)
#
# 用法: bash repro/run_all_modern.sh
#       bash repro/run_all_modern.sh 5    # 从 step 5 开始
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

MODERN_ENV="${MODERN_ENV:-ta_opd_modern}"
START_STEP="${1:-3}"

echo "========================================="
echo " Modern pipeline (sglang 0.5.10 + torch 2.5.1+cu124)"
echo " env:         ${MODERN_ENV}"
echo " start_step:  ${START_STEP}"
echo "========================================="

# ── 1) 激活 modern env ─────────────────────────────────────────
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -xq "${MODERN_ENV}"; then
  echo "❌ modern env '${MODERN_ENV}' 不存在, 先跑:"
  echo "   bash repro/01b_setup_env_modern.sh"
  exit 1
fi
conda activate "${MODERN_ENV}"
echo "✅ activated: ${MODERN_ENV}"

# ── 2) 验证 sglang 版本 ──────────────────────────────────────
SGLANG_VERSION="$(python3 -c "import sglang; print(getattr(sglang, '__version__', '0.0.0'))" 2>/dev/null || echo "0.0.0")"
echo "✅ sglang version: ${SGLANG_VERSION}"
if [[ ! "${SGLANG_VERSION}" == 0.5.* ]] && [[ ! "${SGLANG_VERSION}" == 0.6.* ]]; then
  echo "❌ sglang 版本不对 (期望 0.5.x 或 0.6.x, 实际 ${SGLANG_VERSION})"
  echo "   请在 ${MODERN_ENV} env 里重装: pip install 'sglang==0.5.10.post1'"
  exit 1
fi

# ── 3) 跑 adapt 脚本 (patch 05/06/07) ─────────────────────────
echo ""
echo "=== Running adapt_for_sglang_0_5_10.sh ==="
bash "${SCRIPT_DIR}/adapt_for_sglang_0_5_10.sh"

# ── 4) 跑目标 steps ──────────────────────────────────────────
echo ""
echo "=== Running steps ${START_STEP}..9 ==="
bash "${SCRIPT_DIR}/run_all.sh" "${START_STEP}"
