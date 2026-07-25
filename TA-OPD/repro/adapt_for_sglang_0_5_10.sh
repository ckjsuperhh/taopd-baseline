#!/usr/bin/env bash
# 把 repro/ 下所有脚本适配到 sglang 0.5.10+ (全栈新版本)
# 自动检测 sglang 版本, 决定用 --nccl-port 还是 --dist-init-addr
#
# 背景:
#   - sglang 0.5.10+ 支持 --nccl-port, 原生支持 Qwen3 (含 q_norm/k_norm)
#   - sglang 0.4.1 只有 --dist-init-addr, 不认 Qwen3 (需要 monkey-patch)
#   - 本脚本自动检测, 按版本选参数; 0.4.1 会警告但仍适配 CLI
#
# 用法: bash repro/adapt_for_sglang_0_5_10.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo " 适配 sglang (自动检测版本)"
echo "========================================="

# ── 检测 sglang 版本 ──────────────────────────────────────────
SGLANG_VERSION="$(python3 -c "import sglang; print(getattr(sglang, '__version__', '0.0.0'))" 2>/dev/null || echo "0.0.0")"
echo "📊 当前 sglang: ${SGLANG_VERSION}"

# 比较版本: 0.5.x+ 用 --nccl-port, 0.4.x 用 --dist-init-addr
version_ge() {
  # $1 >= $2 ?
  [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

if version_ge "${SGLANG_VERSION}" "0.5.0"; then
  USE_NCCL_PORT=1
  NCCL_FLAG="--nccl-port"
  NEED_QWEN3_PATCH=0
  echo "✅ sglang 0.5.x+ detected: 用 --nccl-port, 不需要 Qwen3 monkey-patch"
else
  USE_NCCL_PORT=0
  NCCL_FLAG="--dist-init-addr 127.0.0.1:\${PORT_VAR}"
  NEED_QWEN3_PATCH=1
  echo "⚠️  sglang <0.5 detected (${SGLANG_VERSION}): 用 --dist-init-addr, Qwen3 需要 monkey-patch"
  echo "   强烈建议升级到 sglang 0.5.10.post1: pip install 'sglang==0.5.10.post1'"
fi

# ── helper: 改一个文件 ─────────────────────────────────────────
adapt_one() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "  ⏭  不存在: $f"
    return 0
  fi

  # 备份 (幂等: 有 .pre_adapt 就从它开始, 否则新建)
  if [[ -f "${f}.pre_adapt" ]]; then
    cp "${f}.pre_adapt" "$f"
  else
    cp "$f" "${f}.pre_adapt"
  fi

  if [[ "${USE_NCCL_PORT}" -eq 1 ]]; then
    # 模式 A: sglang 0.5.10+
    # 1a) --dist-init-addr "127.0.0.1:${XXX}" → --nccl-port "${XXX}"
    sed -i -E 's|--dist-init-addr "127\.0\.0\.1:\$\{([A-Za-z_]+)\}"|--nccl-port "\${\1}"|g' "$f"
    # 2a) 注释掉 monkey-patch 调用 (0.5.10 不需要)
    sed -i -E 's|^\s*bash "\$\{SCRIPT_DIR\}/patch_qwen3_as_qwen2\.sh"|# &|g' "$f"
    sed -i -E 's|^\s*bash "\$\{SCRIPT_DIR\}/patch_sglang_qwen2_for_qwen3\.sh"|# &|g' "$f"
  else
    # 模式 B: sglang 0.4.1
    # 1b) --nccl-port "${XXX}" → --dist-init-addr "127.0.0.1:${XXX}"
    sed -i -E 's|--nccl-port "\$\{([A-Za-z_]+)\}"|--dist-init-addr "127.0.0.1:\${\1}"|g' "$f"
    # 2b) 保持 monkey-patch 调用 (如果有的话)
  fi

  echo "  ✅ 适配: $f"
}

# ── 适配 3 个含 teacher server launch 的脚本 ───────────────────
echo ""
echo "=== 适配 smoke test / training / diagnostics ==="
adapt_one "${SCRIPT_DIR}/05_smoke_test.sh"
adapt_one "${SCRIPT_DIR}/06_run_all_training.sh"
adapt_one "${SCRIPT_DIR}/07_run_diagnostics.sh"

# ── 检查适配结果 ──────────────────────────────────────────────
echo ""
echo "=== 验证适配结果 ==="
for f in 05_smoke_test.sh 06_run_all_training.sh 07_run_diagnostics.sh; do
  P="${SCRIPT_DIR}/${f}"
  echo "--- $f ---"
  echo "  --nccl-port 出现次数:       $(grep -c '\-\-nccl-port' "$P" 2>/dev/null || echo 0)"
  echo "  --dist-init-addr 出现次数:  $(grep -c '\-\-dist-init-addr' "$P" 2>/dev/null || echo 0)"
  echo "  patch_qwen3 调用 (未注释):  $(grep -E '^[[:space:]]*bash.*patch_qwen3' "$P" 2>/dev/null | wc -l)"
  echo "  patch_sglang 调用 (未注释): $(grep -E '^[[:space:]]*bash.*patch_sglang_qwen2' "$P" 2>/dev/null | wc -l)"
done

# ── 列出可以删除的旧 patch 脚本 (0.5.10 下不需要) ───────────
if [[ "${NEED_QWEN3_PATCH}" -eq 0 ]]; then
  echo ""
  echo "=== 不再需要的脚本 (0.5.10 原生解决, 可手动删除) ==="
  for OLD in patch_qwen3_as_qwen2.sh patch_sglang_qwen2_for_qwen3.sh fix_sglang_deps.sh; do
    [[ -f "${SCRIPT_DIR}/${OLD}" ]] && echo "  📄 ${OLD}"
  done
fi

echo ""
echo "========================================="
echo " 适配完成 (sglang ${SGLANG_VERSION})"
echo ""
echo " 下一步:"
echo "   bash repro/run_all.sh 3    # 先跑 student conversion"
echo "   bash repro/run_all.sh 5    # 再跑 smoke test"
echo "========================================="
