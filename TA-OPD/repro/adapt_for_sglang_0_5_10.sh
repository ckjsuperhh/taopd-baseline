#!/usr/bin/env bash
# 把 repro/ 下所有脚本适配到 sglang 0.5.10.post1 + torch 2.9 (全栈新版本)
#
# 背景:
#   用户已在 Inspur 上装了 sglang 0.5.10.post1 (原生支持 Qwen3),
#   torch 2.9.1, flashinfer 0.6.7, sglang-kernel 0.4.1, flash-attn 4.0.0b23。
#   这套栈和之前为 sglang 0.4.1 写的 repro/*.sh 有 3 处不兼容:
#     1) --dist-init-addr 127.0.0.1:PORT → 应该用 --nccl-port PORT (0.5.10 支持)
#     2) 05/06/07 开头调了 patch_qwen3_as_qwen2.sh / patch_sglang_qwen2_for_qwen3.sh
#        → 0.5.10 原生支持 Qwen3, 不需要这些 monkey-patch
#     3) 旧 fix_sglang_deps.sh 不再需要 (pip 已经把依赖装全了)
#
# 本脚本做:
#   - sed 改 --dist-init-addr "127.0.0.1:${PORT_VAR}" → --nccl-port "${PORT_VAR}"
#   - 注释掉 patch_qwen3_as_qwen2.sh / patch_sglang_qwen2_for_qwen3.sh 调用
#   - 在每个改动文件顶部加标记 # TAOPD_ADAPT_V1 (幂等)
#   - 备份到 .pre_adapt (幂等: 已适配则跳过, 想还原 mv .pre_adapt 回去)
#
# 用法: bash repro/adapt_for_sglang_0_5_10.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo " 适配 sglang 0.5.10 + torch 2.9 (全栈新)"
echo "========================================="

# ── helper: 改一个文件 ─────────────────────────────────────────
adapt_one() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "  ⏭  不存在: $f"
    return 0
  fi
  if grep -q "TAOPD_ADAPT_V1" "$f"; then
    echo "  ⏭  已适配: $f"
    return 0
  fi

  # 备份
  cp "$f" "${f}.pre_adapt"

  # 1) --dist-init-addr "127.0.0.1:${XXX}" → --nccl-port "${XXX}"
  #    注意: 之前改成 --dist-init-addr 时, 端口变量名各不相同
  #    (nccl_port / TEACHER_NCCL_PORT), 这里用通用 regex
  sed -i -E 's|--dist-init-addr "127\.0\.0\.1:\$\{([A-Za-z_]+)\}"|--nccl-port "\${\1}"|g' "$f"

  # 2) 注释掉 patch_qwen3_as_qwen2.sh 调用 (整行)
  sed -i 's|^\(\s*\)bash "\${SCRIPT_DIR}/patch_qwen3_as_qwen2\.sh"|\1# [TAOPD_ADAPT_V1] sglang 0.5.10 原生支持 Qwen3, 不再需要 Qwen3→Qwen2 伪装 patch\n\1# bash "\${SCRIPT_DIR}/patch_qwen3_as_qwen2.sh"|g' "$f"

  # 3) 注释掉 patch_sglang_qwen2_for_qwen3.sh 调用
  sed -i 's|^\(\s*\)bash "\${SCRIPT_DIR}/patch_sglang_qwen2_for_qwen3\.sh"|\1# [TAOPD_ADAPT_V1] sglang 0.5.10 原生支持 Qwen3 q_norm/k_norm, 不再 monkey-patch\n\1# bash "\${SCRIPT_DIR}/patch_sglang_qwen2_for_qwen3.sh"|g' "$f"

  # 4) 顶部加标记 (幂等标识, 下次跑跳过)
  sed -i '2i # TAOPD_ADAPT_V1: adapted for sglang 0.5.10 + torch 2.9 (native Qwen3)' "$f"

  echo "  ✅ 适配: $f  (备份: ${f}.pre_adapt)"
}

# ── 适配 3 个含 teacher server launch 的脚本 ───────────────────
echo ""
echo "=== [1/2] 适配 smoke test / training / diagnostics ==="
adapt_one "${SCRIPT_DIR}/05_smoke_test.sh"
adapt_one "${SCRIPT_DIR}/06_run_all_training.sh"
adapt_one "${SCRIPT_DIR}/07_run_diagnostics.sh"

# ── 检查适配结果 ──────────────────────────────────────────────
echo ""
echo "=== [2/2] 验证适配结果 ==="
for f in 05_smoke_test.sh 06_run_all_training.sh 07_run_diagnostics.sh; do
  P="${SCRIPT_DIR}/${f}"
  echo "--- $f ---"
  echo "  --nccl-port 出现次数: $(grep -c '\-\-nccl-port' "$P" || echo 0)"
  echo "  --dist-init-addr 出现次数: $(grep -c '\-\-dist-init-addr' "$P" || echo 0)"
  echo "  patch_qwen3 调用 (未注释): $(grep -E '^\s*bash.*patch_qwen3' "$P" | wc -l)"
  echo "  patch_sglang 调用 (未注释): $(grep -E '^\s*bash.*patch_sglang_qwen2' "$P" | wc -l)"
done

# ── 列出可以删除的旧 patch 脚本 (可选) ───────────────────────
echo ""
echo "=== 不再需要的脚本 (可手动删除) ==="
for OLD in patch_qwen3_as_qwen2.sh patch_sglang_qwen2_for_qwen3.sh fix_sglang_deps.sh; do
  if [[ -f "${SCRIPT_DIR}/${OLD}" ]]; then
    echo "  📄 ${OLD}  (sglang 0.5.10 原生解决这些问题)"
  fi
done
echo "  # 删除命令 (可选):"
echo "  #   rm repro/patch_qwen3_as_qwen2.sh repro/patch_sglang_qwen2_for_qwen3.sh repro/fix_sglang_deps.sh"

echo ""
echo "========================================="
echo " 适配完成"
echo ""
echo " 下一步:"
echo "   bash repro/run_all.sh 3    # 先跑 student 转换 (Megatron)"
echo "   bash repro/run_all.sh 5    # 再跑 smoke test"
echo ""
echo " 还原 (如果需要):"
echo "   for f in repro/05_smoke_test.sh repro/06_run_all_training.sh repro/07_run_diagnostics.sh; do"
echo "     mv \"\${f}.pre_adapt\" \"\$f\""
echo "   done"
echo "========================================="
