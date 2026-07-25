#!/usr/bin/env bash
# 给 sglang 0.4.1 的 Qwen2Attention 打 monkey-patch, 让它能加载 Qwen3 dense
# (Qwen3 多了 per-head RMSNorm: q_norm / k_norm)
#
# 策略:
#   - 把 _taopd_per_head_rmsnorm helper 写到独立的 _taopd_qwen3_patch.py
#     (和 qwen2.py 同目录, 直接 import 即可, 不嵌入多行字符串)
#   - 只编辑 qwen2.py 的 3 个具体位置, 用 sed 风格的精准行替换
#   - 幂等: 文件里找 # TAOPD_PATCH_V1 标记, 已打则跳过
#
# 用法: bash repro/patch_sglang_qwen2_for_qwen3.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

QWEN2_PY="$(python3 -c "import sglang.srt.models.qwen2 as m, os; print(os.path.abspath(m.__file__))")"
if [[ -z "$QWEN2_PY" ]] || [[ ! -f "$QWEN2_PY" ]]; then
  echo "❌ 找不到 sglang/srt/models/qwen2.py"; exit 1
fi
PATCH_DIR="$(dirname "${QWEN2_PY}")"
HELPER_PY="${PATCH_DIR}/_taopd_qwen3_patch.py"
echo "📄 qwen2.py: ${QWEN2_PY}"
echo "📄 helper:   ${HELPER_PY}"

# ── 1) 已经打过 patch 就跳过 ────────────────────────────────────
if grep -q "TAOPD_PATCH_V1" "$QWEN2_PY"; then
  echo "  ✅ 已打过 patch (TAOPD_PATCH_V1), 跳过"
  exit 0
fi

# ── 2) 如果之前打过坏 patch, 先从 .orig 还原 ───────────────────
if [[ -f "${QWEN2_PY}.orig" ]]; then
  echo "  ♻️  从 .orig 还原 (覆盖可能的坏 patch)"
  cp "${QWEN2_PY}.orig" "$QWEN2_PY"
else
  cp "$QWEN2_PY" "${QWEN2_PY}.orig"
  echo "  💾 备份到 ${QWEN2_PY}.orig"
fi

# ── 3) 写 helper 模块 (独立文件, 不嵌入 qwen2.py) ───────────────
cat > "$HELPER_PY" <<'PY'
"""TAOPD_PATCH_V1: Qwen3 per-head QK RMSNorm helper.

被 patch 后的 qwen2.py import 此模块, 在 Qwen2Attention.forward 里调用。
"""
import torch


def per_head_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对 x 的最后一维做 RMSNorm, weight shape 必须等于 head_dim.

    输入 shape: 任意, 最后一维是 head_dim. 典型:
        q, k: [num_tokens, num_heads, head_dim] 或 [num_tokens, num_heads * head_dim]
    如果是 2D, 调用方需先 reshape 成 3D 再传进来。
    """
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(orig_dtype)
PY
echo "  ✅ 写了 ${HELPER_PY}"

# ── 4) 用 Python 做精准行级编辑 (不嵌入多行字符串, 用 list join) ──
python3 - "$QWEN2_PY" <<'PY'
import sys, re
p = sys.argv[1]
lines = open(p).read().splitlines(keepends=True)

# ── helper: 在 "import" 块之后插入 import helper ──
# 找最后一个 import 行 (包括 from X import Y), 在它之后插入
last_import_idx = -1
for i, line in enumerate(lines):
    s = line.lstrip()
    if s.startswith("import ") or s.startswith("from "):
        last_import_idx = i
assert last_import_idx >= 0, "找不到任何 import 行"
inject_import = [
    "from ._taopd_qwen3_patch import per_head_rmsnorm as _taopd_per_head_rmsnorm  # TAOPD_PATCH_V1\n",
]
lines = lines[:last_import_idx+1] + inject_import + lines[last_import_idx+1:]

# ── helper: 找 self.rotary_emb 赋值所在的行 (Qwen2Attention.__init__ 里) ──
rotary_assign_idx = -1
for i, line in enumerate(lines):
    if "self.rotary_emb" in line and "=" in line and "self.rotary_emb(" not in line.replace(" ", ""):
        # 匹配 "self.rotary_emb = ..." 但不是 "self.rotary_emb(...)"
        if re.search(r"^\s*self\.rotary_emb\s*=", line):
            rotary_assign_idx = i
            break
assert rotary_assign_idx >= 0, "找不到 'self.rotary_emb = ...' 赋值 (Qwen2Attention.__init__)"
indent_init = re.match(r"^\s*", lines[rotary_assign_idx]).group(0)
inject_init = [
    f"{indent_init}# TAOPD_PATCH_V1: Qwen3 per-head QK RMSNorm (Qwen2 默认关闭, Qwen3 config 会开)\n",
    f"{indent_init}_hd = getattr(config, 'head_dim', None) or (config.hidden_size // config.num_attention_heads)\n",
    f"{indent_init}self.head_dim = _hd\n",
    f"{indent_init}if getattr(config, 'use_qk_norm', False):\n",
    f"{indent_init}    self.q_norm = torch.nn.Parameter(torch.ones(_hd))\n",
    f"{indent_init}    self.k_norm = torch.nn.Parameter(torch.ones(_hd))\n",
    f"{indent_init}    self._use_qk_norm = True\n",
    f"{indent_init}    self._qk_norm_eps = getattr(config, 'rms_norm_eps', 1e-6)\n",
    f"{indent_init}else:\n",
    f"{indent_init}    self._use_qk_norm = False\n",
]
lines = lines[:rotary_assign_idx+1] + inject_init + lines[rotary_assign_idx+1:]

# ── helper: 找 self.rotary_emb(positions, ...) 调用所在的行 (forward) ──
rotary_call_idx = -1
for i, line in enumerate(lines):
    if "self.rotary_emb(positions" in line:
        rotary_call_idx = i
        break
assert rotary_call_idx >= 0, "找不到 'self.rotary_emb(positions, ...)' 调用 (forward)"
indent_fwd = re.match(r"^\s*", lines[rotary_call_idx]).group(0)
inject_fwd = [
    f"{indent_fwd}# TAOPD_PATCH_V1: apply per-head QK RMSNorm before RoPE\n",
    f"{indent_fwd}if getattr(self, '_use_qk_norm', False):\n",
    f"{indent_fwd}    _q_shape = q.shape\n",
    f"{indent_fwd}    _k_shape = k.shape\n",
    f"{indent_fwd}    if q.dim() == 2:\n",
    f"{indent_fwd}        q = q.view(q.shape[0], -1, self.head_dim)\n",
    f"{indent_fwd}    if k.dim() == 2:\n",
    f"{indent_fwd}        k = k.view(k.shape[0], -1, self.head_dim)\n",
    f"{indent_fwd}    q = _taopd_per_head_rmsnorm(q, self.q_norm, self._qk_norm_eps)\n",
    f"{indent_fwd}    k = _taopd_per_head_rmsnorm(k, self.k_norm, self._qk_norm_eps)\n",
    f"{indent_fwd}    q = q.view(_q_shape)\n",
    f"{indent_fwd}    k = k.view(_k_shape)\n",
]
lines = lines[:rotary_call_idx] + inject_fwd + lines[rotary_call_idx:]

# ── helper: 找 "param = params_dict[name]" 所在行 (load_weights) ──
params_idx = -1
for i, line in enumerate(lines):
    if re.match(r"^\s*param\s*=\s*params_dict\[name\]\s*$", line):
        params_idx = i
        break
assert params_idx >= 0, "找不到 'param = params_dict[name]' (load_weights)"
indent_lw = re.match(r"^\s*", lines[params_idx]).group(0)
inject_lw = [
    f"{indent_lw}# TAOPD_PATCH_V1: 宽容 load_weights (Qwen3 多的 q_norm/k_norm 直接 load 到 self)\n",
    f"{indent_lw}if name not in params_dict:\n",
    f"{indent_lw}    _parts = name.split('.')\n",
    f"{indent_lw}    _obj = self\n",
    f"{indent_lw}    _found = True\n",
    f"{indent_lw}    for _p in _parts[:-1]:\n",
    f"{indent_lw}        if hasattr(_obj, _p):\n",
    f"{indent_lw}            _obj = getattr(_obj, _p)\n",
    f"{indent_lw}        else:\n",
    f"{indent_lw}            _found = False\n",
    f"{indent_lw}            break\n",
    f"{indent_lw}    if _found and hasattr(_obj, _parts[-1]):\n",
    f"{indent_lw}        _attr = getattr(_obj, _parts[-1])\n",
    f"{indent_lw}        if isinstance(_attr, torch.nn.Parameter):\n",
    f"{indent_lw}            with torch.no_grad():\n",
    f"{indent_lw}                _attr.copy_(loaded_weight)\n",
    f"{indent_lw}    continue\n",
]
lines = lines[:params_idx] + inject_lw + lines[params_idx:]

open(p, 'w').writelines(lines)
print(f"✅ patch 成功: {p}")
print(f"   插入点:")
print(f"     - import helper (after last import)")
print(f"     - __init__ q_norm/k_norm (after self.rotary_emb = ... @ line ~{rotary_assign_idx})")
print(f"     - forward per-head norm (before self.rotary_emb(positions, ...) @ line ~{rotary_call_idx})")
print(f"     - load_weights 宽容 (before param = params_dict[name] @ line ~{params_idx})")
PY

# ── 5) 语法验证 (确保 patch 后 qwen2.py 能被 import) ────────────
echo ""
echo "=== 语法验证 ==="
if python3 -c "import ast; ast.parse(open('${QWEN2_PY}').read()); print('  ✅ ast.parse OK')" 2>&1; then
  :
else
  echo "  ❌ qwen2.py 语法错误, 还原 .orig"
  cp "${QWEN2_PY}.orig" "$QWEN2_PY"
  exit 1
fi

if python3 -c "import sglang.srt.models.qwen2; print('  ✅ sglang.srt.models.qwen2 import OK')" 2>&1; then
  :
else
  echo "  ❌ sglang.srt.models.qwen2 import 失败, 还原 .orig"
  cp "${QWEN2_PY}.orig" "$QWEN2_PY"
  exit 1
fi

if grep -q "TAOPD_PATCH_V1" "$QWEN2_PY"; then
  echo "  ✅ TAOPD_PATCH_V1 标记存在"
else
  echo "  ❌ TAOPD_PATCH_V1 标记缺失"; exit 1
fi

echo ""
echo "========================================="
echo " 下一步: bash repro/run_all.sh 5"
echo ""
echo " 还原: cp ${QWEN2_PY}.orig ${QWEN2_PY}"
echo "========================================="
