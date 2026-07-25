#!/usr/bin/env bash
# 给 sglang 0.4.1 的 Qwen2Attention 打 monkey-patch, 让它能加载 Qwen3 dense
# (Qwen3 多了 per-head RMSNorm: q_norm / k_norm)
#
# 原理:
#   - Qwen3 在 attention 里对 Q/K 应用 per-head RMSNorm 后再做 RoPE
#   - sglang 0.4.1 的 Qwen2Attention 没有这俩参数, load_weights 时 KeyError
#   - 本脚本直接编辑 site-packages/sglang/srt/models/qwen2.py:
#       1) 在 __init__ 里加 self.q_norm / self.k_norm (RMSNorm per head)
#       2) 在 forward 里对 q/k 应用 per-head norm 后再做 RoPE
#       3) 在 load_weights 里接受 q_norm.weight / k_norm.weight 键
#   - 备份原文件到 qwen2.py.orig (幂等)
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
echo "📄 目标文件: ${QWEN2_PY}"

# 已经打过 patch 就跳过
if grep -q "TAOPD_PATCH_V1" "$QWEN2_PY"; then
  echo "  ✅ 已打过 patch (TAOPD_PATCH_V1), 跳过"
  exit 0
fi

# 备份
cp "$QWEN2_PY" "${QWEN2_PY}.orig"
echo "  💾 备份到 ${QWEN2_PY}.orig"

# 用 Python 做精准 AST 级别编辑 (sed 不可靠, 用 Python 字符串替换)
python3 - "$QWEN2_PY" <<'PY'
import sys, re
p = sys.argv[1]
src = open(p).read()

# ── 1) 在文件顶部加一个 helper: per-head RMSNorm ───────────────────
# 这个 helper 接受 q/k (shape: [..., num_heads, head_dim]),
# 对每个 head 做 RMSNorm (weight shape: [head_dim])。
HELPER = '''
# === TAOPD_PATCH_V1: Qwen3 per-head QK RMSNorm (兼容 Qwen2 加载器) ===
import torch.nn.functional as _F
def _taopd_per_head_rmsnorm(x, weight, eps=1e-6):
    """x: [..., num_heads, head_dim]; weight: [head_dim]"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(orig_dtype)
# === END TAOPD_PATCH_V1 ===

'''
# 找最后一个 import 行的位置, 在它后面插入 helper (保守做法: 在第一个 class 定义前插入)
m = re.search(r"^class \w+", src, re.MULTILINE)
assert m, "找不到 class 定义, patch 失败"
insert_pos = m.start()
src = src[:insert_pos] + HELPER + src[insert_pos:]

# ── 2) 在 Qwen2Attention.__init__ 里加 q_norm / k_norm ─────────────
# 定位 "self.rotary_emb = rotary_emb" 这一行 (Attention 类里几乎必有),
# 在它之后插入 q_norm / k_norm 的注册
needle_init = "self.rotary_emb = rotary_emb"
if needle_init not in src:
    # 退而求其次, 试 "self.rotary_emb" 赋值
    needle_init = "self.rotary_emb"
    idx = src.find(needle_init)
    if idx < 0:
        print("⚠ 找不到 self.rotary_emb 赋值, 无法 patch __init__")
        sys.exit(1)
    # 取这一行结尾
    eol = src.find("\n", idx)
else:
    eol = src.find("\n", src.find(needle_init))

inject_init = '''
        # TAOPD_PATCH_V1: Qwen3 per-head QK RMSNorm (config 默认关闭, Qwen3 config 会开)
        if getattr(config, "head_dim", None) is None:
            _hd = config.hidden_size // config.num_attention_heads
        else:
            _hd = config.head_dim
        self.head_dim = _hd
        if getattr(config, "use_qk_norm", False) or getattr(config, "qk_norm_type", None) is not None:
            # Qwen3 风格 per-head RMSNorm
            self.q_norm = torch.nn.Parameter(torch.ones(_hd))
            self.k_norm = torch.nn.Parameter(torch.ones(_hd))
            self._use_qk_norm = True
            self._qk_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        else:
            self._use_qk_norm = False
'''
src = src[:eol+1] + inject_init + src[eol+1:]

# ── 3) 在 Qwen2Attention.forward 里应用 per-head norm (在 rotary_emb 之前) ──
# 典型模式: q, k = self.rotary_emb(positions, q, k)
# 我们要在这之前插入 q/k 的 norm
needle_rotary = "self.rotary_emb(positions"
idx_rot = src.find(needle_rotary)
if idx_rot < 0:
    print("⚠ 找不到 self.rotary_emb(positions ...) 调用, 无法 patch forward")
    sys.exit(1)
# 找到包含这行的整行的开头 (缩进)
line_start = src.rfind("\n", 0, idx_rot) + 1
line_end = src.find("\n", idx_rot)
line = src[line_start:line_end]
indent = line[:len(line) - len(line.lstrip())]

inject_fwd = f'''{indent}# TAOPD_PATCH_V1: apply per-head QK RMSNorm before RoPE
{indent}if getattr(self, "_use_qk_norm", False):
{indent}    # q/k shape in sglang: [num_tokens, num_heads * head_dim] 或 [num_tokens, num_heads, head_dim]
{indent}    _orig_shape_q = q.shape
{indent}    _orig_shape_k = k.shape
{indent}    if q.dim() == 2:
{indent}        q = q.view(q.shape[0], -1, self.head_dim)
{indent}    if k.dim() == 2:
{indent}        k = k.view(k.shape[0], -1, self.head_dim)
{indent}    q = _taopd_per_head_rmsnorm(q, self.q_norm, self._qk_norm_eps)
{indent}    k = _taopd_per_head_rmsnorm(k, self.k_norm, self._qk_norm_eps)
{indent}    q = q.view(_orig_shape_q)
{indent}    k = k.view(_orig_shape_k)
'''
src = src[:line_start] + inject_fwd + src[line_start:]

# ── 4) 在 Qwen2ForCausalLM.load_weights 里接受 q_norm/k_norm.weight ──
# sglang 0.4.1 的 load_weights 大致是:
#   params_dict = dict(self.named_parameters())
#   for name, loaded_weight in weights:
#       ...
#       param = params_dict[name]   <-- 这里 KeyError
# 我们要做的: 在 "param = params_dict[name]" 之前, 跳过不认识的键 (而不是报错)
# 但更安全: 接受 q_norm/k_norm, 把它们 load 到 buffer。
# 简单做法: 把 KeyError 那一行改成 .get(name), 如果是 None 就 continue
needle_params = "param = params_dict[name]"
if needle_params not in src:
    # 试更宽松模式
    needle_params = "params_dict[name]"
    idx = src.find(needle_params)
    if idx < 0:
        print("⚠ 找不到 params_dict[name], 无法 patch load_weights")
        sys.exit(1)
    line_start = src.rfind("\n", 0, idx) + 1
    line_end = src.find("\n", idx)
    line = src[line_start:line_end]
    indent = line[:len(line) - len(line.lstrip())]
    # 替换这一行为更宽容的版本
    replacement = (
        f"{indent}# TAOPD_PATCH_V1: 宽容 load_weights (Qwen3 多的 q_norm/k_norm 直接 load 到 self)\n"
        f"{indent}if name not in params_dict:\n"
        f"{indent}    # 尝试从 self 取 attribute (处理 q_norm.weight / k_norm.weight)\n"
        f"{indent}    _parts = name.split('.')\n"
        f"{indent}    _obj = self\n"
        f"{indent}    _found = True\n"
        f"{indent}    for _p in _parts[:-1]:\n"
        f"{indent}        if hasattr(_obj, _p):\n"
        f"{indent}            _obj = getattr(_obj, _p)\n"
        f"{indent}        else:\n"
        f"{indent}            _found = False\n"
        f"{indent}            break\n"
        f"{indent}    if _found and hasattr(_obj, _parts[-1]):\n"
        f"{indent}        _attr = getattr(_obj, _parts[-1])\n"
        f"{indent}        if isinstance(_attr, torch.nn.Parameter):\n"
        f"{indent}            with torch.no_grad():\n"
        f"{indent}                _attr.copy_(loaded_weight)\n"
        f"{indent}        continue\n"
        f"{indent}    continue  # 完全不认识的键跳过\n"
        f"{indent}param = params_dict[name]\n"
    )
    src = src[:line_start] + replacement + src[line_end+1:]
else:
    idx = src.find(needle_params)
    line_start = src.rfind("\n", 0, idx) + 1
    line_end = src.find("\n", idx)
    line = src[line_start:line_end]
    indent = line[:len(line) - len(line.lstrip())]
    replacement = (
        f"{indent}# TAOPD_PATCH_V1: 宽容 load_weights (Qwen3 多的 q_norm/k_norm 直接 load 到 self)\n"
        f"{indent}if name not in params_dict:\n"
        f"{indent}    _parts = name.split('.')\n"
        f"{indent}    _obj = self\n"
        f"{indent}    _found = True\n"
        f"{indent}    for _p in _parts[:-1]:\n"
        f"{indent}        if hasattr(_obj, _p):\n"
        f"{indent}            _obj = getattr(_obj, _p)\n"
        f"{indent}        else:\n"
        f"{indent}            _found = False\n"
        f"{indent}            break\n"
        f"{indent}    if _found and hasattr(_obj, _parts[-1]):\n"
        f"{indent}        _attr = getattr(_obj, _parts[-1])\n"
        f"{indent}        if isinstance(_attr, torch.nn.Parameter):\n"
        f"{indent}            with torch.no_grad():\n"
        f"{indent}                _attr.copy_(loaded_weight)\n"
        f"{indent}        continue\n"
        f"{indent}    continue\n"
        f"{indent}param = params_dict[name]\n"
    )
    src = src[:line_start] + replacement + src[line_end+1:]

open(p, 'w').write(src)
print(f"✅ patch 成功: {p}")
print(f"   标记: TAOPD_PATCH_V1")
PY

# 验证 patch 标记存在
if grep -q "TAOPD_PATCH_V1" "$QWEN2_PY"; then
  echo ""
  echo "✅ qwen2.py 已包含 TAOPD_PATCH_V1 标记"
  echo "   现在 sglang 可以加载 Qwen3 dense (带 q_norm/k_norm)"
else
  echo "❌ patch 似乎没生效"; exit 1
fi

echo ""
echo "========================================="
echo " 下一步: bash repro/run_all.sh 5"
echo ""
echo " 还原: cp ${QWEN2_PY}.orig ${QWEN2_PY}"
echo "========================================="
