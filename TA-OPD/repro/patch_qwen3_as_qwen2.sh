#!/usr/bin/env bash
# Patch Qwen3 dense 模型的 config.json, 让 sglang 0.4.1 把它当成 Qwen2 加载
# 原理: Qwen3 dense (4B/1.7B/0.6B 等) 的模型结构和 Qwen2 完全一样,
# 只是 transformers 给了新的 model_type="qwen3" / architectures=["Qwen3ForCausalLM"]。
# sglang 0.4.1 (2024-11) 早于 Qwen3 发布, 不知道 Qwen3ForCausalLM,
# 但认识 Qwen2ForCausalLM。改 config.json 即可绕过。
#
# 注意: 这只对 dense Qwen3 模型有效。Qwen3 MoE (30B-A3B 等) 不能用,
# 因为 Qwen2MoE 结构和 Qwen3 MoE 有细微差异 (expert 数量 / router 等)。
#
# 用法: bash repro/patch_qwen3_as_qwen2.sh
set -eo pipefail  # 不能用 -u:conda 内部 activate 脚本有 unbound variable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

patch_one() {
  local model_dir="$1"
  local cfg="${model_dir}/config.json"
  if [[ ! -f "$cfg" ]]; then
    echo "  ⚠ 找不到 ${cfg}, 跳过"
    return 0
  fi
  echo "  📄 ${cfg}"
  python3 - "$cfg" <<'PY'
import json, sys, shutil
p = sys.argv[1]
with open(p) as f:
    c = json.load(f)
orig_type = c.get('model_type')
orig_arch = (c.get('architectures') or [None])[0]
changed = False

# Qwen3 dense → Qwen2
if c.get('model_type') == 'qwen3':
    c['model_type'] = 'qwen2'
    changed = True
if c.get('architectures') == ['Qwen3ForCausalLM']:
    c['architectures'] = ['Qwen2ForCausalLM']
    changed = True

# Qwen3 MoE → Qwen2MoE (谨慎, 不保证完全等价, 但 sglang 0.4.1 认识 Qwen2MoeForCausalLM)
if c.get('model_type') == 'qwen3_moe':
    c['model_type'] = 'qwen2_moe'
    changed = True
if c.get('architectures') == ['Qwen3MoeForCausalLM']:
    c['architectures'] = ['Qwen2MoeForCausalLM']
    changed = True

if changed:
    shutil.copy(p, p + '.orig')
    with open(p, 'w') as f:
        json.dump(c, f, indent=2)
    print(f'    ✅ patched: model_type={orig_type}→{c["model_type"]}, arch={orig_arch}→{c["architectures"][0]}')
    print(f'    💾 原配置已备份到 {p}.orig')
else:
    print(f'    (已经是 Qwen2 / 非 Qwen3, 无需 patch)')
PY
}

echo "=== Patch Qwen3 dense models → Qwen2 (for sglang 0.4.1) ==="
echo "MODEL_DIR = ${MODEL_DIR}"

# 两个 dense 模型都要 patch (teacher + student)
for M in "${TEACHER_MODEL}" "${STUDENT_HF}"; do
  echo ""
  echo "--- ${M} ---"
  patch_one "${M}"
done

echo ""
echo "=== 验证 config.json ==="
for M in "${TEACHER_MODEL}" "${STUDENT_HF}"; do
  echo "--- ${M} ---"
  python3 -c "
import json
c = json.load(open('${M}/config.json'))
print(f'  model_type   = {c.get(\"model_type\")}')
print(f'  architectures = {c.get(\"architectures\")}')
"
done

echo ""
echo "========================================="
echo " 下一步: bash repro/run_all.sh 5"
echo ""
echo " 注: 想还原时把 .orig 覆盖回去:"
echo "     cp config.json.orig config.json"
echo "========================================="
