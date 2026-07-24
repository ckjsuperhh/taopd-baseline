#!/usr/bin/env bash
# 诊断 step 3 转换脚本的真实错误 (绕过 torchrun, 直接跑单进程, 拿到完整 traceback)
# 用法: bash repro/diag_step3_convert.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

# 安全地显示前 N 行 (避免 pipefail + head SIGPIPE)
head_safe() {
  python3 -c "
import sys
n = int(sys.argv[1])
for i, line in enumerate(sys.stdin):
    if i >= n: break
    sys.stdout.write(line)
" "$1"
}

export PYTHONPATH="${MEGATRON_LM_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
ENV_PREFIX="$(python3 -c 'import sys; print(sys.prefix)')"
TORCH_LIB="$(python3 -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
TORCH_CUDA_LIB="$(get_torch_cuda_lib)"
CONDA_LIB="$(get_conda_lib)"
export LD_LIBRARY_PATH="${CONDA_LIB}:${TORCH_LIB}:${TORCH_CUDA_LIB}:${LD_LIBRARY_PATH:-}"

echo "=== Paths ==="
echo "STUDENT_HF        = ${STUDENT_HF}"
echo "STUDENT_TORCH_DIST= ${STUDENT_TORCH_DIST}"
echo "MEGATRON_LM_DIR   = ${MEGATRON_LM_DIR}"
echo "SLIME_DIR         = ${SLIME_DIR}"
echo ""

echo "=== [1] 学生模型文件 ==="
ls "${STUDENT_HF}" 2>&1 | head_safe 15
echo ""

echo "=== [2] 关键 import 测试 ==="
python3 -c "
mods = [
    'slime_plugins.mbridge',
    'mbridge',
    'mbridge.core',
    'mbridge.models',
    'megatron',
    'megatron.core',
    'megatron.bridge',
    'megatron.bridge.models',
]
for m in mods:
    try:
        __import__(m)
        print(f'  ✅ {m}')
    except Exception as e:
        print(f'  ❌ {m}: {type(e).__name__}: {e}')
"
echo ""

echo "=== [3] 加载 qwen3-1.7B 模型参数 ==="
cd "${SLIME_DIR}"
set +e
source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"
set -e
echo "MODEL_ARGS (${#MODEL_ARGS[@]} 项):"
printf '  %s\n' "${MODEL_ARGS[@]}" | head_safe 30
echo ""

echo "=== [4] convert 脚本 help ==="
python3 "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" --help 2>&1 | head_safe 40 || true
echo ""

echo "=== [5] 单进程跑转换 (完整 traceback) ==="
export CUDA_VISIBLE_DEVICES=0
set +e
python3 "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
  "${MODEL_ARGS[@]}" \
  --no-rope-fusion \
  --transformer-impl local \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --hf-checkpoint "${STUDENT_HF}" \
  --save "${STUDENT_TORCH_DIST}" \
  2>&1 | tee /tmp/step3_convert.log
CONV_RC=${PIPESTATUS[0]}
set -e
echo ""
echo "(exit code: ${CONV_RC})"
echo ""

echo "=== [6] 完整错误堆栈 (最后 60 行) ==="
tail -60 /tmp/step3_convert.log
echo ""
echo "========================================="
echo "日志: /tmp/step3_convert.log"
echo "把 [2]/[5]/[6] 的完整输出贴给 QoderCN"
echo "========================================="
