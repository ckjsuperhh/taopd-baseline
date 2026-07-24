#!/usr/bin/env bash
# 降级 numpy 到 1.x (Megatron 硬性要求: np.__version__.startswith("1."))
# 用法: bash repro/fix_numpy.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

echo "=== 当前 numpy ==="
python3 -c "import numpy; print(f'numpy {numpy.__version__}')"
echo ""

echo "=== 降级 numpy==1.26.4 ==="
pip install "numpy==1.26.4"
echo ""

echo "=== 验证 ==="
python3 -c "
import numpy as np
print(f'numpy {np.__version__}')
assert np.__version__.startswith('1.'), f'FAIL: {np.__version__}'
print('✅ numpy 1.x, Megatron 能过 assert')
"
echo ""

echo "========================================="
echo " 继续: bash repro/run_all.sh 3"
echo "========================================="
